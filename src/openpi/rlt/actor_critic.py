"""Lightweight actor-critic networks for Stage 2 online RL.

Implements the Gaussian actor (Eq. 4-5) and an N-head Q-critic ensemble
(article-style generalization of the paper's TD3 twin critic) from the RLT paper.
Both are small MLPs operating on the RL token representation.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch import Tensor


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
    """Build a simple MLP with ReLU activations."""
    layers: list[nn.Module] = []
    for i in range(num_layers):
        in_d = input_dim if i == 0 else hidden_dim
        layers.append(nn.Linear(in_d, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Gaussian policy that refines VLA reference action chunks.

    pi_theta(a | x, a_tilde) = N(mu_theta(x, a_tilde), sigma^2 I)

    The actor takes the RL token, proprioceptive state, and a VLA reference
    context (which may be longer than the output chunk — see
    ``ref_context_chunk_dim`` — so the actor can "compress" a longer horizon into
    its shorter emitted chunk, per the article's h_context > h_chunk idea).
    The output dimension stays at ``action_chunk_dim`` (= C * d).

    Args:
        z_rl_dim: Dimension of the RL token z_rl.
        state_dim: Dimension of proprioceptive state s_p.
        action_chunk_dim: Flattened OUTPUT chunk dimension (C * d).
        ref_context_chunk_dim: Flattened VLA reference context dim fed in.
            Defaults to ``action_chunk_dim`` (no extended context). Must be >=
            ``action_chunk_dim`` for the "extended context" setting.
        hidden_dim: MLP hidden layer width.
        num_layers: Number of hidden layers.
        fixed_std: Fixed standard deviation for the Gaussian.
    """

    def __init__(
        self,
        z_rl_dim: int = 2048,
        state_dim: int = 16,
        action_chunk_dim: int = 160,
        ref_context_chunk_dim: int | None = None,
        hidden_dim: int = 256,
        num_layers: int = 2,
        fixed_std: float = 0.1,
    ) -> None:
        super().__init__()
        if ref_context_chunk_dim is None:
            ref_context_chunk_dim = action_chunk_dim
        if ref_context_chunk_dim < action_chunk_dim:
            raise ValueError(
                f"ref_context_chunk_dim ({ref_context_chunk_dim}) must be >= "
                f"action_chunk_dim ({action_chunk_dim})"
            )
        self.action_chunk_dim = action_chunk_dim
        self.ref_context_chunk_dim = ref_context_chunk_dim
        input_dim = z_rl_dim + state_dim + ref_context_chunk_dim
        self.mlp = _build_mlp(input_dim, hidden_dim, action_chunk_dim, num_layers)
        self.fixed_std = fixed_std

    def forward(
        self, z_rl: Tensor, state: Tensor, ref_actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute action mean and sample.

        Args:
            z_rl: [B, z_rl_dim] RL token.
            state: [B, state_dim] proprioceptive state.
            ref_actions: [B, ref_context_chunk_dim] flattened VLA reference context
                (may be longer than the output chunk; zero-padded if shorter).

        Returns:
            sampled: [B, action_chunk_dim] sampled actions (mean + noise).
            mean: [B, action_chunk_dim] action mean (deterministic component).
        """
        mean = self.forward_mean(z_rl, state, ref_actions)
        noise = torch.randn_like(mean) * self.fixed_std
        return mean + noise, mean

    def forward_mean(self, z_rl: Tensor, state: Tensor, ref_actions: Tensor) -> Tensor:
        """Deterministic action mean μ(x, a_ref) without exploration noise."""
        x = torch.cat([z_rl, state, ref_actions], dim=-1)
        return self.mlp(x)


class MultiQCritic(nn.Module):
    """Ensemble of ``num_critics`` independent Q-networks for action-chunk evaluation.

    Generalizes the paper's TD3-style twin critic. The article uses 4 heads
    (doubled from 2) alongside gradient clipping to reduce reward-hacking Q-spikes.
    ``q_min`` returns the elementwise minimum across all heads for conservative
    target bootstrapping, and ``forward`` returns the full list for per-head MSE.

    Args:
        z_rl_dim: Dimension of the RL token z_rl.
        state_dim: Dimension of proprioceptive state s_p.
        action_chunk_dim: Flattened action chunk dimension (C * d).
        hidden_dim: MLP hidden layer width.
        num_layers: Number of hidden layers.
        num_critics: Number of independent Q-heads (article uses 4).
    """

    def __init__(
        self,
        z_rl_dim: int = 2048,
        state_dim: int = 16,
        action_chunk_dim: int = 160,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_critics: int = 4,
    ) -> None:
        super().__init__()
        if num_critics < 2:
            raise ValueError(f"num_critics must be >= 2, got {num_critics}")
        self.num_critics = num_critics
        input_dim = z_rl_dim + state_dim + action_chunk_dim
        self.qs = nn.ModuleList(
            [_build_mlp(input_dim, hidden_dim, 1, num_layers) for _ in range(num_critics)]
        )

    def forward(self, z_rl: Tensor, state: Tensor, actions: Tensor) -> list[Tensor]:
        """Compute all Q-values.

        Args:
            z_rl: [B, z_rl_dim] RL token.
            state: [B, state_dim] proprioceptive state.
            actions: [B, C*d] flattened action chunk.

        Returns:
            List of ``num_critics`` tensors of shape [B], one per head.
        """
        x = torch.cat([z_rl, state, actions], dim=-1)
        return [q(x).squeeze(-1) for q in self.qs]

    def q_min(self, z_rl: Tensor, state: Tensor, actions: Tensor) -> Tensor:
        """Elementwise min over all heads (conservative target estimate)."""
        qs = self.forward(z_rl, state, actions)
        stacked = torch.stack(qs, dim=0)  # [num_critics, B]
        return stacked.min(dim=0).values  # [B]


# Backwards-compat alias. Older checkpoints / call sites expect ``TwinQCritic`` with
# ``.q1`` / ``.q2`` attributes, but new code should use ``MultiQCritic`` directly.
TwinQCritic = MultiQCritic


def create_target_critic(critic: MultiQCritic) -> MultiQCritic:
    """Create a target critic as a frozen deep copy."""
    target = copy.deepcopy(critic)
    for p in target.parameters():
        p.requires_grad_(False)
    return target


def soft_update_target(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging: target = tau * source + (1 - tau) * target."""
    for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
        tp.data.lerp_(sp.data, tau)
