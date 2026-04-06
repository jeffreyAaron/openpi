"""Lightweight actor-critic networks for Stage 2 online RL.

Implements the Gaussian actor (Eq. 4-5) and TD3-style twin Q-critic (Eq. 3)
from the RLT paper. Both are small MLPs operating on the RL token representation.
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

    The actor takes the RL token, proprioceptive state, and a reference action
    chunk from the VLA, and outputs a Gaussian mean over the flattened action
    chunk. During training, reference actions are dropped out 50% of the time
    (replaced with zeros) to prevent copying.

    Args:
        z_rl_dim: Dimension of the RL token z_rl.
        state_dim: Dimension of proprioceptive state s_p.
        action_chunk_dim: Flattened action chunk dimension (C * d).
        hidden_dim: MLP hidden layer width.
        num_layers: Number of hidden layers.
        fixed_std: Fixed standard deviation for the Gaussian.
    """

    def __init__(
        self,
        z_rl_dim: int = 2048,
        state_dim: int = 16,
        action_chunk_dim: int = 160,
        hidden_dim: int = 256,
        num_layers: int = 2,
        fixed_std: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = z_rl_dim + state_dim + action_chunk_dim
        self.mlp = _build_mlp(input_dim, hidden_dim, action_chunk_dim, num_layers)
        self.fixed_std = fixed_std

    def forward(
        self, z_rl: Tensor, state: Tensor, ref_actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute action mean and sample.

        Args:
            z_rl: [B, z_rl_dim] RL token.
            state: [B, state_dim] proprioceptive state.
            ref_actions: [B, C*d] flattened VLA reference chunk (or zeros if dropped).

        Returns:
            sampled: [B, C*d] sampled actions (mean + noise).
            mean: [B, C*d] action mean (deterministic component).
        """
        x = torch.cat([z_rl, state, ref_actions], dim=-1)
        mean = self.mlp(x)
        noise = torch.randn_like(mean) * self.fixed_std
        return mean + noise, mean


class TwinQCritic(nn.Module):
    """TD3-style twin Q-function for action-chunk evaluation.

    Two independent Q-networks that estimate Q(x, a_1:C) where x = (z_rl, s_p).
    Uses the minimum of both for target computation to reduce overestimation.

    Args:
        z_rl_dim: Dimension of the RL token z_rl.
        state_dim: Dimension of proprioceptive state s_p.
        action_chunk_dim: Flattened action chunk dimension (C * d).
        hidden_dim: MLP hidden layer width.
        num_layers: Number of hidden layers.
    """

    def __init__(
        self,
        z_rl_dim: int = 2048,
        state_dim: int = 16,
        action_chunk_dim: int = 160,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        input_dim = z_rl_dim + state_dim + action_chunk_dim
        self.q1 = _build_mlp(input_dim, hidden_dim, 1, num_layers)
        self.q2 = _build_mlp(input_dim, hidden_dim, 1, num_layers)

    def forward(
        self, z_rl: Tensor, state: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute both Q-values.

        Args:
            z_rl: [B, z_rl_dim] RL token.
            state: [B, state_dim] proprioceptive state.
            actions: [B, C*d] flattened action chunk.

        Returns:
            q1: [B] Q-value from first network.
            q2: [B] Q-value from second network.
        """
        x = torch.cat([z_rl, state, actions], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q_min(self, z_rl: Tensor, state: Tensor, actions: Tensor) -> Tensor:
        """Compute min(Q1, Q2) for conservative target estimates.

        Returns:
            q_min: [B] element-wise minimum of both Q-values.
        """
        q1, q2 = self.forward(z_rl, state, actions)
        return torch.min(q1, q2)


def create_target_critic(critic: TwinQCritic) -> TwinQCritic:
    """Create a target critic as a frozen deep copy."""
    target = copy.deepcopy(critic)
    for p in target.parameters():
        p.requires_grad_(False)
    return target


def soft_update_target(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging: target = tau * source + (1 - tau) * target."""
    for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
        tp.data.lerp_(sp.data, tau)
