"""Off-policy replay buffer for chunk-level RL transitions.

Stores (z_rl, state, action_chunk, ref_action_chunk, rewards, next_z_rl, next_state, done)
transitions. Supports stride-2 subsampling to increase data efficiency from each episode.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from torch import Tensor


@dataclasses.dataclass
class Transition:
    """A single chunk-level transition for the replay buffer."""

    z_rl: np.ndarray  # [z_rl_dim]
    state: np.ndarray  # [state_dim]
    action_chunk: np.ndarray  # [C, action_dim]
    ref_action_chunk: np.ndarray  # [C, action_dim]
    rewards: np.ndarray  # [C] per-step rewards within chunk
    next_z_rl: np.ndarray  # [z_rl_dim]
    next_state: np.ndarray  # [state_dim]
    done: bool


class ReplayBuffer:
    """Fixed-capacity ring buffer for off-policy RL with action chunks.

    Pre-allocates numpy arrays for memory efficiency. Supports stride-2
    subsampling from episode data to increase sample count.

    Args:
        capacity: Maximum number of transitions.
        chunk_length: RL chunk length C.
        action_dim: Per-timestep action dimension d.
        z_rl_dim: RL token dimension.
        state_dim: Proprioceptive state dimension.
    """

    def __init__(
        self,
        capacity: int,
        chunk_length: int = 10,
        action_dim: int = 16,
        z_rl_dim: int = 2048,
        state_dim: int = 16,
    ) -> None:
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

        self.z_rl = np.zeros((capacity, z_rl_dim), dtype=np.float32)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, chunk_length, action_dim), dtype=np.float32)
        self.ref_actions = np.zeros((capacity, chunk_length, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, chunk_length), dtype=np.float32)
        self.next_z_rl = np.zeros((capacity, z_rl_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, transition: Transition) -> None:
        """Add a single transition to the buffer."""
        idx = self.ptr % self.capacity
        self.z_rl[idx] = transition.z_rl
        self.states[idx] = transition.state
        self.actions[idx] = transition.action_chunk
        self.ref_actions[idx] = transition.ref_action_chunk
        self.rewards[idx] = transition.rewards
        self.next_z_rl[idx] = transition.next_z_rl
        self.next_states[idx] = transition.next_state
        self.dones[idx] = float(transition.done)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, Tensor]:
        """Sample a random batch of transitions.

        Returns a dict of tensors on the specified device with keys:
        z_rl, state, action, ref_action, reward, next_z_rl, next_state, done.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "z_rl": torch.as_tensor(self.z_rl[idxs], device=device),
            "state": torch.as_tensor(self.states[idxs], device=device),
            "action": torch.as_tensor(self.actions[idxs], device=device),
            "ref_action": torch.as_tensor(self.ref_actions[idxs], device=device),
            "reward": torch.as_tensor(self.rewards[idxs], device=device),
            "next_z_rl": torch.as_tensor(self.next_z_rl[idxs], device=device),
            "next_state": torch.as_tensor(self.next_states[idxs], device=device),
            "done": torch.as_tensor(self.dones[idxs], device=device),
        }

    def add_chunk_with_subsampling(
        self,
        z_rls: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        ref_actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        stride: int = 2,
    ) -> None:
        """Add overlapping chunk transitions from a sequence with stride-2 subsampling.

        Given a sequence of T timesteps of executed actions (with corresponding
        observations), creates chunk transitions starting at every `stride` steps.

        Args:
            z_rls: [T, z_rl_dim] RL tokens at each timestep.
            states: [T, state_dim] proprioceptive states at each timestep.
            actions: [T, action_dim] executed actions at each timestep.
            ref_actions: [T, action_dim] VLA reference actions at each timestep.
            rewards: [T] per-step rewards.
            dones: [T] done flags.
        """
        T = z_rls.shape[0]
        C = self.actions.shape[1]  # chunk_length

        for start in range(0, T - C, stride):
            end = start + C
            # Next observation is at end (start of next chunk).
            next_idx = min(end, T - 1)
            transition = Transition(
                z_rl=z_rls[start],
                state=states[start],
                action_chunk=actions[start:end],
                ref_action_chunk=ref_actions[start:end],
                rewards=rewards[start:end],
                next_z_rl=z_rls[next_idx],
                next_state=states[next_idx],
                done=bool(dones[end - 1]),
            )
            self.add(transition)
