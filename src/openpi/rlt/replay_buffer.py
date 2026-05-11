"""Off-policy replay buffer for chunk-level RL transitions.

Stores (z_rl, state, action_chunk, ref_action_chunk, rewards, next_z_rl, next_state, done)
transitions. Supports stride-2 subsampling to increase data efficiency from each episode.

Thread-safe add/sample via an internal lock, so a background learner thread can draw
batches while the rollout thread appends new transitions (paper Sec. "Update": async
rollouts and learning with UTD=5).
"""

from __future__ import annotations

import dataclasses
import threading

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
        self._lock = threading.Lock()

        self.z_rl = np.zeros((capacity, z_rl_dim), dtype=np.float32)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, chunk_length, action_dim), dtype=np.float32)
        self.ref_actions = np.zeros((capacity, chunk_length, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, chunk_length), dtype=np.float32)
        self.next_z_rl = np.zeros((capacity, z_rl_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, transition: Transition) -> None:
        """Add a single transition to the buffer (thread-safe)."""
        with self._lock:
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
        """Sample a random batch of transitions (thread-safe).

        Returns a dict of tensors on the specified device with keys:
        z_rl, state, action, ref_action, reward, next_z_rl, next_state, done.
        """
        # Snapshot indices + per-field slices under the lock so a concurrent
        # writer can't tear a row mid-read. Torch tensors are materialized after
        # the lock is released to minimize contention with the rollout thread.
        with self._lock:
            size = self.size
            idxs = np.random.randint(0, size, size=batch_size)
            z_rl_np = self.z_rl[idxs].copy()
            states_np = self.states[idxs].copy()
            actions_np = self.actions[idxs].copy()
            ref_actions_np = self.ref_actions[idxs].copy()
            rewards_np = self.rewards[idxs].copy()
            next_z_rl_np = self.next_z_rl[idxs].copy()
            next_states_np = self.next_states[idxs].copy()
            dones_np = self.dones[idxs].copy()
        return {
            "z_rl": torch.as_tensor(z_rl_np, device=device),
            "state": torch.as_tensor(states_np, device=device),
            "action": torch.as_tensor(actions_np, device=device),
            "ref_action": torch.as_tensor(ref_actions_np, device=device),
            "reward": torch.as_tensor(rewards_np, device=device),
            "next_z_rl": torch.as_tensor(next_z_rl_np, device=device),
            "next_state": torch.as_tensor(next_states_np, device=device),
            "done": torch.as_tensor(dones_np, device=device),
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
    ) -> int:
        """Add overlapping chunk transitions from a sequence with stride subsampling.

        Given a sequence of T timesteps of executed actions (with corresponding
        observations), creates chunk transitions starting at every ``stride`` steps,
        **plus** the terminal chunk starting at ``T - C`` so ``dones[T-1]`` and any
        reward on the last step are always represented (when ``T > C``).

        Args:
            z_rls: [T, z_rl_dim] RL tokens at each timestep.
            states: [T, state_dim] proprioceptive states at each timestep.
            actions: [T, action_dim] executed actions at each timestep.
            ref_actions: [T, action_dim] VLA reference actions at each timestep.
            rewards: [T] per-step rewards.
            dones: [T] done flags.

        Returns:
            Number of transitions added.
        """
        T = z_rls.shape[0]
        C = self.actions.shape[1]  # chunk_length

        if T <= C:
            return 0

        # Stride subsampling over overlapping chunk starts. Always include the
        # terminal chunk start ``T - C`` so the timestep with ``dones[T-1]==True``
        # and any terminal reward appear in the buffer. The old ``range(0, T-C, stride)``
        # never visited ``T - C`` (off-by-one stop) and also skipped it whenever
        # ``(T-C) % stride != 0``, so successful-episode reward was often missing
        # from every transition of that episode.
        term_start = T - C
        chunk_starts = set(range(0, term_start + 1, stride))
        chunk_starts.add(term_start)

        added = 0
        for start in sorted(chunk_starts):
            end = start + C
            # Next observation is at end (clamped to last valid index).
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
            added += 1
        return added
