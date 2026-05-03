"""Off-policy replay buffer(s) for chunk-level RL transitions.

Stores (z_rl, state, action_chunk, ref_action_chunk, rewards, next_z_rl, next_state, done)
transitions. Supports stride-N subsampling and an optional longer reference context
(``ref_context_length > chunk_length``) for the "extended context" actor setting.

``DualReplayBuffer`` provides the article's demo + online two-buffer split, sampling
a configurable fraction of each batch from the demo buffer.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from torch import Tensor


@dataclasses.dataclass
class Transition:
    """A single chunk-level transition for the replay buffer.

    ``ref_action_chunk`` may be longer than ``action_chunk`` when using the
    extended-context actor (article's h_context > h_chunk idea); in that case
    its length is ``ref_context_length``, not the executed chunk length C.
    """

    z_rl: np.ndarray  # [z_rl_dim]
    state: np.ndarray  # [state_dim]
    action_chunk: np.ndarray  # [C, action_dim]
    ref_action_chunk: np.ndarray  # [ref_context_length, action_dim]
    rewards: np.ndarray  # [C] per-step rewards within chunk
    next_z_rl: np.ndarray  # [z_rl_dim]
    next_state: np.ndarray  # [state_dim]
    next_ref_action_chunk: np.ndarray  # [ref_context_length, action_dim] for next_state
    done: bool


class ReplayBuffer:
    """Fixed-capacity ring buffer for off-policy RL with action chunks.

    Pre-allocates numpy arrays for memory efficiency. Supports stride-N
    subsampling from episode data to increase sample count.

    Args:
        capacity: Maximum number of transitions.
        chunk_length: RL chunk length C (length of executed actions / rewards).
        action_dim: Per-timestep action dimension d.
        z_rl_dim: RL token dimension.
        state_dim: Proprioceptive state dimension.
        ref_context_length: Length of the VLA reference context stored per transition.
            Defaults to ``chunk_length`` (no extended context). When larger, the
            first ``chunk_length`` entries overlap the executed chunk and the rest
            are future VLA-proposed actions used only as actor input context.
    """

    def __init__(
        self,
        capacity: int,
        chunk_length: int = 10,
        action_dim: int = 16,
        z_rl_dim: int = 2048,
        state_dim: int = 16,
        ref_context_length: int | None = None,
    ) -> None:
        if ref_context_length is None:
            ref_context_length = chunk_length
        if ref_context_length < chunk_length:
            raise ValueError(
                f"ref_context_length ({ref_context_length}) must be >= chunk_length ({chunk_length})"
            )
        self.capacity = capacity
        self.chunk_length = chunk_length
        self.ref_context_length = ref_context_length
        self.size = 0
        self.ptr = 0

        self.z_rl = np.zeros((capacity, z_rl_dim), dtype=np.float32)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, chunk_length, action_dim), dtype=np.float32)
        self.ref_actions = np.zeros((capacity, ref_context_length, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, chunk_length), dtype=np.float32)
        self.next_z_rl = np.zeros((capacity, z_rl_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_ref_actions = np.zeros(
            (capacity, ref_context_length, action_dim), dtype=np.float32
        )
        self.dones = np.zeros(capacity, dtype=np.float32)

    def clear(self) -> None:
        """Reset the ring buffer in-place (useful for resetting the online buffer)."""
        self.size = 0
        self.ptr = 0

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
        self.next_ref_actions[idx] = transition.next_ref_action_chunk
        self.dones[idx] = float(transition.done)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, Tensor]:
        """Sample a random batch of transitions.

        Returns a dict of tensors on the specified device with keys:
        z_rl, state, action, ref_action, reward, next_z_rl, next_state,
        next_ref_action, done.
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
            "next_ref_action": torch.as_tensor(self.next_ref_actions[idxs], device=device),
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
        segment_end_step: int | None = None,
    ) -> None:
        """Add overlapping chunk transitions from a sequence with stride-N subsampling.

        Given a sequence of T timesteps of executed actions (with corresponding
        observations), creates chunk transitions starting at every `stride` steps.

        Extended context: if the buffer's ``ref_context_length > chunk_length``,
        each transition stores ``ref_context_length`` future VLA actions starting
        at the chunk start (zero-padded past the episode tail).

        Args:
            z_rls: [T, z_rl_dim] RL tokens at each timestep.
            states: [T, state_dim] proprioceptive states at each timestep.
            actions: [T, action_dim] executed actions at each timestep.
            ref_actions: [T, action_dim] VLA reference actions at each timestep.
            rewards: [T] per-step rewards.
            dones: [T] done flags.
            segment_end_step: If provided, mark any transition whose chunk ends
                at or after this step as terminal for RLT bootstrapping (even
                when the underlying environment episode continues). Used to
                gate training to a hand-picked ``[X, Y)`` segment.
        """
        T = z_rls.shape[0]
        C = self.chunk_length
        H = self.ref_context_length

        _, d = ref_actions.shape

        # Include chunks whose end == T (previously off-by-one; those chunks
        # correspond to the final window of the episode and carry the true
        # done / segment-boundary signal).
        for start in range(0, T - C + 1, stride):
            end = start + C
            next_idx = min(end, T - 1)

            ref_ctx_end = min(start + H, T)
            ref_ctx = np.zeros((H, d), dtype=np.float32)
            ref_ctx[: ref_ctx_end - start] = ref_actions[start:ref_ctx_end]

            next_ctx_start = next_idx
            next_ctx_end = min(next_ctx_start + H, T)
            next_ref_ctx = np.zeros((H, d), dtype=np.float32)
            next_ref_ctx[: next_ctx_end - next_ctx_start] = ref_actions[next_ctx_start:next_ctx_end]

            terminal = bool(dones[end - 1])
            if segment_end_step is not None and end >= segment_end_step:
                terminal = True

            transition = Transition(
                z_rl=z_rls[start],
                state=states[start],
                action_chunk=actions[start:end],
                ref_action_chunk=ref_ctx,
                rewards=rewards[start:end],
                next_z_rl=z_rls[next_idx],
                next_state=states[next_idx],
                next_ref_action_chunk=next_ref_ctx,
                done=terminal,
            )
            self.add(transition)


class DualReplayBuffer:
    """Article's demo + online two-buffer setup with mixed sampling.

    - ``demo_buffer`` is populated from warmup VLA rollouts and prior runs.
    - ``online_buffer`` is populated from the current policy's rollouts. It
      can be cleared between Learner runs with :meth:`reset_online`.
    - :meth:`sample` draws ``demo_fraction * batch_size`` from the demo buffer
      (when non-empty) and the rest from the online buffer, then concatenates.

    Both buffers share the same shape contract (chunk/ref/state/z_rl dims),
    so their tensors can be stacked directly.
    """

    def __init__(
        self,
        demo_capacity: int,
        online_capacity: int,
        chunk_length: int = 10,
        action_dim: int = 16,
        z_rl_dim: int = 2048,
        state_dim: int = 16,
        ref_context_length: int | None = None,
    ) -> None:
        kwargs = dict(
            chunk_length=chunk_length,
            action_dim=action_dim,
            z_rl_dim=z_rl_dim,
            state_dim=state_dim,
            ref_context_length=ref_context_length,
        )
        self.demo_buffer = ReplayBuffer(capacity=demo_capacity, **kwargs)
        self.online_buffer = ReplayBuffer(capacity=online_capacity, **kwargs)

    @property
    def size(self) -> int:
        """Total transition count across both buffers."""
        return self.demo_buffer.size + self.online_buffer.size

    @property
    def online_size(self) -> int:
        return self.online_buffer.size

    @property
    def demo_size(self) -> int:
        return self.demo_buffer.size

    def reset_online(self) -> None:
        """Clear the online buffer (between Learner runs, unless resuming)."""
        self.online_buffer.clear()

    def add_to_demo(self, transition: Transition) -> None:
        self.demo_buffer.add(transition)

    def add_to_online(self, transition: Transition) -> None:
        self.online_buffer.add(transition)

    def add_chunk_with_subsampling_to_online(
        self,
        z_rls: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        ref_actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        stride: int = 2,
        segment_end_step: int | None = None,
    ) -> None:
        self.online_buffer.add_chunk_with_subsampling(
            z_rls, states, actions, ref_actions, rewards, dones,
            stride=stride, segment_end_step=segment_end_step,
        )

    def add_chunk_with_subsampling_to_demo(
        self,
        z_rls: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        ref_actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        stride: int = 2,
        segment_end_step: int | None = None,
    ) -> None:
        self.demo_buffer.add_chunk_with_subsampling(
            z_rls, states, actions, ref_actions, rewards, dones,
            stride=stride, segment_end_step=segment_end_step,
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        demo_fraction: float = 0.5,
    ) -> dict[str, Tensor]:
        """Sample a mixed batch: ``demo_fraction`` from demo, rest from online.

        Falls back gracefully when one of the two buffers is empty:
        - Demo empty → all from online.
        - Online empty → all from demo.
        - Both empty → ``ValueError``.
        """
        demo_empty = self.demo_buffer.size == 0
        online_empty = self.online_buffer.size == 0
        if demo_empty and online_empty:
            raise ValueError("DualReplayBuffer.sample called with both buffers empty.")
        if demo_empty:
            return self.online_buffer.sample(batch_size, device)
        if online_empty:
            return self.demo_buffer.sample(batch_size, device)

        demo_frac = float(np.clip(demo_fraction, 0.0, 1.0))
        demo_bs = int(round(batch_size * demo_frac))
        demo_bs = max(0, min(batch_size, demo_bs))
        online_bs = batch_size - demo_bs

        if demo_bs == 0:
            return self.online_buffer.sample(batch_size, device)
        if online_bs == 0:
            return self.demo_buffer.sample(batch_size, device)

        demo_batch = self.demo_buffer.sample(demo_bs, device)
        online_batch = self.online_buffer.sample(online_bs, device)
        return {k: torch.cat([demo_batch[k], online_batch[k]], dim=0) for k in demo_batch}
