"""Abstract environment interface for RLT online RL (Stage 2).

Defines the contract that any environment (sim or real) must satisfy to be
used with the RLT training loop. Observations are returned as dicts matching
the openpi canonical format used by TSHInputs.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np


class RLEnvironment(abc.ABC):
    """Abstract base class for RL environments compatible with RLT.

    Observations are returned as dicts with keys:
        exo_image:         [H, W, 3] uint8 RGB
        wrist_left_image:  [H, W, 3] uint8 RGB
        wrist_right_image: [H, W, 3] uint8 RGB
        state:             [state_dim] float32 proprioceptive state
    """

    @abc.abstractmethod
    def reset(self) -> dict[str, Any]:
        """Reset the environment and return the initial observation."""

    @abc.abstractmethod
    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute a single-step action in the environment.

        Args:
            action: [action_dim] action to execute.

        Returns:
            obs: Observation dict.
            reward: Scalar reward.
            done: Whether the episode has ended.
            info: Additional info dict.
        """

    @abc.abstractmethod
    def get_proprioception(self) -> np.ndarray:
        """Return the current proprioceptive state vector."""

    @abc.abstractmethod
    def task_completed(self) -> bool:
        """Return True if the task has been successfully completed."""

    def step_chunk(
        self, action_chunk: np.ndarray,
    ) -> tuple[dict[str, Any], np.ndarray, bool, dict[str, Any]]:
        """Execute an action chunk (multiple steps) and return aggregated results.

        Args:
            action_chunk: [C, action_dim] chunk of actions to execute sequentially.

        Returns:
            obs: Final observation after executing all actions.
            rewards: [C] per-step rewards.
            done: Whether the episode ended during chunk execution.
            info: Info from the last step.
        """
        C = action_chunk.shape[0]
        rewards = np.zeros(C, dtype=np.float32)
        done = False
        info: dict[str, Any] = {}
        obs: dict[str, Any] = {}

        for i in range(C):
            obs, rewards[i], done, info = self.step(action_chunk[i])
            if done:
                # Zero-fill remaining rewards if episode ends early.
                break

        return obs, rewards, done, info
