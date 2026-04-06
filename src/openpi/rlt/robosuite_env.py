"""Robosuite environment wrapper for RLT online RL.

Wraps FrankaRobosuiteTapeHandover from the mike/dependencies/robosuite fork
and adapts it to the RLEnvironment interface expected by the RLT training loop.

Handles:
- Camera name mapping (robosuite -> openpi canonical names)
- Image resizing (512x512 -> target size for VLA)
- Proprioception extraction (16D bimanual Franka state)
- Single-step action interface via robosuite_env.step()
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from openpi.rlt.env_interface import RLEnvironment

# Add robosuite fork to path if not already importable.
_ROBOSUITE_PATH = Path.home() / "mike" / "dependencies" / "robosuite"
if str(_ROBOSUITE_PATH) not in sys.path:
    sys.path.insert(0, str(_ROBOSUITE_PATH))


# Camera name mapping: robosuite keys -> openpi canonical keys.
_CAMERA_MAP = {
    "agentview_image": "exo_image",
    "robot0_eye_in_hand_image": "wrist_left_image",
    "robot1_eye_in_hand_image": "wrist_right_image",
}


def format_proprio(
    obs: dict,
    gripper_qpos_max: float = 0.08,
    gripper_qpos_min: float = 0.0,
) -> np.ndarray:
    """Convert robosuite obs dict to 16D proprioceptive vector.

    Layout: [left_j0..6, left_gripper, right_j0..6, right_gripper]
    Gripper encoding: +1.0 = fully open, -1.0 = fully closed.
    """
    span = gripper_qpos_max - gripper_qpos_min
    left_raw = float(np.sum(np.abs(obs["robot0_gripper_qpos"])))
    left_gripper = np.array([1.0 - 2.0 * (left_raw / span)])
    right_raw = float(np.sum(np.abs(obs["robot1_gripper_qpos"])))
    right_gripper = np.array([1.0 - 2.0 * (right_raw / span)])
    return np.concatenate([
        obs["robot0_joint_pos"],
        left_gripper,
        obs["robot1_joint_pos"],
        right_gripper,
    ]).astype(np.float32)


class RobosuiteRLTEnv(RLEnvironment):
    """RLT-compatible wrapper around FrankaRobosuiteTapeHandover.

    Args:
        controller_cfg: Path to the robosuite controller config JSON.
        image_size: Target image size (images are resized from 512x512).
        use_wrist_cameras: Whether to enable wrist cameras.
        max_steps: Maximum steps per episode.
        seed: Random seed for the environment.
    """

    def __init__(
        self,
        controller_cfg: str = "robosuite/environments/custom/configs/panda_joint_ctrl.json",
        image_size: int = 224,
        use_wrist_cameras: bool = True,
        max_steps: int = 1800,
        seed: int | None = None,
    ) -> None:
        from robosuite.environments.custom.franka_robosuite_tape_handover import (
            FrankaRobosuiteTapeHandover,
        )

        self.image_size = image_size
        self.env = FrankaRobosuiteTapeHandover(
            controller_cfg=controller_cfg,
            viser_debug=False,
            privileged=True,
            enable_render=False,
            use_wrist_cameras=use_wrist_cameras,
            max_steps=max_steps,
            seed=seed,
        )
        self._raw_obs: dict[str, Any] = {}

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size using bilinear interpolation."""
        if image.shape[0] == self.image_size and image.shape[1] == self.image_size:
            return image
        return cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )

    def _format_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """Convert raw robosuite observations to openpi canonical format."""
        obs: dict[str, Any] = {}

        # Map and resize camera images.
        for rs_key, canonical_key in _CAMERA_MAP.items():
            if rs_key in raw_obs:
                image = raw_obs[rs_key]
                # Robosuite images may need vertical flip (opencv convention).
                if image.ndim == 3 and image.shape[2] == 3:
                    obs[canonical_key] = self._resize_image(image)
                else:
                    obs[canonical_key] = image

        # Extract proprioception.
        obs["state"] = format_proprio(raw_obs)

        return obs

    def reset(self) -> dict[str, Any]:
        """Reset the environment."""
        _obs, _info = self.env.reset()
        self._raw_obs = self.env.robosuite_env._get_observations()
        return self._format_obs(self._raw_obs)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute a single action step.

        Args:
            action: [16] joint position action for bimanual Franka.

        Returns:
            obs, reward, done, info.
        """
        self.env.robosuite_env.step(action)
        self._raw_obs = self.env.robosuite_env._get_observations()
        obs = self._format_obs(self._raw_obs)

        reward = 1.0 if self.task_completed() else 0.0
        done = self.task_completed() or self.env._step_count >= self.env.max_steps
        self.env._step_count += 1
        info: dict[str, Any] = {"success": self.task_completed()}

        return obs, reward, done, info

    def get_proprioception(self) -> np.ndarray:
        """Return current 16D proprioceptive state."""
        return format_proprio(self._raw_obs)

    def task_completed(self) -> bool:
        """Check if the task has been successfully completed."""
        return self.env.task_completed()
