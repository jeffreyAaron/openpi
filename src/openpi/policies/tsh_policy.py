import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_tsh_example() -> dict:
    """Creates a random input example for the TSH bimanual Franka policy."""
    return {
        "observation/state": np.random.rand(16),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_left_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_right_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Pick up the yellow tape, hand it over, and place it on the gray tape",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class TSHInputs(transforms.DataTransformFn):
    """
    Input transform for the TSH bimanual Franka robot.

    State (16D): [left_joint_0..6, left_gripper, right_joint_0..6, right_gripper]
    Actions (16D): same layout; arm joints are absolute radians, grippers ~[-1,1]
    for robosuite GRIP (-1 ≈ open, +1 ≈ closed), matching state encoding.
    """

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        # State is already the full 16D robot joint + gripper vector.
        state = np.asarray(data["state"])

        # Parse images to uint8 (H,W,C). LeRobot stores as float32 (C,H,W)
        # during training; during inference images are passed in directly.
        exo_image = _parse_image(data["exo_image"])
        wrist_left_image = _parse_image(data["wrist_left_image"])
        wrist_right_image = _parse_image(data["wrist_right_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": exo_image,
                "left_wrist_0_rgb": wrist_left_image,
                "right_wrist_0_rgb": wrist_right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Pad 16D actions to the model's expected action_dim (32 for PI0.5).
        inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TSHOutputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        # Return only the first 16 dims -- the meaningful Franka DOFs.
        return {"actions": np.asarray(data["actions"][:, :16])}
