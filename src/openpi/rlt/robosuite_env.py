"""Robosuite environment wrapper for RLT online RL.

Wraps FrankaRobosuiteTapeHandover from the mike/dependencies/robosuite fork
and adapts it to the RLEnvironment interface expected by the RLT training loop.

Handles:
- Camera name mapping (robosuite -> openpi canonical names)
- Visual + control defaults aligned with ``groundTruthEval`` (opencv image convention,
  ``panda_joint_ctrl_slow.json``) so the frozen VLA sees the same pixels and dynamics
  as in high-accuracy eval
- Optional image resize (default: full-res, policy applies ``resize_with_pad`` like eval)
- Optional ``handover_index.json`` tape layouts (match training / ``groundTruthEval``)
- Observations from ``FrankaRobosuiteTapeHandover.get_observation()`` like eval
- Proprioception extraction (16D bimanual Franka state)
- Single-step action interface via robosuite_env.step()
- First ``reset()`` with tape layout index ``0`` skips a redundant inner ``reset()`` so the
  initial observation matches ``groundTruthEval`` (Franka ``__init__`` already reset with
  ``_tape_combos[0]``); later episodes always full-reset the scene.
- Optional sim state snapshots (``sim_states_dir``): on each reset, restores a
  pre-captured MuJoCo sim state so the episode starts from a mid-demo configuration
  rather than the beginning of the scene. One snapshot per unique tape position,
  created by ``misc_scripts/capture_sim_states.py``. After ``reset_from_xml_string``,
  training-time ``contact_solref`` / ``contact_solimp`` are applied again so they
  match ``groundTruthEval`` even though the XML was reloaded from disk.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from openpi.rlt.env_interface import RLEnvironment

# Tolerance (metres) for nearest-neighbour tape position lookup.
# A warning is issued when the closest snapshot is farther than this.
_SIM_STATE_MATCH_WARN_DIST = 0.005

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid tape position grid (mirrors sweep_handover_offsets.sh exactly)
#
# Each tape uses a 4×2 grid → 8 positions per tape → 64 valid (yellow, duct)
# combinations total.
#
# Yellow tape:  x ∈ [-0.2, 0.1] (4 steps), y ∈ [0.25, 0.5]  (2 steps)
# Duct tape:    x ∈ [-0.2, 0.1] (4 steps), y ∈ [-0.5, -0.25] (2 steps)
# ---------------------------------------------------------------------------
def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if n == 1:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [round(lo + i * step, 6) for i in range(n)]

_YELLOW_X_VALS = _linspace(-0.2, 0.1, 4)   # -0.2, -0.1, 0.0, 0.1
_YELLOW_Y_VALS = _linspace(0.25, 0.5, 2)   # 0.25, 0.5
_DUCT_X_VALS   = _linspace(-0.2, 0.1, 4)   # -0.2, -0.1, 0.0, 0.1
_DUCT_Y_VALS   = _linspace(-0.5, -0.25, 2) # -0.5, -0.25

_VALID_YELLOW: list[tuple[float, float]] = [
    (x, y) for x in _YELLOW_X_VALS for y in _YELLOW_Y_VALS
]  # 8 positions
_VALID_DUCT: list[tuple[float, float]] = [
    (x, y) for x in _DUCT_X_VALS for y in _DUCT_Y_VALS
]  # 8 positions

# All 64 valid (yellow, duct) pairs — the same set swept by sweep_handover_offsets.sh.
_VALID_COMBOS: list[tuple[tuple[float, float], tuple[float, float]]] = [
    (yellow, duct) for yellow in _VALID_YELLOW for duct in _VALID_DUCT
]

_RESET_MAX_ATTEMPTS = 3  # initial attempt + 2 retries


def load_tape_combos_from_handover_json(path: str | Path) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Load tape layouts from the same JSON schema as ``groundTruthEval.offsets_json_path``.

    Each entry must have ``yellow_x``, ``yellow_y``, ``duct_x``, ``duct_y`` (floats).
    Using the dataset / eval index file ensures rollouts hit the **same** XY pairs the
    VLA was trained on; the built-in 4×4 linspace grid can differ slightly in float
    values and ordering.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Tape offsets JSON not found: {p}")
    with p.open() as f:
        data = json.load(f)
    combos: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for entry in data:
        combos.append(
            (
                (float(entry["yellow_x"]), float(entry["yellow_y"])),
                (float(entry["duct_x"]), float(entry["duct_y"])),
            )
        )
    return combos

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


def _resolve_controller_cfg(controller_cfg: str) -> str:
    """Turn controller paths into an absolute file path for robosuite's loader.

    Robosuite opens ``*.json`` paths as-is; defaults like
    ``robosuite/environments/custom/configs/...`` only work when cwd is the
    robosuite repo root. Anchor relative paths to the ``robosuite`` package
    directory instead.
    """
    path = Path(controller_cfg)
    if path.is_file():
        return str(path.resolve())
    import robosuite

    pkg_root = Path(robosuite.__file__).resolve().parent
    rel = controller_cfg.replace("\\", "/")
    if rel.startswith("robosuite/"):
        rel = rel.split("/", 1)[1]
    candidate = (pkg_root / rel).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Robosuite controller config not found: {controller_cfg!r} "
            f"(resolved to {candidate}; robosuite package root {pkg_root})"
        )
    return str(candidate)


_NEWER_MJCF_ATTRS = ("colorspace",)


def _sanitize_mjcf_for_local_mujoco(xml: str) -> str:
    """Drop MJCF attributes added by newer mujoco that older parsers reject.

    Captures done with mujoco>=3.x emit ``colorspace="..."`` on ``<texture>``; mujoco<=2.x
    raises ``Schema violation: unrecognized attribute: 'colorspace'``. We only ever
    *replay* states from these XMLs, so dropping such attributes is safe.
    """
    cleaned = xml
    dropped: dict[str, int] = {}
    for attr in _NEWER_MJCF_ATTRS:
        pattern = re.compile(rf'\s+{re.escape(attr)}="[^"]*"')
        new_xml, n = pattern.subn("", cleaned)
        if n:
            dropped[attr] = n
            cleaned = new_xml
    if dropped:
        logger.info(
            "sim_states XML: stripped %s for local mujoco compatibility.",
            ", ".join(f"{k}×{v}" for k, v in dropped.items()),
        )
    return cleaned


def _ensure_robosuite_opencv_image_convention() -> None:
    """Use the same camera convention as ``groundTruthEval`` (``suite.macros.IMAGE_CONVENTION``).

    Robosuite applies a vertical flip inside the camera sensor when this is ``opencv``,
    producing upright RGB. Without it (default ``opengl``), a second flip in the env
    wrapper was used historically — that diverges from eval, which does not flip again
    after setting ``opencv``.
    """
    import robosuite

    robosuite.macros.IMAGE_CONVENTION = "opencv"


def sanitize_bimanual_panda_action(action: np.ndarray) -> np.ndarray:
    """Prepare a 16D action for ``TwoArmTapeHandover.step``.

    Arm joints (dims 0-6 and 8-14) are absolute joint positions in radians.
    Gripper dims 7 and 15 must stay in [-1, 1] for robosuite's GRIP controller:
    approximately -1 = open, +1 = closed (matches ``format_proprio`` and
    ``FrankaRobosuiteTapeHandover``'s internal ``1.0 - 2 * opening_fraction``).

    Values outside [-1, 1] on gripper channels (common from unbounded RL actor
    heads) are clipped so the fingers still receive valid controller commands.
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.size != 16:
        raise ValueError(
            f"Expected 16-dim bimanual action, got shape {np.asarray(action).shape}"
        )
    a = a.copy()
    a[7] = float(np.clip(a[7], -1.0, 1.0))
    a[15] = float(np.clip(a[15], -1.0, 1.0))
    return a


def apply_gripper_zero_hold(
    action: np.ndarray,
    last_grip_cmds: np.ndarray,
    *,
    eps: float = 0.02,
) -> np.ndarray:
    """Avoid stalling Panda grippers when the policy emits near-zero gripper targets.

    Robosuite's ``PandaGripper.format_action`` updates fingers using ``np.sign(cmd)``
    (incremental motion). If ``cmd`` is exactly ~0, ``sign(0)=0`` and the gripper does
    not move — common with flow-matching noise or blended actions that average toward 0.

    For each gripper dim (7 and 15), if ``|cmd| < eps``, reuse ``last_grip_cmds``;
    otherwise clip to [-1, 1], write back to ``last_grip_cmds``, and use that value.
    Initialize ``last_grip_cmds`` from proprio (state[7], state[15]) at episode start.
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1).copy()
    if a.size != 16:
        raise ValueError(
            f"Expected 16-dim bimanual action, got shape {np.asarray(action).shape}"
        )
    if last_grip_cmds.shape != (2,):
        raise ValueError("last_grip_cmds must be shape (2,) for [left, right] grippers.")
    for i, idx in enumerate((7, 15)):
        v = float(a[idx])
        if abs(v) < eps:
            a[idx] = float(np.clip(last_grip_cmds[i], -1.0, 1.0))
        else:
            c = float(np.clip(v, -1.0, 1.0))
            last_grip_cmds[i] = c
            a[idx] = c
    return a


def format_proprio(
    obs: dict,
    gripper_qpos_max: float = 0.08,
    gripper_qpos_min: float = 0.0,
) -> np.ndarray:
    """Convert robosuite obs dict to 16D proprioceptive vector.

    Layout: [left_j0..6, left_gripper, right_j0..6, right_gripper]
    Gripper encoding (from finger qpos span): about -1.0 = open, +1.0 = closed,
    aligned with the GRIP action convention above.
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
        controller_cfg: Path to the robosuite controller config JSON. Default matches
            ``groundTruthEval`` (``panda_joint_ctrl_slow.json``); the stiffer
            ``panda_joint_ctrl.json`` makes the same policy targets track very differently.
        image_size: If set, resize square RGB to this side length (cv2 linear). If ``None``,
            pass native camera resolution through — same as eval, letting the policy's
            ``ResizeImages`` / ``resize_with_pad`` handle downscaling.
        use_wrist_cameras: Whether to enable wrist cameras.
        max_steps: Maximum steps per episode.
        seed: Random seed for the environment.
        tape_offsets_json: If set, load yellow/duct XY pairs from this file (eval / dataset
            index). Otherwise use the built-in 64-combo linspace grid.
        contact_solref: If set, assign all ``geom_solref`` to this ``[timeconst, dampratio]``
            pair (same as ``groundTruthEval``).
        contact_solimp: If set, assign ``geom_solimp[:, :n]`` for the first ``n`` columns
            (e.g. three values for ``[dmin, dmax, width]``).

    Note:
        On the first call to :meth:`reset`, if the chosen tape layout index is ``0`` (from
        ``tape_layout_index=0`` or random sampling), the inner ``Franka.reset()`` is skipped:
        ``__init__`` already ran a full mujoco reset with ``_tape_combos[0]``, matching
        ``groundTruthEval``. Later ``reset`` calls always re-randomize the scene.
    """

    def __init__(
        self,
        controller_cfg: str = "robosuite/environments/custom/configs/panda_joint_ctrl_slow.json",
        image_size: int | None = None,
        use_wrist_cameras: bool = True,
        max_steps: int = 1800,
        seed: int | None = None,
        tape_offsets_json: str | Path | None = None,
        contact_solref: list[float] | None = None,
        contact_solimp: list[float] | None = None,
        tape_layout_index: int | None = None,
        gripper_action_log: str | Path | None = None,
        sim_states_dir: str | Path | None = None,
    ) -> None:
        _ensure_robosuite_opencv_image_convention()
        from robosuite.environments.custom.franka_robosuite_tape_handover import (
            FrankaRobosuiteTapeHandover,
        )

        self.image_size = image_size
        self._rng = np.random.default_rng(seed)
        if tape_offsets_json is not None:
            self._tape_combos = load_tape_combos_from_handover_json(tape_offsets_json)
            logger.info(
                "RobosuiteRLTEnv: using %d tape layouts from %s",
                len(self._tape_combos),
                tape_offsets_json,
            )
        else:
            self._tape_combos = _VALID_COMBOS

        # Pick the first valid combo so the construction-time reset succeeds.
        (yellow_x, yellow_y), (duct_x, duct_y) = self._tape_combos[0]

        controller_cfg = _resolve_controller_cfg(controller_cfg)
        self._contact_solref = contact_solref
        self._contact_solimp = contact_solimp
        self.env = FrankaRobosuiteTapeHandover(
            controller_cfg=controller_cfg,
            viser_debug=False,
            privileged=True,
            enable_render=False,
            use_wrist_cameras=use_wrist_cameras,
            max_steps=max_steps,
            seed=seed,
            yellow_tape_offset=[yellow_x, yellow_y, 0.0],
            duct_tape_offset=[duct_x, duct_y, 0.0],
        )
        self._apply_contact_overrides(self._contact_solref, self._contact_solimp)
        self._raw_obs: dict[str, Any] = {}
        self._last_grip_cmd = np.array([1.0, 1.0], dtype=np.float64)
        self._tape_layout_index = tape_layout_index
        self._gripper_action_log_path = (
            str(Path(gripper_action_log).resolve()) if gripper_action_log else None
        )
        self._gripper_log_episode_idx = 0
        self._gripper_log_local_step = 0
        # True until the first :meth:`reset` completes. Used to avoid a second full mujoco reset
        # when the first episode uses layout 0 — same as ``groundTruthEval`` after Franka __init__.
        self._first_reset_after_construct = True

        # Sim state snapshots (optional). Loaded eagerly so missing files fail at
        # construction time, not mid-training.
        self._sim_states: np.ndarray | None = None      # [N, state_dim]
        self._sim_state_positions: np.ndarray | None = None  # [N, 4] (yx, yy, dx, dy)
        self._sim_state_xml: str | None = None
        if sim_states_dir is not None:
            self._load_sim_states(Path(sim_states_dir), use_wrist_cameras=use_wrist_cameras)

    def _load_sim_states(self, sim_states_dir: Path, *, use_wrist_cameras: bool) -> None:
        """Load pre-captured sim state snapshots from *sim_states_dir*.

        Supports two layouts produced by ``misc_scripts/capture_sim_states.py``:

        **Per-file layout** (current default):
          index.json  -- list of {yellow_x, yellow_y, duct_x, duct_y, state_file}
          <state_file>.npy  -- one flattened MjSimState per tape position
          model.xml   -- MuJoCo model XML string (required for correct restore)

        **Combined layout** (legacy):
          index.json  -- list of {yellow_x, yellow_y, duct_x, duct_y, state_idx}
          states.npz  -- 'states': [N, state_dim] array of flattened MjSimState
          model.xml   -- MuJoCo model XML string (required)
        """
        index_path = sim_states_dir / "index.json"
        xml_path = sim_states_dir / "model.xml"

        if not index_path.exists():
            raise FileNotFoundError(f"sim_states index.json not found: {index_path}")

        with index_path.open() as f:
            index = json.load(f)

        if not index:
            raise ValueError(f"sim_states index.json is empty: {index_path}")

        positions = np.array(
            [[e["yellow_x"], e["yellow_y"], e["duct_x"], e["duct_y"]] for e in index],
            dtype=np.float64,
        )

        # Per-file layout: each entry has a 'state_file' key.
        if "state_file" in index[0]:
            ordered_states_list = []
            for entry in index:
                npy_path = sim_states_dir / entry["state_file"]
                if not npy_path.exists():
                    raise FileNotFoundError(f"sim_states file not found: {npy_path}")
                ordered_states_list.append(np.load(str(npy_path)))
            ordered_states = np.stack(ordered_states_list, axis=0)
        else:
            # Legacy combined layout: entries have a 'state_idx' key.
            states_path = sim_states_dir / "states.npz"
            if not states_path.exists():
                raise FileNotFoundError(f"sim_states states.npz not found: {states_path}")
            data = np.load(str(states_path))
            states = data["states"]
            ordered_states = np.zeros((len(index), states.shape[1]), dtype=np.float64)
            for i, entry in enumerate(index):
                ordered_states[i] = states[entry["state_idx"]]

        self._sim_states = ordered_states
        self._sim_state_positions = positions

        if not xml_path.exists():
            raise FileNotFoundError(
                f"sim_states model.xml not found: {xml_path}. "
                "Re-run misc_scripts/capture_sim_states.py (it writes model.xml on the first "
                "successful layout). Without it, nq/nv may not match the saved .npy vectors."
            )
        self._sim_state_xml = _sanitize_mjcf_for_local_mujoco(
            xml_path.read_text(encoding="utf-8")
        )
        xml_text = self._sim_state_xml.lower()
        has_wrist = "robot0_eye_in_hand" in xml_text or "eye_in_hand" in xml_text
        if use_wrist_cameras and not has_wrist:
            logger.warning(
                "sim_states model.xml looks like agentview-only (no wrist camera names in XML) "
                "but RobosuiteRLTEnv has use_wrist_cameras=True. Re-capture with "
                "misc_scripts/capture_sim_states.py without --no-wrist-cameras, or set "
                "--robosuite_use_wrist_cameras false on Stage 2."
            )
        elif not use_wrist_cameras and has_wrist:
            logger.warning(
                "sim_states appear to include wrist cameras but use_wrist_cameras=False; "
                "observation keys may not match what the VLA expects."
            )

        logger.info(
            "RobosuiteRLTEnv: loaded %d sim state snapshots from %s",
            len(index), sim_states_dir,
        )

    def _lookup_sim_state(
        self,
        yellow_x: float,
        yellow_y: float,
        duct_x: float,
        duct_y: float,
    ) -> np.ndarray:
        """Return the flattened MjSimState closest to the given tape position.

        Uses Euclidean distance in (yellow_x, yellow_y, duct_x, duct_y) space.
        Warns if the nearest neighbour is farther than ``_SIM_STATE_MATCH_WARN_DIST``.
        """
        query = np.array([yellow_x, yellow_y, duct_x, duct_y], dtype=np.float64)
        dists = np.linalg.norm(self._sim_state_positions - query, axis=1)
        idx = int(np.argmin(dists))
        dist = float(dists[idx])
        if dist > _SIM_STATE_MATCH_WARN_DIST:
            logger.warning(
                "_lookup_sim_state: nearest snapshot is %.4f m away from "
                "yellow=(%.4f, %.4f) duct=(%.4f, %.4f). "
                "Consider adding a snapshot for this exact position.",
                dist, yellow_x, yellow_y, duct_x, duct_y,
            )
        return self._sim_states[idx]

    def _apply_contact_overrides(
        self,
        contact_solref: list[float] | None,
        contact_solimp: list[float] | None,
    ) -> None:
        """Match ``groundTruthEval`` sim_worker contact tweaks when provided."""
        rs = self.env.robosuite_env
        if contact_solref is not None:
            solref = np.array(contact_solref, dtype=np.float64)
            rs.sim.model.geom_solref[:] = solref
        if contact_solimp is not None:
            solimp = np.array(contact_solimp, dtype=np.float64)
            n = len(solimp)
            rs.sim.model.geom_solimp[:, :n] = solimp

    def _sync_robots_after_sim_state(self, rs: Any) -> None:
        """Align composite controllers with ``sim.data`` after ``set_state_from_flattened``.

        ``reset_from_xml_string`` ends with ``env.reset()``, which sets controller goals from
        the transient pose at that moment. We then overwrite *only* time/qpos/qvel via the
        flattened snapshot, so goals still target the old configuration until this runs —
        the arms ease toward the nominal init over the first steps (looks like "start").
        """
        robots = getattr(rs, "robots", None)
        if not robots:
            return
        for robot in robots:
            cc = getattr(robot, "composite_controller", None)
            if cc is None:
                continue
            cc.update_state()
            cc.reset()

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size using bilinear interpolation."""
        if self.image_size is None:
            return image
        if image.shape[0] == self.image_size and image.shape[1] == self.image_size:
            return image
        return cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )

    def _format_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """Convert raw robosuite observations to openpi canonical format."""
        obs: dict[str, Any] = {}

        # Map and optionally resize camera images. Orientation comes from
        # ``IMAGE_CONVENTION=opencv`` (see ``_ensure_robosuite_opencv_image_convention``),
        # matching ``groundTruthEval`` — do not flip here.
        for rs_key, canonical_key in _CAMERA_MAP.items():
            if rs_key in raw_obs:
                image = raw_obs[rs_key]
                if image.ndim == 3 and image.shape[2] == 3:
                    obs[canonical_key] = self._resize_image(np.asarray(image))
                else:
                    obs[canonical_key] = image

        # Extract proprioception.
        obs["state"] = format_proprio(raw_obs)

        return obs

    def _update_tape_positions(self, yellow_x: float, yellow_y: float, duct_x: float, duct_y: float) -> None:
        """Set the next episode's tape XY from the valid grid.

        ``TwoArmTapeHandover`` uses ``yellow_tape_offset`` / ``duct_tape_offset`` when
        building placement samplers. With ``hard_reset=True`` (robosuite default), every
        ``reset()`` calls ``_load_model()`` → ``_get_placement_initializer()`` and **rebuilds**
        those samplers from these offsets. Mutating only the old sampler objects is ignored
        after the first reset, which is why tape poses looked fixed across episodes.

        We update both the stored offsets (for hard reset) and, when samplers already exist,
        their ranges (for soft-reset / consistency).
        """
        rs = self.env.robosuite_env
        rs.yellow_tape_offset = np.array([yellow_x, yellow_y, float(rs.yellow_tape_offset[2])], dtype=np.float64)
        rs.duct_tape_offset = np.array([duct_x, duct_y, float(rs.duct_tape_offset[2])], dtype=np.float64)

        if rs.placement_initializer is not None and hasattr(rs.placement_initializer, "samplers"):
            samplers = rs.placement_initializer.samplers
            if "YellowTapeSampler" in samplers:
                s_yellow = samplers["YellowTapeSampler"]
                s_yellow.x_range = [yellow_x, yellow_x]
                s_yellow.y_range = [yellow_y, yellow_y]
            if "DuctTapeSampler" in samplers:
                s_duct = samplers["DuctTapeSampler"]
                s_duct.x_range = [duct_x, duct_x]
                s_duct.y_range = [duct_y, duct_y]

    def reset(self) -> dict[str, Any]:
        """Reset the environment, randomizing tape positions from the valid grid.

        The first successful reset with layout index ``0`` does **not** call
        ``FrankaRobosuiteTapeHandover.reset()`` again: the inner env was already fully reset in
        ``__init__`` with ``_tape_combos[0]``, matching ``groundTruthEval`` before the first
        policy step. All later resets always call the inner ``reset()``.

        Retries up to ``_RESET_MAX_ATTEMPTS - 1`` extra times on
        ``RandomizationError`` before re-raising.
        """
        from robosuite.utils.errors import RandomizationError

        last_exc: Exception | None = None
        for attempt in range(_RESET_MAX_ATTEMPTS):
            if self._tape_layout_index is not None:
                idx = int(self._tape_layout_index) % len(self._tape_combos)
            else:
                idx = int(self._rng.integers(len(self._tape_combos)))
            (yellow_x, yellow_y), (duct_x, duct_y) = self._tape_combos[idx]
            self._update_tape_positions(yellow_x, yellow_y, duct_x, duct_y)

            try:
                if self._first_reset_after_construct and idx == 0 and attempt == 0:
                    # Inner Franka already ran ``robosuite_env.reset()`` in ``__init__`` for
                    # ``_tape_combos[0]``.  ``groundTruthEval.sim_worker`` uses that state for the
                    # first inference; do not re-sample initialization noise a second time.
                    self._raw_obs = self.env.get_observation()
                else:
                    _obs, _info = self.env.reset()
                    # Match ``groundTruthEval`` / Franka API (not raw ``_get_observations``).
                    self._raw_obs = self.env.get_observation()
                self._first_reset_after_construct = False

                # Restore pre-captured sim state if snapshots are configured.
                # This overwrites the normal reset state so the episode starts from
                # exactly the mid-demo configuration at the captured sim time.
                if self._sim_states is not None:
                    state_flat = np.asarray(
                        self._lookup_sim_state(yellow_x, yellow_y, duct_x, duct_y),
                        dtype=np.float64,
                    ).reshape(-1)
                    rs = self.env.robosuite_env
                    # Match robosuite DemoSamplerWrapper: ``reset_from_xml_string`` reloads the
                    # model and runs a full ``env.reset()`` internally. Do **not** call
                    # ``sim.reset()`` between that and ``set_state_from_flattened`` —
                    # ``mj_resetData`` zeros state and breaks the placement/state sequence.
                    if self._sim_state_xml is None:
                        raise RuntimeError("sim state XML missing (should have been required at load).")
                    rs.reset_from_xml_string(self._sim_state_xml)
                    expected = 1 + rs.sim.model.nq + rs.sim.model.nv
                    if state_flat.size != expected:
                        raise ValueError(
                            f"Saved sim state length {state_flat.size} != 1+nq+nv={expected} "
                            f"(nq={rs.sim.model.nq}, nv={rs.sim.model.nv}). "
                            "Recapture with the same robosuite build (cameras, controller) as Stage 2."
                        )
                    if int(rs.sim.model.na) != 0:
                        raise ValueError(
                            f"Sim model has na={rs.sim.model.na}; flattened snapshots require na=0."
                        )
                    rs.sim.set_state_from_flattened(state_flat)
                    rs.sim.forward()
                    self._sync_robots_after_sim_state(rs)
                    self._apply_contact_overrides(self._contact_solref, self._contact_solimp)
                    # Robosuite caches observable values (proprio + camera RGB) at the end of
                    # ``env.reset()``, BEFORE our ``set_state_from_flattened`` ran. Without a
                    # forced refresh, ``get_observation`` returns the post-reset start pose and
                    # the policy plans from there — which makes rollout videos look like the
                    # snapshot was never applied. Force-refresh + clear stale cache.
                    rs._obs_cache = {}  # noqa: SLF001
                    rs._update_observables(force=True)  # noqa: SLF001
                    self.env._step_count = 0
                    self._raw_obs = self.env.get_observation()
                    logger.info(
                        "Restored sim snapshot: yellow=(%.4f,%.4f) duct=(%.4f,%.4f) "
                        "mj_time=%.4f state_dim=%d",
                        yellow_x,
                        yellow_y,
                        duct_x,
                        duct_y,
                        float(state_flat[0]),
                        state_flat.size,
                    )

                out = self._format_obs(self._raw_obs)
                st = out["state"]
                self._last_grip_cmd = np.array([float(st[7]), float(st[15])], dtype=np.float64)
                self._gripper_log_local_step = 0
                if self._gripper_action_log_path:
                    mode = "w" if self._gripper_log_episode_idx == 0 else "a"
                    with open(self._gripper_action_log_path, mode, encoding="utf-8") as gf:
                        if self._gripper_log_episode_idx == 0:
                            gf.write("step,gripper_left_dim7,gripper_right_dim15\n")
                        else:
                            gf.write(f"\n# episode {self._gripper_log_episode_idx}\n")
                    self._gripper_log_episode_idx += 1
                return out
            except RandomizationError as exc:
                last_exc = exc
                logger.warning(
                    "Placement randomization failed (attempt %d/%d) for "
                    "yellow=(%.3f, %.3f) duct=(%.3f, %.3f): %s",
                    attempt + 1, _RESET_MAX_ATTEMPTS,
                    yellow_x, yellow_y, duct_x, duct_y, exc,
                )

        raise RuntimeError(
            f"Environment reset failed after {_RESET_MAX_ATTEMPTS} attempts"
        ) from last_exc

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute a single action step.

        Args:
            action: [16] joint position action for bimanual Franka.

        Returns:
            obs, reward, done, info.
        """
        a = sanitize_bimanual_panda_action(
            apply_gripper_zero_hold(np.asarray(action), self._last_grip_cmd),
        )
        if self._gripper_action_log_path:
            with open(self._gripper_action_log_path, "a", encoding="utf-8") as gf:
                gf.write(
                    f"{self._gripper_log_local_step},{a[7]:.8f},{a[15]:.8f}\n",
                )
            self._gripper_log_local_step += 1
        self.env.robosuite_env.step(a)
        self._raw_obs = self.env.get_observation()
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
