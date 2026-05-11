"""PyRoki / IK based oracle macro for the bimanual tape-handover.

Replaces the RLT GaussianActor during the handover phase. The macro is a direct
adaptation of the post-lift segment of ``build_handover_action_code`` in
``mike/dependencies/robosuite/test_scripts/handover_step.py`` (the upstream
reference oracle), restricted to the steps that move the tape from arm1 to
arm0 and retract arm1. The lift (VLA) and the place (VLA) phases stay outside
this module.

The macro is **blocking**: each ``goto_pose_*`` runs ``move_to_joints_blocking``
inside the underlying ``FrankaRobosuiteTapeHandover``, consuming many physics
substeps before returning. The training rollout caller is responsible for
advancing its own per-step counters by the number of inner sim steps reported
back via :attr:`HandoverMacroResult.inner_steps`.

The oracle assumes the **arm1** end-effector currently holds the yellow tape
(matches ``lift_handover``'s ``lh_first_grasp_arm == 1`` case). When arm0 is
the picker the macro raises :class:`HandoverPickerMismatchError` and the
caller is expected to fall back to the VLA / RLT path for that episode.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# sys.path bootstrap so `from api import ...` and `import pyroki as pk` work
# without modifying the project-level uv environment. Mirrors the pattern in
# ``openpi.rlt.robosuite_env``.
# ---------------------------------------------------------------------------
_MIKE_ROOT = Path.home() / "mike" / "dependencies"
_ROBOSUITE_FORK = _MIKE_ROOT / "robosuite"
_PYROKI_SRC = _MIKE_ROOT / "pyroki" / "src"


def _ensure_paths() -> None:
    for p in (_ROBOSUITE_FORK, _PYROKI_SRC):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


class HandoverPickerMismatchError(RuntimeError):
    """Raised when arm0 (rather than arm1) is currently holding the tape.

    The handover oracle is hard-coded for arm1-led picks; arm0-led picks
    require a mirrored macro that hasn't been written yet.
    """


@dataclass
class HandoverMacroResult:
    """Summary returned by :meth:`PyrokiHandoverOracle.run`.

    Attributes:
        inner_steps: Number of underlying ``robosuite_env.step`` calls
            consumed by the macro (delta of ``_sim_step_count`` across the
            run).  The training rollout caller adds this to its own
            ``ep_steps`` so ``< max_episode_steps`` stays accurate.
        succeeded: ``True`` iff every IK solve returned without raising and
            the env wasn't terminated mid-macro by its internal step budget.
        skipped_reason: Optional string set when the macro short-circuited
            (e.g. ``"sim_step_budget_exhausted"``). ``None`` on a normal
            completion.
    """

    inner_steps: int
    succeeded: bool
    skipped_reason: str | None = None


# Geometry constants reproduced verbatim from
# ``handover_step.build_handover_action_code._handover_geometry`` so the
# resulting waypoints match the reference oracle one-for-one.  Kept module-
# level so they appear in tracebacks and are easy to grep.
_ARM0_OFFSET_FROM_HANDOVER = np.array([0.035, -0.1025, 0.0], dtype=np.float64)
_HANDOVER_OFFSET_FROM_CENTER = np.array([-0.15, 0.10, 0.0], dtype=np.float64)
# Constant gripper-side base quats (wxyz). The final orientation is
# ``Rz(angle_shift) * base_quat * [0, 1, 0, 0]`` (gripper-down flip).
_GR_BASE_QUAT = np.array([0.707, 0, 0.707, 0], dtype=np.float64)
_GS_BASE_QUAT = np.array([0.707, 0, -0.707, 0], dtype=np.float64)
_GRIPPER_DOWN_FLIP = np.array([0, 1, 0, 0], dtype=np.float64)


def _handover_geometry(
    center: np.ndarray,
    *,
    x_shift: float,
    y_shift: float,
    angle_shift: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replicate ``_handover_geometry`` from the reference oracle.

    Returns ``(handover_pos, arm0_handover_pos, gripper_side_quat,
    gripper_rotated_side_quat)`` — all in robot0's base frame.
    """
    import viser.transforms as vtf  # local import: heavy + only needed here

    ang = float(angle_shift)
    Rz_q = np.array([np.cos(ang / 2), 0, 0, np.sin(ang / 2)], dtype=np.float64)
    Rz = vtf.SO3(wxyz=Rz_q).as_matrix()
    h_offset = _HANDOVER_OFFSET_FROM_CENTER + np.array(
        [-float(x_shift), float(y_shift), 0.0], dtype=np.float64,
    )
    h_pos = center + Rz @ h_offset
    a0_pos = h_pos + Rz @ _ARM0_OFFSET_FROM_HANDOVER
    gs_quat = (
        vtf.SO3(wxyz=Rz_q)
        @ vtf.SO3(wxyz=_GS_BASE_QUAT)
        @ vtf.SO3(wxyz=_GRIPPER_DOWN_FLIP)
    ).wxyz
    gr_quat = (
        vtf.SO3(wxyz=Rz_q)
        @ vtf.SO3(wxyz=_GR_BASE_QUAT)
        @ vtf.SO3(wxyz=_GRIPPER_DOWN_FLIP)
    ).wxyz
    return h_pos, a0_pos, np.asarray(gs_quat), np.asarray(gr_quat)


class PyrokiHandoverOracle:
    """Blocking IK macro that performs only the handover segment.

    Args:
        franka_env: The wrapped ``FrankaRobosuiteTapeHandover`` instance
            (i.e. ``RobosuiteRLTEnv.env``). Must expose
            ``move_to_joints_blocking[_arm1]``, ``_set_gripper[_arm1]``,
            ``_step_once``, ``get_observation``, ``base_link_wxyz_xyz_0/1``
            and ``handover_params``.
        x_shift, y_shift, angle_shift: Same handover geometry knobs as the
            reference ``handover_step.py`` CLI; default to 0 = canonical
            handover pose centred between the two arms.
        perturb_radius: Lateral waypoint perturbation radius in metres.
            ``0.0`` (default) disables perturbation — the macro becomes
            fully deterministic.
        robot_urdf: Forwarded to PyRoKi's robot loader. ``"panda_description"``
            matches the upstream oracle; pass an explicit URDF path for
            non-panda variants.
        seed: Optional RNG seed used only when ``perturb_radius > 0``.
    """

    def __init__(
        self,
        franka_env: Any,
        *,
        x_shift: float = 0.0,
        y_shift: float = 0.0,
        angle_shift: float = 0.0,
        perturb_radius: float = 0.0,
        robot_urdf: str = "panda_description",
        seed: int | None = None,
    ) -> None:
        _ensure_paths()
        try:
            from api import pyroki_snippets as pks  # type: ignore
            from api.franka_priviledged_api import (  # type: ignore
                FrankaControlTapeHandoverPrivilegedApi,
            )
            from api.pyroki_context import get_pyroki_context  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyRoki handover oracle requires the mike/dependencies/robosuite + "
                "pyroki packages to be importable. Expected at "
                f"{_ROBOSUITE_FORK} and {_PYROKI_SRC}. Original error: {exc}"
            ) from exc

        # The upstream API expects the wrapped Franka env (not the raw
        # robosuite_env). Reuse the existing class wholesale so we get its
        # already-tested goto_pose / gripper / IK fallback paths.
        self._franka_env = franka_env
        self._api = FrankaControlTapeHandoverPrivilegedApi(franka_env)
        self._pks = pks
        self._get_pyroki_context = get_pyroki_context
        self._robot_urdf = robot_urdf

        self._x_shift = float(x_shift)
        self._y_shift = float(y_shift)
        self._angle_shift = float(angle_shift)
        self._perturb_radius = float(perturb_radius)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Picker-arm sanity check (call before run() if you want to fall back
    # to the VLA on arm0-led picks).
    # ------------------------------------------------------------------
    def assert_arm1_holds_tape(
        self,
        *,
        grip_closed_thresh: float = 0.06,
    ) -> None:
        """Raise :class:`HandoverPickerMismatchError` if arm0 holds the tape.

        Uses the same finger-span heuristic as
        ``openpi.rlt.shaped_reward._gripper_span``: a closed gripper has
        ``sum(|qpos|) < grip_closed_thresh`` (default 0.06 ≈ partially
        closed around ~3 cm tape thickness).
        """
        obs = self._franka_env.get_observation()
        g0 = float(np.sum(np.abs(obs.get("robot0_gripper_qpos", np.zeros(2)))))
        g1 = float(np.sum(np.abs(obs.get("robot1_gripper_qpos", np.zeros(2)))))
        if g0 < grip_closed_thresh and g1 >= grip_closed_thresh:
            raise HandoverPickerMismatchError(
                f"PyRoki handover oracle is hard-coded for arm1-led picks but "
                f"detected arm0 closed (g0_span={g0:.3f}) with arm1 open "
                f"(g1_span={g1:.3f}). Fall back to the RLT actor / VLA for "
                f"this episode or implement a mirrored macro."
            )

    # ------------------------------------------------------------------
    # Macro entry point.
    # ------------------------------------------------------------------
    def run(self) -> HandoverMacroResult:
        """Execute the handover segment.

        Steps (numbering matches the comment headers in the reference oracle):

        1. Compute the handover geometry from the current arm0/arm1 pose
           midpoint and the configured ``x/y/angle_shift``.
        2. Arm1 goes to ``handover_pos`` with the rotated side quat.
        3. Arm0 opens, approaches via ``handover_pos + (-0.15, -0.15, 0)``,
           then descends with ``z_approach=0.10`` and closes.
        4. Arm1 opens, retracts ``-0.1`` along its gripper z, arm0 retracts
           the same way; arm1 then returns home.

        Returns:
            HandoverMacroResult with ``inner_steps`` = sim-step delta and
            ``succeeded = True`` on a complete run.
        """
        # Push the live geometry params into the env so any privileged-API
        # hook that re-reads them via get_handover_params() sees the same
        # values we use here. Mirrors handover_step.main lines 279-284.
        params = getattr(self._franka_env, "handover_params", None)
        if params is not None:
            params.update({
                "x_shift": self._x_shift,
                "y_shift": self._y_shift,
                "angle_shift": self._angle_shift,
                "perturb_radius": self._perturb_radius,
            })

        # Force a fresh IK solve at the start of each macro: cached cfg from
        # the previous episode could be far from the current pose and bias
        # the vel_cost solver toward an awkward configuration.
        self._api.cfg = None
        self._api.cfg_1 = None

        start_steps = int(getattr(self._franka_env, "_sim_step_count", 0))
        max_steps = int(getattr(self._franka_env, "max_steps", 0)) or None

        try:
            self._execute_handover_segment()
            succeeded = True
            skipped_reason: str | None = None
        except RuntimeError as exc:
            # Most commonly an IK failure inside goto_pose_*; bubble up via
            # result so the caller can log + early-terminate cleanly instead
            # of crashing the whole training run.
            logger.warning("PyRoki handover macro failed mid-run: %s", exc)
            succeeded = False
            skipped_reason = f"runtime_error: {exc}"

        end_steps = int(getattr(self._franka_env, "_sim_step_count", start_steps))
        inner_steps = max(0, end_steps - start_steps)
        if max_steps is not None and end_steps >= max_steps and succeeded:
            # move_to_joints_blocking respects the env step budget but the
            # final motion may have been cut short — surface that to the
            # caller so the loop knows to terminate.
            skipped_reason = "sim_step_budget_exhausted"

        return HandoverMacroResult(
            inner_steps=inner_steps,
            succeeded=succeeded,
            skipped_reason=skipped_reason,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _execute_handover_segment(self) -> None:
        """The post-lift portion of ``build_handover_action_code``."""
        import viser.transforms as vtf  # heavy import — keep it scoped

        api = self._api
        arm1_pos, _ = api.get_arm1_gripper_pose()
        arm0_pos, _ = api.get_arm0_gripper_pose()
        center = (np.asarray(arm1_pos) + np.asarray(arm0_pos)) / 2.0

        handover_pos, arm0_handover_pos, gripper_side_quat, gripper_rotated_side_quat = (
            _handover_geometry(
                center,
                x_shift=self._x_shift,
                y_shift=self._y_shift,
                angle_shift=self._angle_shift,
            )
        )

        # --- (Optional) lateral waypoint perturbation, mirrors the reference. ---
        arm1_sphere = arm0_sphere = None
        if self._perturb_radius > 0:
            sep = handover_pos - arm0_handover_pos
            sep_d = float(np.linalg.norm(sep))
            sep_hat = sep / sep_d if sep_d > 1e-6 else np.array([1.0, 0.0, 0.0])
            mid = (handover_pos + arm0_handover_pos) / 2.0
            arm1_sphere = mid + self._perturb_radius * sep_hat
            arm0_sphere = mid - self._perturb_radius * sep_hat

            arm1_now, _ = api.get_arm1_gripper_pose()
            approach = handover_pos - np.asarray(arm1_now)
            approach_hat = approach / max(np.linalg.norm(approach), 1e-6)
            up = np.array([0.0, 0.0, 1.0])
            lateral = np.cross(approach_hat, up)
            ln = float(np.linalg.norm(lateral))
            lateral = lateral / ln if ln > 1e-6 else np.array([1.0, 0.0, 0.0])
            vert = np.cross(lateral, approach_hat)
            side = float(self._rng.choice([-1.0, 1.0]))
            theta = float(self._rng.uniform(-np.pi / 6, np.pi / 6))
            direction = side * np.cos(theta) * lateral + np.sin(theta) * vert
            direction = direction / max(np.linalg.norm(direction), 1e-6)
            waypoint = arm1_sphere + self._perturb_radius * direction
            api.goto_pose_arm1(waypoint, gripper_rotated_side_quat)

        # --- Step 1: arm1 -> handover ---
        api.goto_pose_arm1(handover_pos, gripper_rotated_side_quat)

        # --- Step 2: arm0 -> approach + grasp ---
        arm0_quat = np.asarray(gripper_side_quat)
        api.open_gripper_arm0()
        # Lateral approach waypoint (matches the reference oracle even when
        # perturb_radius=0, since the constant offset is independent of it).
        approach_offset = np.array([-0.15, -0.15, 0.0], dtype=np.float64)
        api.goto_pose_arm0(arm0_handover_pos + approach_offset, arm0_quat)
        api.goto_pose_arm0(arm0_handover_pos, arm0_quat, z_approach=0.10)
        api.close_gripper_arm0()

        # --- Step 3: arm1 release + retract, arm0 retract, arm1 home ---
        api.open_gripper_arm1()
        gr_R = vtf.SO3(wxyz=gripper_rotated_side_quat).as_matrix()
        a0_R = vtf.SO3(wxyz=arm0_quat).as_matrix()
        retract_local = np.array([0.0, 0.0, -0.1], dtype=np.float64)
        shifted_handover = handover_pos + gr_R @ retract_local
        shifted_arm0 = arm0_handover_pos + a0_R @ retract_local
        api.goto_pose_arm0(shifted_arm0, arm0_quat)
        api.goto_pose_arm1(shifted_handover, gripper_rotated_side_quat)
        api.goto_home_joint_position_arm1()


__all__ = [
    "HandoverMacroResult",
    "HandoverPickerMismatchError",
    "PyrokiHandoverOracle",
]
