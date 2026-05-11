"""Phase-aware shaped reward for the bimanual tape-handover task.

Per-step reward is

    r_t = milestone_bonus · max(0, phase_max_t − phase_max_{t−1})
        + (potential_difference_term  if mode == "dense_sticky")
        + success_bonus · I[task_completed at step t]

Modes:

``lift_only`` (recommended starting point)
    Single observable: is the yellow tape ``> lift_thresh`` above the table?
    Phase taxonomy collapses to ``{0=on table, 1=lifted}``, with a final
    ``PHASE_PLACE=4`` jump on success. Reward is the sticky milestone bonus
    when the tape is first lifted, plus the success terminal. No grip or
    proximity gating, so detection cannot silently fail on edge-case grasps.
    This is the right default when downstream gating (e.g. swap to RLT
    actor when the tape is off the table) only needs a single coarse signal.

``lift_handover``
    Tape lift height plus **which arm first grasps** while lifted. On the first
    step the yellow tape is lifted, we record ``first_grasp_arm`` ∈ {0, 1} as the
    arm whose gripper is closed alone; if both are closed that timestep we break
    ties with the eef **closer to the yellow tape**. Phase taxonomy is
    ``{0=on table, 1=lifted (pre-handoff), 3=lifted + *that* arm fully open, 4=success}``
    — phase ``2`` is skipped so ``milestone_bonus·Δphase_max`` still pays two
    steps on 1→3 and ``end_actor_after_handover`` (``phase_max >= 3``) is unchanged.
    Phase 3 means the **original holder** opened (not the idle arm that stayed
    open throughout a robot1-led pick). After phase 3, ``end_on_tape_drop``
    stays debounced like before.

``sticky``
    Five monotone phases — reach, grasp0, handoff_pose, grasp1, place. Pure
    step-function progress: reward fires only on phase advance + success.
    More credit-assignment signal than ``lift_only`` but relies on grip- and
    proximity-based detectors that can miss real progress (e.g. arm0 grasps
    just outside ``near_thresh``).

``dense_sticky``
    ``sticky`` plus Ng-1999 potential-based difference shaping inside each
    phase: ``r_smooth = γ Φ(s_{t+1}) − Φ(s_t)``. Optimality-preserving in
    theory; in practice the dense gradient can fight credit assignment when
    phase detection is noisy.

Gripper / proximity gates in the multi-phase modes are intentionally
conservative to avoid the upstream ``reward_shaping=True`` failure modes:

- The upstream lift bonus rewards ``yellow_z > table+0.05`` regardless of grip,
  so a slingshot off the table farms +0.5. We gate phase 1 on
  ``g0_closed ∧ near0`` so the lift must come with a real grasp.
- The upstream reach term keeps rewarding arm 0 ↔ yellow distance after the
  handover, fighting the right behavior (arm 0 should release). We switch the
  reach target to arm 1 in phase 2+, and reward arm-0 retreat in phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Phase IDs (also used as "actor_task_phases" config values in stage B).
PHASE_REACH = 0
PHASE_GRASP0 = 1
PHASE_HANDOFF = 2
PHASE_GRASP1 = 3
PHASE_PLACE = 4
NUM_PHASES = 5

# -----------------------------------------------------------------------------
# Tape-drop early termination (``Stage2Config.end_on_tape_drop``)
# -----------------------------------------------------------------------------
# Before handover completes, several *consecutive* steps with ``phase_now ==
# PHASE_REACH`` after ``phase_max >= PHASE_GRASP0`` mean the tape truly fell during
# lift/handoff (short streak debounces phase-detector flicker; see constants below).
#
# After ``lift_handover`` reaches ``PHASE_GRASP1`` (=3), ``yellow_lifted`` /
# ``phase_now`` can flicker below threshold while arm1 still holds the tape
# during the place phase — the old instantaneous rule killed episodes around
# ~100–200 steps with success_bonus never reached. We therefore **never** apply
# that heuristic past handover in ``lift_handover`` mode.
#
# Pre-handover we still need debouncing: ``phase_now`` can flicker to REACH for a
# handful of sim steps while the tape is mid-air (lift height near ``lift_thresh``,
# or grip/near predicates briefly failing in ``sticky``/``dense_sticky``) even though
# ``phase_max`` stays ≥ GRASP0. The old ``pre_streak=1`` rule terminated entire
# episodes on single-frame misclassification. Match the same order of magnitude as
# post-handover streak so drops remain detectable within ~0.3–0.8s at 20 Hz control.
TAPE_DROP_PRE_HANDOVER_STREAK = 12
TAPE_DROP_POST_HANDOVER_STREAK = 16


def tape_drop_episode_should_end(
    *,
    shaped_reward_mode: str,
    end_on_tape_drop: bool,
    phase_now: int,
    phase_max_seen: int,
    streak: int,
    pre_streak: int = TAPE_DROP_PRE_HANDOVER_STREAK,
    post_streak: int = TAPE_DROP_POST_HANDOVER_STREAK,
) -> tuple[bool, int]:
    """Whether to force episode termination because the tape fell back down.

    Returns ``(terminate_now, streak_next)``. The caller persists ``streak_next``
    across env steps and resets both outputs on ``env.reset``.

    Parameters mirror the rollout logic in ``scripts/train_rlt_stage2.py``:
    ``phase_max_seen`` is the running max of ``info[\"phase_max\"]`` so far in
    the episode (including the current step).
    """
    if not end_on_tape_drop or phase_max_seen < PHASE_GRASP0:
        return False, 0
    looks_like_drop = phase_now < PHASE_GRASP0
    if not looks_like_drop:
        return False, 0

    if shaped_reward_mode == "lift_handover" and phase_max_seen >= PHASE_GRASP1:
        return False, 0

    need = pre_streak if phase_max_seen < PHASE_GRASP1 else post_streak
    streak_next = streak + 1
    return (streak_next >= need, streak_next)


@dataclass
class ShapedRewardConfig:
    """Tunables for :func:`compute_shaped_reward`.

    Defaults are chosen so a successful episode earns roughly
    ``5 · milestone_bonus + success_bonus ≈ 3.5`` of *sticky* reward, with
    additional dense-but-bounded contribution from the Ng-style potential
    difference (typically O(0.1) per phase entered). The total order of
    magnitude is intentionally similar to the BC-regularized actor loss so
    neither term dominates.
    """

    mode: str = "off"  # "off" | "lift_only" | "lift_handover" | "sticky" | "dense_sticky"

    # Per-phase potential scale (Φ ∈ [0, h] for the active phase only).
    reach_h: float = 0.20
    grasp0_h: float = 0.20
    handoff_h: float = 0.20
    grasp1_h: float = 0.25
    place_h: float = 0.30

    # Discount used in the potential-difference shaping ``r = γ Φ' − Φ``.
    # Should match ``Stage2Config.discount`` for theoretical optimality
    # preservation, but small mismatches are fine in practice.
    potential_discount: float = 0.99

    # One-time bonus when ``phase_max`` advances. Multiplied by the number of
    # phases skipped, so a 0→2 jump still pays both transitions.
    milestone_bonus: float = 0.50

    # Added on the success step, in addition to whatever potential / milestone
    # contributions land that same step.
    success_bonus: float = 1.0

    # Detector thresholds.
    near_thresh: float = 0.04           # m, eef ↔ tape distance.
    # Finger qpos span (= sum |qpos| over both fingers, range [0, 0.08] for
    # PandaGripper) thresholds for "closed enough to be holding something" and
    # "fully released". Yellow/duct tape are grasped across their narrow
    # ~2.5 cm dimension (the ~9.5 cm diameter exceeds the gripper's ~8 cm
    # max opening), giving span ≈ 0.025 mid-grasp. The previous 0.02 / 0.05
    # defaults were tuned for fully-closed-empty and fully-open and **never
    # fired during an actual tape grasp**, which prevented phase 1 from ever
    # being detected: the only way phase_max advanced past 0 was the
    # ``success → PHASE_PLACE`` shortcut, dumping all milestones into the
    # final step. That broke (a) per-step shaped reward (ep_reward stayed at 0
    # except for the success step), (b) ``end_on_tape_drop`` (gated on
    # ``phase_max >= PHASE_GRASP0``), and (c) ``phase_mode=task_phase`` (gated
    # on the same ``phase_max``).
    # 0.06 captures any grasp ≤ ~3 cm thick + fully closed; 0.07 keeps a
    # 0.06–0.07 hysteresis band so a partially-released grasp is not
    # prematurely marked "open" (which would let Phase 3 fire before arm0
    # has fully let go). Override per-task via ShapedRewardConfig if your
    # objects are very different.
    grip_closed_thresh: float = 0.06    # finger qpos span < threshold => closed/holding.
    grip_open_thresh: float = 0.07      # finger qpos span > threshold => fully open.
    # Lift detection threshold: yellow tape z must exceed ``table_top_z + lift_thresh``.
    # 2 cm is small enough to fire reliably as soon as the tape leaves the table during
    # a real grasp, but well above the ~5 mm of pose noise we see when it sits on the
    # table after placement initialization.
    lift_thresh: float = 0.02
    handoff_y_thresh: float = 0.10      # |yellow_y - midline_y|.
    midline_y: float = 0.0              # Y where the handoff is "between" the arms.
    retreat_dist: float = 0.15          # Phase-3 target: arm0 at least this far from yellow.

    # Tanh sharpness for distance potentials. Higher = steeper gradient near 0.
    tanh_k: float = 10.0


@dataclass
class PhaseTrackerState:
    """Per-episode mutable state. Reset in ``RobosuiteRLTEnv.reset``."""

    phase_max: int = 0
    prev_phase_max: int = 0
    # Last-step potential value for the Ng-1999 potential-difference shaping.
    # Reset to the *current* potential at every phase transition (see
    # :func:`compute_shaped_reward`) so phase-boundary discontinuities do not
    # leak into ``r_smooth`` — milestone_bonus pays the transition instead.
    prev_phi: float = 0.0
    last_phase_for_phi: int = -1  # -1 forces a phi-reset on the first step.
    # For diagnostics — number of steps spent at each phase_max.
    steps_at_phase: list[int] = field(default_factory=lambda: [0] * NUM_PHASES)
    # ``lift_handover`` only: which arm (0 or 1) first reads grasped while the
    # tape is lifted; phase 3 requires *that* gripper's span to exceed
    # ``grip_open_thresh`` (picker / original holder released).
    lift_handover_first_grasp_arm: int | None = None


def _gripper_span(gripper_qpos: np.ndarray) -> float:
    """Return ``sum(|qpos|)`` proxy for finger opening (0 closed, ~0.08 open)."""
    return float(np.sum(np.abs(np.asarray(gripper_qpos, dtype=np.float64))))


def _read_sim_pose(sim: Any, body_id: int) -> np.ndarray:
    return np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)


def _read_eef_pos(sim: Any, site_id: int) -> np.ndarray:
    return np.asarray(sim.data.site_xpos[site_id], dtype=np.float64)


def compute_phase_now(
    raw_obs: dict[str, Any],
    sim: Any,
    body_ids: dict[str, int],
    site_ids: dict[str, int],
    table_top_z: float,
    cfg: ShapedRewardConfig,
    *,
    phase_max: int,
    success: bool,
) -> tuple[int, dict[str, float]]:
    """Detect which phase the current sim state belongs to.

    Returns ``(phase_now, debug)``. ``phase_max`` is consulted only to gate
    higher phases (a phase 3 detector requires ``phase_max >= 1``); detection
    itself is otherwise stateless.
    """
    yellow = _read_sim_pose(sim, body_ids["yellow"])
    duct = _read_sim_pose(sim, body_ids["duct"])
    eef0 = _read_eef_pos(sim, site_ids["eef0"])
    eef1 = _read_eef_pos(sim, site_ids["eef1"])

    g0_span = _gripper_span(raw_obs.get("robot0_gripper_qpos", np.zeros(2)))
    g1_span = _gripper_span(raw_obs.get("robot1_gripper_qpos", np.zeros(2)))

    dist_e0_yellow = float(np.linalg.norm(eef0 - yellow))
    dist_e1_yellow = float(np.linalg.norm(eef1 - yellow))
    yellow_lift_height = float(yellow[2] - table_top_z)
    yellow_lifted = yellow_lift_height > cfg.lift_thresh

    debug = {
        "dist_e0_yellow": dist_e0_yellow,
        "dist_e1_yellow": dist_e1_yellow,
        "dist_yellow_duct": float(np.linalg.norm(yellow[:2] - duct[:2])),
        "yellow_lifted": float(yellow_lifted),
        "yellow_lift_height": yellow_lift_height,
        "yellow_y": float(yellow[1]),
        "g0_span": g0_span,
        "g1_span": g1_span,
    }

    # Success terminal collapses everything to PHASE_PLACE regardless of mode.
    if success:
        return PHASE_PLACE, debug

    # ``lift_only``: the simplest possible detector. Phase 1 = tape off the table.
    # No grip / proximity coupling, so the detector cannot silently fail when an
    # edge-case grasp keeps the gripper a few millimetres past ``near_thresh``.
    if cfg.mode == "lift_only":
        return (PHASE_GRASP0 if yellow_lifted else PHASE_REACH), debug

    g0_closed = g0_span < cfg.grip_closed_thresh
    g1_closed = g1_span < cfg.grip_closed_thresh
    g0_open = g0_span > cfg.grip_open_thresh

    # ``lift_handover``: REACH vs lifted placeholder only. Phase 3 is resolved in
    # :func:`compute_shaped_reward` using ``lift_handover_first_grasp_arm`` so we
    # wait for the **pick** arm to open, not whichever arm stayed idle-open.
    if cfg.mode == "lift_handover":
        if not yellow_lifted:
            return PHASE_REACH, debug
        return PHASE_GRASP0, debug
    near0 = dist_e0_yellow < cfg.near_thresh
    near1 = dist_e1_yellow < cfg.near_thresh

    # Phase 3: arm1 has the tape, arm0 has released, tape still up.
    if phase_max >= PHASE_GRASP0 and near1 and g1_closed and g0_open and yellow_lifted:
        phase_now = PHASE_GRASP1
    # Phase 2: tape lifted near midline (handoff zone). Requires arm0 already
    # grasped at some point.
    elif phase_max >= PHASE_GRASP0 and yellow_lifted and abs(yellow[1] - cfg.midline_y) < cfg.handoff_y_thresh:
        phase_now = PHASE_HANDOFF
    # Phase 1: arm0 has yellow + lifted off table.
    elif near0 and g0_closed and yellow_lifted:
        phase_now = PHASE_GRASP0
    else:
        phase_now = PHASE_REACH

    return phase_now, debug


def _phase_potential(
    phase_now: int,
    raw_obs: dict[str, Any],
    sim: Any,
    body_ids: dict[str, int],
    site_ids: dict[str, int],
    cfg: ShapedRewardConfig,
) -> float:
    """Smooth potential Φ for the *current* phase only.

    Each phase has its own scalar in [0, ``h_phase``]; phases that are not
    currently active contribute 0. This is what stops the policy from farming
    e.g. reach reward by parking near yellow tape — once it grasps, reach
    stops paying, and the only way to keep earning is to advance.
    """
    yellow = _read_sim_pose(sim, body_ids["yellow"])
    duct = _read_sim_pose(sim, body_ids["duct"])
    eef0 = _read_eef_pos(sim, site_ids["eef0"])
    eef1 = _read_eef_pos(sim, site_ids["eef1"])
    k = cfg.tanh_k

    if phase_now == PHASE_REACH:
        d = float(np.linalg.norm(eef0 - yellow))
        return cfg.reach_h * (1.0 - float(np.tanh(k * d)))

    if phase_now == PHASE_GRASP0:
        # Encourage moving the (held) tape toward the midline so the handoff
        # zone becomes reachable for arm1.
        dy = abs(float(yellow[1]) - cfg.midline_y)
        return cfg.grasp0_h * (1.0 - float(np.tanh(k * dy)))

    if phase_now == PHASE_HANDOFF:
        d = float(np.linalg.norm(eef1 - yellow))
        return cfg.handoff_h * (1.0 - float(np.tanh(k * d)))

    if phase_now == PHASE_GRASP1:
        # Reward arm0 retreating away from the tape (so it doesn't fight arm1).
        d = float(np.linalg.norm(eef0 - yellow))
        progress = min(1.0, d / max(cfg.retreat_dist, 1e-6))
        return cfg.grasp1_h * progress

    if phase_now == PHASE_PLACE:
        d = float(np.linalg.norm(yellow[:2] - duct[:2]))
        # Use 2x-scale tanh on placing so the gradient survives at the
        # success threshold (~0.08 m). Otherwise tanh saturates earlier.
        return cfg.place_h * (1.0 - float(np.tanh(k * 0.5 * d)))

    return 0.0


def compute_shaped_reward(
    *,
    raw_obs: dict[str, Any],
    sim: Any,
    body_ids: dict[str, int],
    site_ids: dict[str, int],
    table_top_z: float,
    success: bool,
    state: PhaseTrackerState,
    cfg: ShapedRewardConfig,
) -> tuple[float, dict[str, Any]]:
    """Single per-step reward evaluator.

    Mutates ``state`` in place to advance ``phase_max`` and update
    ``steps_at_phase``. Returns ``(reward, info)`` where ``info`` is suitable
    for merging into ``env.step`` output.
    """
    phase_now, dbg = compute_phase_now(
        raw_obs, sim, body_ids, site_ids, table_top_z, cfg,
        phase_max=state.phase_max, success=success,
    )
    if cfg.mode == "lift_handover" and phase_now != PHASE_PLACE:
        y_lift = float(dbg["yellow_lifted"]) >= 0.5
        g0_span_v = float(dbg["g0_span"])
        g1_span_v = float(dbg["g1_span"])
        g0_open = g0_span_v > cfg.grip_open_thresh
        g1_open = g1_span_v > cfg.grip_open_thresh
        g0_closed = g0_span_v < cfg.grip_closed_thresh
        g1_closed = g1_span_v < cfg.grip_closed_thresh

        if y_lift and state.lift_handover_first_grasp_arm is None:
            if g0_closed and not g1_closed:
                state.lift_handover_first_grasp_arm = 0
            elif g1_closed and not g0_closed:
                state.lift_handover_first_grasp_arm = 1
            elif g0_closed and g1_closed:
                d0 = float(dbg["dist_e0_yellow"])
                d1 = float(dbg["dist_e1_yellow"])
                state.lift_handover_first_grasp_arm = 0 if d0 <= d1 else 1

        if not y_lift:
            phase_now = PHASE_REACH
        elif (
            state.phase_max >= PHASE_GRASP0
            and state.lift_handover_first_grasp_arm is not None
            and (
                (state.lift_handover_first_grasp_arm == 0 and g0_open)
                or (state.lift_handover_first_grasp_arm == 1 and g1_open)
            )
        ):
            phase_now = PHASE_GRASP1
        else:
            phase_now = PHASE_GRASP0
    new_phase_max = max(state.phase_max, phase_now)

    # --- Within-phase shaping ---
    # ``sticky`` mode emits 0 inside a phase — only the sticky milestone bonus
    # below pays for progress.  ``dense_sticky`` adds Ng-1999 potential-based
    # difference shaping ``r_smooth = γ Φ(s_{t+1}) − Φ(s_t)``, which is
    # optimality-preserving (Ng 1999 Theorem 1) but adds gradient noise from
    # phase-detection flicker. We always *evaluate* Φ for logging.
    phi_now = _phase_potential(phase_now, raw_obs, sim, body_ids, site_ids, cfg)
    if cfg.mode == "dense_sticky":
        if phase_now != state.last_phase_for_phi:
            shaping = 0.0
            state.prev_phi = phi_now
            state.last_phase_for_phi = phase_now
        else:
            # One-sided potential shaping: only credit positive progress, never
            # penalize stalling or transient decreases. Without this clamp,
            # the (γ−1)·Φ stalling term accumulates ~−Φ·(1−γ) per step, which
            # over a long episode (e.g. 1800 steps × 0.002 ≈ −3.6) dominates
            # the +1.0 success bonus and makes successful slow episodes net
            # *more negative* than fast failures.
            raw_shaping = cfg.potential_discount * phi_now - state.prev_phi
            shaping = max(0.0, raw_shaping)
            state.prev_phi = phi_now
    else:
        shaping = 0.0
        state.prev_phi = phi_now
        state.last_phase_for_phi = phase_now

    milestone = cfg.milestone_bonus * float(max(0, new_phase_max - state.prev_phase_max))
    success_term = cfg.success_bonus if success else 0.0

    reward = shaping + milestone + success_term

    state.prev_phase_max = new_phase_max
    state.phase_max = new_phase_max
    state.steps_at_phase[phase_now] += 1

    info = {
        "phase": int(phase_now),
        "phase_max": int(new_phase_max),
        "shaping_reward": float(shaping),
        "milestone_reward": float(milestone),
        "success_reward": float(success_term),
        "phi_now": float(phi_now),
        **dbg,
    }
    if cfg.mode == "lift_handover":
        fa = state.lift_handover_first_grasp_arm
        info["lh_first_grasp_arm"] = float(fa) if fa is not None else -1.0
    return float(reward), info


def _resolve_eef_site_id(robot: Any) -> int:
    """Return the MJCF site ID for the robot's main end-effector.

    ``robot.eef_site_id`` is a dict keyed by ``robot.arms`` (e.g. ``["right"]``
    for a single-arm Panda, ``["right", "left"]`` for a Baxter). The bimanual
    tape-handover env uses two single-arm Pandas in ``parallel`` configuration,
    so each robot exposes exactly one arm. We pick the first arm in the dict to
    stay agnostic to single- vs dual-arm robots.
    """
    eef = robot.eef_site_id
    if isinstance(eef, dict):
        if "right" in eef:
            return int(eef["right"])
        # Fallback for any future single-arm robot whose key isn't "right".
        return int(next(iter(eef.values())))
    # Older robosuite revisions may expose a scalar; preserve that path.
    return int(eef)


def cache_sim_handles(robosuite_env: Any) -> tuple[dict[str, int], dict[str, int], float]:
    """Snapshot body / site IDs and the table top z used for lift detection.

    Called once at env construction; the values are constant for the lifetime
    of the underlying ``TwoArmTapeHandover`` instance (re-cache on hard reset
    if the model is rebuilt — see ``RobosuiteRLTEnv.reset`` notes).
    """
    body_ids = {
        "yellow": int(robosuite_env.yellow_tape_body_id),
        "duct": int(robosuite_env.duct_tape_body_id),
    }
    site_ids = {
        "eef0": _resolve_eef_site_id(robosuite_env.robots[0]),
        "eef1": _resolve_eef_site_id(robosuite_env.robots[1]),
    }
    # ``table_offsets[0, 2]`` is the table-top z used everywhere upstream
    # (placement initializer, lift bonus). Mirror that here.
    table_top_z = float(robosuite_env.table_offsets[0, 2])
    return body_ids, site_ids, table_top_z
