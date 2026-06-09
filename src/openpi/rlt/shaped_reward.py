"""Phase-aware shaped reward for the bimanual tape-handover task (lift_handover mode).

Per-step reward:

    r_t = milestone_bonus · max(0, phase_max_t − phase_max_{t−1})
        + lh_shortcut_penalty   (if anti-shortcut triggers, typically ≤ 0)
        + success_bonus · I[effective_success at step t]

Phases (PHASE_HANDOFF = 2 is skipped so milestone_bonus·Δphase pays correctly on 1→3):

    PHASE_REACH  (0)  tape on table
    PHASE_GRASP0 (1)  tape lifted; picker arm identified, still holding
    PHASE_GRASP1 (3)  handover done — picker opened, receiver near+closed on tape
    PHASE_PLACE  (4)  task success

Phase detection for lift_handover
----------------------------------
On the first step the tape is lifted we record ``lift_handover_first_grasp_arm`` ∈ {0,1}
as the arm whose gripper is closed alone; ties broken by eef distance to the tape.
Phase 3 requires *that* arm to be fully open AND the other arm to be near+closed —
simply opening the picker without the receiver grasping does not advance phase.

Anti-shortcut features
-----------------------
``lift_handover_shortcut_xy_thresh_m > 0``:
    Add ``lift_handover_shortcut_penalty_per_step`` (set negative, e.g. -0.002) every
    step where the tape is lifted, phase_max < PHASE_GRASP1, AND the yellow–duct XY
    distance is below the threshold. Discourages throwing / sliding to goal.

``lift_handover_gate_success_bonus = True``:
    When task success fires before phase_max reached PHASE_GRASP1, scale success_bonus
    by ``lift_handover_shortcut_success_bonus_frac`` (e.g. 0.15) instead of paying full.

Tape-drop early termination
-----------------------------
``tape_drop_episode_should_end`` detects when the tape fell back to the table after
being lifted.  Pre-handover it requires 12 consecutive REACH steps; post-handover
(phase_max >= PHASE_GRASP1) the check is suppressed to avoid false kills during
placement.  See function docstring for details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Phase constants (also used as "actor_task_phases" config values in stage B)
# ---------------------------------------------------------------------------
PHASE_REACH = 0
PHASE_GRASP0 = 1
PHASE_HANDOFF = 2   # skipped in lift_handover, reserved for future use
PHASE_GRASP1 = 3
PHASE_PLACE = 4
NUM_PHASES = 5

# ---------------------------------------------------------------------------
# Tape-drop debounce streaks
# ---------------------------------------------------------------------------
# Pre-handover: 12 consecutive REACH steps after phase_max >= PHASE_GRASP0
# debounces single-frame phase-detector flicker (~0.6 s at 20 Hz control).
# Post-handover (phase_max >= PHASE_GRASP1): check is suppressed entirely
# because yellow_lifted / phase_now can flicker while arm1 moves to place.
TAPE_DROP_PRE_HANDOVER_STREAK = 12
TAPE_DROP_POST_HANDOVER_STREAK = 16  # kept for callers that pass post_streak explicitly


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
    across env steps and resets both on ``env.reset``.

    ``phase_max_seen`` is the running max of ``info["phase_max"]`` so far in
    the episode (including the current step).
    """
    if not end_on_tape_drop or phase_max_seen < PHASE_GRASP0:
        return False, 0
    if phase_now >= PHASE_GRASP0:
        return False, 0

    # In lift_handover mode: never trigger after the handover is complete —
    # arm1 can dip near the table during placement without meaning the tape dropped.
    if shaped_reward_mode == "lift_handover" and phase_max_seen >= PHASE_GRASP1:
        return False, 0

    need = pre_streak if phase_max_seen < PHASE_GRASP1 else post_streak
    streak_next = streak + 1
    return (streak_next >= need, streak_next)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ShapedRewardConfig:
    """Tunables for :func:`compute_shaped_reward`.

    A successful episode earns roughly ``2 · milestone_bonus + success_bonus``
    (REACH→GRASP0 +0.5, GRASP0→GRASP1 +0.5, success +1.0 = 2.0 total by default).
    """

    mode: str = "off"   # "off" | "lift_handover"

    # One-time bonus each time phase_max advances (multiplied by phases skipped).
    milestone_bonus: float = 0.50
    # Terminal bonus on the effective-success step.
    success_bonus: float = 1.0

    # --- Detector thresholds ---
    # eef ↔ tape distance (m) for "near enough to be grasping".
    near_thresh: float = 0.04
    # Finger qpos span = sum(|qpos|), range [0, 0.08] for PandaGripper.
    # 0.06 captures any grasp ≤ ~3 cm thick; 0.07 hysteresis avoids premature
    # "open" detection on a partially-released grasp.
    grip_closed_thresh: float = 0.06
    grip_open_thresh: float = 0.07
    # Yellow tape z must exceed table_top_z + lift_thresh to count as lifted.
    lift_thresh: float = 0.02

    # --- Anti-shortcut (lift_handover) ---
    # Per-step penalty when tape is lifted close to duct tape but handover not done.
    # Set to a negative value (e.g. -0.002). Disabled when thresh is 0.
    lift_handover_shortcut_xy_thresh_m: float = 0.0
    lift_handover_shortcut_penalty_per_step: float = 0.0
    # Gate the success bonus: if success fires before PHASE_GRASP1, scale it by
    # shortcut_success_bonus_frac instead (e.g. 0.15 means 15% of success_bonus).
    lift_handover_gate_success_bonus: bool = False
    lift_handover_shortcut_success_bonus_frac: float = 0.0


# ---------------------------------------------------------------------------
# Per-episode state
# ---------------------------------------------------------------------------

@dataclass
class PhaseTrackerState:
    """Mutable per-episode state. Reset via ``RobosuiteRLTEnv.reset``."""

    phase_max: int = 0
    prev_phase_max: int = 0
    # Per-phase step counters for diagnostics.
    steps_at_phase: list[int] = field(default_factory=lambda: [0] * NUM_PHASES)
    # lift_handover: index (0 or 1) of the arm that first grasped while lifted.
    # None until the tape is first seen lifted with a closed gripper.
    lift_handover_first_grasp_arm: int | None = None


# ---------------------------------------------------------------------------
# Sim read helpers
# ---------------------------------------------------------------------------

def _gripper_span(gripper_qpos: np.ndarray) -> float:
    """sum(|qpos|) proxy for finger opening: 0 = fully closed, ~0.08 = fully open."""
    return float(np.sum(np.abs(np.asarray(gripper_qpos, dtype=np.float64))))


def _read_body_pos(sim: Any, body_id: int) -> np.ndarray:
    return np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)


def _read_site_pos(sim: Any, site_id: int) -> np.ndarray:
    return np.asarray(sim.data.site_xpos[site_id], dtype=np.float64)


# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------

def _read_sim_state(
    raw_obs: dict[str, Any],
    sim: Any,
    body_ids: dict[str, int],
    site_ids: dict[str, int],
    table_top_z: float,
    cfg: ShapedRewardConfig,
) -> dict[str, Any]:
    """Read all sim quantities needed for phase detection and reward into one dict."""
    yellow = _read_body_pos(sim, body_ids["yellow"])
    duct = _read_body_pos(sim, body_ids["duct"])
    eef0 = _read_site_pos(sim, site_ids["eef0"])
    eef1 = _read_site_pos(sim, site_ids["eef1"])
    g0_span = _gripper_span(raw_obs.get("robot0_gripper_qpos", np.zeros(2)))
    g1_span = _gripper_span(raw_obs.get("robot1_gripper_qpos", np.zeros(2)))
    yellow_lift_height = float(yellow[2] - table_top_z)
    yellow_lifted = yellow_lift_height > cfg.lift_thresh
    dist_e0 = float(np.linalg.norm(eef0 - yellow))
    dist_e1 = float(np.linalg.norm(eef1 - yellow))
    return {
        "yellow": yellow,
        "duct": duct,
        "eef0": eef0,
        "eef1": eef1,
        "g0_span": g0_span,
        "g1_span": g1_span,
        "yellow_lift_height": yellow_lift_height,
        "yellow_lifted": yellow_lifted,
        "yellow_y": float(yellow[1]),
        "dist_e0_yellow": dist_e0,
        "dist_e1_yellow": dist_e1,
        "dist_yellow_duct": float(np.linalg.norm(yellow[:2] - duct[:2])),
        "g0_closed": g0_span < cfg.grip_closed_thresh,
        "g1_closed": g1_span < cfg.grip_closed_thresh,
        "g0_open": g0_span > cfg.grip_open_thresh,
        "g1_open": g1_span > cfg.grip_open_thresh,
        "near0": dist_e0 < cfg.near_thresh,
        "near1": dist_e1 < cfg.near_thresh,
    }


def _update_lift_handover_phase(
    s: dict[str, Any],
    state: PhaseTrackerState,
    cfg: ShapedRewardConfig,
) -> tuple[int, bool, bool]:
    """Compute phase_now for lift_handover mode and update first_grasp_arm in state.

    Returns ``(phase_now, receiver_grasped, handover_blocked_no_receiver_grasp)``.

    Logic:
        1. If tape not lifted → PHASE_REACH.
        2. On first lifted step, record which arm is the picker (gripper closed alone;
           ties broken by eef distance).
        3. PHASE_GRASP1 when: tape lifted + picker opened + receiver near+closed.
        4. Otherwise PHASE_GRASP0 (tape lifted, handover not yet complete).
        ``handover_blocked_no_receiver_grasp`` is True when the picker has opened but
        the receiver hasn't grasped yet — useful for diagnostics.
    """
    if not s["yellow_lifted"]:
        return PHASE_REACH, False, False

    # Record picker arm on first lifted step.
    if state.lift_handover_first_grasp_arm is None:
        if s["g0_closed"] and not s["g1_closed"]:
            state.lift_handover_first_grasp_arm = 0
        elif s["g1_closed"] and not s["g0_closed"]:
            state.lift_handover_first_grasp_arm = 1
        elif s["g0_closed"] and s["g1_closed"]:
            state.lift_handover_first_grasp_arm = 0 if s["dist_e0_yellow"] <= s["dist_e1_yellow"] else 1

    picker = state.lift_handover_first_grasp_arm

    if picker == 0:
        receiver_grasped = bool(s["g1_closed"] and s["near1"])
        picker_opened = bool(s["g0_open"])
        handover_done = bool(
            state.phase_max >= PHASE_GRASP0
            and picker_opened and s["g1_closed"] and s["near1"]
        )
    elif picker == 1:
        receiver_grasped = bool(s["g0_closed"] and s["near0"])
        picker_opened = bool(s["g1_open"])
        handover_done = bool(
            state.phase_max >= PHASE_GRASP0
            and picker_opened and s["g0_closed"] and s["near0"]
        )
    else:
        # Picker arm not yet identified (tape just lifted this step, both open).
        return PHASE_GRASP0, False, False

    if handover_done:
        return PHASE_GRASP1, receiver_grasped, False

    blocked = bool(
        state.phase_max >= PHASE_GRASP0
        and picker_opened
        and not receiver_grasped
    )
    return PHASE_GRASP0, receiver_grasped, blocked


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def compute_shaped_reward(
    *,
    raw_obs: dict[str, Any],
    sim: Any,
    body_ids: dict[str, int],
    site_ids: dict[str, int],
    table_top_z: float,
    success: bool,
    base_success: bool = False,
    state: PhaseTrackerState,
    cfg: ShapedRewardConfig,
) -> tuple[float, dict[str, Any]]:
    """Compute per-step reward and advance phase tracker state.

    Mutates ``state`` in place. Returns ``(reward, info)`` where ``info`` can be
    merged directly into ``env.step`` output for logging.

    Args:
        success: Effective (gated) task success for this step — fires the success bonus
            and collapses phase to PHASE_PLACE.
        base_success: Raw robosuite success before any gate (handover / home-return).
            Used only for the anti-shortcut penalty so it doesn't fire after placement.
    """
    phase_max_at_entry = int(state.phase_max)
    receiver_grasped = False
    handover_blocked = False

    s = _read_sim_state(raw_obs, sim, body_ids, site_ids, table_top_z, cfg)

    # Success collapses to PHASE_PLACE regardless of sensor state.
    if success:
        phase_now = PHASE_PLACE
    elif cfg.mode == "lift_handover":
        phase_now, receiver_grasped, handover_blocked = _update_lift_handover_phase(
            s, state, cfg
        )
    else:
        phase_now = PHASE_REACH  # mode == "off": no phase progression

    new_phase_max = max(state.phase_max, phase_now)

    # --- Milestone bonus (fires once per phase advance) ---
    milestone = cfg.milestone_bonus * float(max(0, new_phase_max - state.prev_phase_max))

    # --- Success bonus (optionally discounted on shortcuts) ---
    success_term = cfg.success_bonus if success else 0.0
    if (
        success
        and cfg.mode == "lift_handover"
        and cfg.lift_handover_gate_success_bonus
        and phase_max_at_entry < PHASE_GRASP1
    ):
        success_term *= float(cfg.lift_handover_shortcut_success_bonus_frac)

    # --- Anti-shortcut penalty ---
    # Fires when tape is lifted and close to duct tape but handover not complete.
    # Suppressed once base_success is True (tape already placed).
    lh_shortcut_penalty = 0.0
    if (
        cfg.mode == "lift_handover"
        and cfg.lift_handover_shortcut_xy_thresh_m > 0.0
        and cfg.lift_handover_shortcut_penalty_per_step != 0.0
        and s["yellow_lifted"]
        and new_phase_max < PHASE_GRASP1
        and not bool(base_success)
        and s["dist_yellow_duct"] < cfg.lift_handover_shortcut_xy_thresh_m
    ):
        lh_shortcut_penalty = float(cfg.lift_handover_shortcut_penalty_per_step)

    reward = milestone + success_term + lh_shortcut_penalty

    # --- Advance state ---
    state.prev_phase_max = new_phase_max
    state.phase_max = new_phase_max
    state.steps_at_phase[phase_now] += 1

    info: dict[str, Any] = {
        "phase": int(phase_now),
        "phase_max": int(new_phase_max),
        "milestone_reward": float(milestone),
        "success_reward": float(success_term),
        "lh_shortcut_penalty": float(lh_shortcut_penalty),
        # Sensor readings (logged to wandb).
        "dist_e0_yellow": s["dist_e0_yellow"],
        "dist_e1_yellow": s["dist_e1_yellow"],
        "dist_yellow_duct": s["dist_yellow_duct"],
        "yellow_lifted": float(s["yellow_lifted"]),
        "yellow_lift_height": s["yellow_lift_height"],
        "yellow_y": s["yellow_y"],
        "g0_span": s["g0_span"],
        "g1_span": s["g1_span"],
    }
    if cfg.mode == "lift_handover":
        fa = state.lift_handover_first_grasp_arm
        info["lh_first_grasp_arm"] = float(fa) if fa is not None else -1.0
        info["lh_receiver_grasped"] = float(receiver_grasped)
        info["lh_blocked_no_receiver_grasp"] = float(handover_blocked)

    return float(reward), info


# ---------------------------------------------------------------------------
# Sim handle caching (call once at env construction)
# ---------------------------------------------------------------------------

def _resolve_eef_site_id(robot: Any) -> int:
    """Return the MuJoCo site ID for the robot's main end-effector.

    Handles both dict-keyed (newer robosuite) and scalar (older) ``eef_site_id``.
    """
    eef = robot.eef_site_id
    if isinstance(eef, dict):
        return int(eef["right"]) if "right" in eef else int(next(iter(eef.values())))
    return int(eef)


def cache_sim_handles(robosuite_env: Any) -> tuple[dict[str, int], dict[str, int], float]:
    """Snapshot body/site IDs and table-top z for lift detection.

    Call once at env construction (or after a hard reset that rebuilds the model).
    """
    body_ids = {
        "yellow": int(robosuite_env.yellow_tape_body_id),
        "duct": int(robosuite_env.duct_tape_body_id),
    }
    site_ids = {
        "eef0": _resolve_eef_site_id(robosuite_env.robots[0]),
        "eef1": _resolve_eef_site_id(robosuite_env.robots[1]),
    }
    table_top_z = float(robosuite_env.table_offsets[0, 2])
    return body_ids, site_ids, table_top_z
