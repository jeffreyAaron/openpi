"""Stage 2: Online RL training with actor-critic on RL token representation.

Loads a frozen VLA + frozen RL token encoder from Stage 1, then trains
lightweight actor and critic MLPs via off-policy TD3-style RL in a
Robosuite environment.

Usage:
    uv run scripts/train_rlt_stage2.py \
        --vla_config_name pi05_tsh \
        --vla_checkpoint_path /path/to/vla/checkpoint \
        --rl_token_checkpoint_path /path/to/stage1/checkpoint/rl_token.safetensors \\
        --exp_name rlt_stage2

    ``--rl_token_checkpoint_path`` may be ``.safetensors`` (Stage 1 default) or ``.pt``.

GTE-aligned warm-up / rollouts (full VLA horizon + same smoothing as groundTruthEval), e.g.:
    ... train_rlt_stage2.py ... --match_ground_truth_eval_rollout \\
        --match_ground_truth_eval_mode temporal \\
        --robosuite_tape_offsets_json /path/to/handover_index.json
    Use ``--match_ground_truth_eval_mode none`` to match ``plot_gte_vs_rlt2_rollout_actions.py
    --smoothing none``. VLA-only sanity check: add ``--vla_only``.

Phased control (VLA -> RLT actor -> VLA): let the frozen VLA handle the easy approach +
recovery and only let the RL actor drive a contact-rich middle window, e.g. steps 400-1200:
    ... train_rlt_stage2.py ... \\
        --phase_mode chunk \\
        --vla_phase_end_step 400 \\
        --rl_phase_end_step 1200
    The replay buffer is filled only from transitions inside ``[x, y)`` (with ``done=1.0`` forced at
    the boundary so the critic doesn't bootstrap past it) — including warm-up episodes, where the
    executed actions are VLA. This seeds the actor with VLA-quality data over exactly the state
    distribution it will eventually control.
    During warm-up, failed/timeout rollouts are excluded from the replay buffer by default
    (only successful warm-up episodes seed the buffer). Post-warmup episodes are always
    added so the actor learns from real failures. Use ``--no-replay_only_successful`` to
    keep failed warm-up episodes too.

Tape-handover early termination + control hand-back (default on, requires shaped reward):
    * ``--end_on_tape_drop`` (default): episode ends when the shaped-reward detector
      concludes the tape fell after lift (see ``tape_drop_episode_should_end`` in
      ``openpi.rlt.shaped_reward`` — ``lift_handover`` disables this after phase_max≥3
      so place-phase lift jitter does not truncate episodes). The terminal step records
      ``done=True`` so the critic learns the drop is a sink with no terminal reward.
    * ``--end_actor_after_handover`` (default): once arm 1 has the tape (phase_max >=
      PHASE_GRASP1=3), control hands back to the frozen VLA at the next chunk
      boundary regardless of ``--phase_mode`` / ``--actor_task_phases``.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import logging
import os
import pathlib
import tempfile
import threading
import time
from typing import Any

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import safetensors.torch
import torch
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
from openpi.models.model import Observation
from openpi.policies import policy_config as openpi_policy_config
from openpi.policies.policy import Policy
from openpi.rlt.action_smoothing import ActionSmoothingRuntime
from openpi.rlt.actor_critic import (
    GaussianActor,
    TwinQCritic,
    create_target_critic,
    soft_update_target,
)
from openpi.rlt.async_learner import AsyncLearner
from openpi.rlt.config import RLTokenModelConfig, Stage2Config
from openpi.rlt.env_interface import RLEnvironment
from openpi.rlt.replay_buffer import ReplayBuffer, Transition
from openpi.rlt.rl_token import RLTokenEncoder
from openpi.rlt.shaped_reward import PHASE_GRASP1, tape_drop_episode_should_end
from openpi.rlt.vla_wrapper import VLAEmbeddingExtractor


logger = logging.getLogger(__name__)

_RL_TOKEN_CFG_DEFAULTS = RLTokenModelConfig()


def init_logging() -> None:
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        root.addHandler(ch)
    else:
        root.handlers[0].setFormatter(formatter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLT Stage 2: Online RL")
    parser.add_argument("--vla_config_name", type=str, default="pi05_tsh")
    parser.add_argument("--vla_checkpoint_path", type=str, required=True)
    parser.add_argument("--rl_token_checkpoint_path", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="rlt_stage2")
    parser.add_argument("--checkpoint_base_dir", type=str, default="./checkpoints/rlt")
    parser.add_argument(
        "--init_actor_critic_path",
        type=str,
        default=None,
        help="Optional warm-start. Path to a Stage 2-style checkpoint directory containing "
        "actor.safetensors and critic.safetensors (e.g. produced by "
        "scripts/pretrain_rlt_actor_critic.py). If provided, the actor + critic weights are "
        "loaded into the freshly built networks before training starts; the target critic is "
        "re-derived from the loaded critic. Optimizer state is NOT restored — fresh AdamW state "
        "is used so online RL hyperparameters (lr, target_noise_std, etc.) take effect cleanly.",
    )

    # Stage 2 config overrides.
    parser.add_argument("--rl_chunk_length", type=int, default=50)
    parser.add_argument("--action_dim", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--actor_hidden_dim", type=int, default=256)
    parser.add_argument("--actor_num_layers", type=int, default=2)
    parser.add_argument("--actor_fixed_std", type=float, default=0.03,
                        help="Gaussian std on flattened chunk when using stochastic rollout / actor.forward().")
    parser.add_argument("--actor_stochastic_rollout", action="store_true",
                        help="Add exploration noise during env rollout. Default: execute actor mean only (smoother).")
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument(
        "--actor_delta_max",
        type=float,
        default=1.0,
        help="Maximum per-element residual the actor may add on top of the VLA reference. "
        "Actor mean is `ref + delta_max * tanh(mlp(...))`, which hard-bounds outputs and "
        "prevents the TD3-style deadly-triad blowup (unbounded mu -> Q chases out-of-distribution "
        "actions). 1.0 lets the gripper fully flip from VLA and arm joints deviate up to ~57deg/step.",
    )
    parser.add_argument("--critic_hidden_dim", type=int, default=256)
    parser.add_argument("--critic_num_layers", type=int, default=2)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--bc_reg_weight", type=float, default=5.0,
                        help="Weight on MSE(actor mean, VLA ref chunk) in actor loss.")
    parser.add_argument("--ref_action_dropout", type=float, default=0.2,
                        help="Approx. fraction of batch with ref actions zeroed during actor updates.")
    parser.add_argument("--utd_ratio", type=int, default=5,
                        help="Update-to-data ratio: total (critic+actor) gradient updates performed "
                        "per environment step collected (paper: 5).")
    parser.add_argument("--critic_updates_per_actor", type=int, default=2,
                        help="Number of critic updates performed per actor update (paper: 2).")
    parser.add_argument(
        "--target_noise_std",
        type=float,
        default=0.2,
        help="TD3 target-policy smoothing: std of Gaussian noise added to the actor's next-state "
        "mean before the target critic evaluates it. 0 disables smoothing. Default 0.2 = 20%% of "
        "the default delta_max.",
    )
    parser.add_argument(
        "--target_noise_clip",
        type=float,
        default=0.5,
        help="TD3 target-policy smoothing: per-element clip on the smoothing noise. 0 disables clipping. "
        "Default 0.5 = 50%% of the default delta_max.",
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="Max-norm clip applied to actor and critic gradients before optimizer.step. 0 disables. "
        "Recommended 1.0 to prevent single-batch blowups under TD3-style off-policy learning.",
    )
    parser.add_argument("--async_learning", action=argparse.BooleanOptionalAction, default=True,
                        help="Run the learner in a background thread (paper: async rollouts + learning). "
                        "Disable with --no-async_learning to fall back to synchronous, per-episode updates "
                        "(still respects UTD=5 per env step but blocks rollouts while updating).")
    parser.add_argument("--learner_min_buffer_size", type=int, default=0,
                        help="Minimum replay-buffer size before the async learner starts sampling. "
                        "0 = use batch_size.")
    parser.add_argument("--buffer_capacity", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--replay_only_successful",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set (default), during warm-up only add transitions to the replay buffer when "
        "the episode finishes with task success (env.task_completed()). Post-warmup episodes "
        "are always added regardless. Disable with --no-replay_only_successful to keep "
        "failed/timeout warm-up trajectories too (legacy behavior).",
    )
    parser.add_argument(
        "--end_on_tape_drop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="End the episode early when shaped reward detects the tape fell after lift "
        "(see ``tape_drop_episode_should_end``). lift_handover: only fires before handover "
        "(phase_max<3); after arm0 releases to arm1, place-phase lift jitter no longer truncates "
        "the episode. sticky/dense_sticky: debounced after handover so brief REACH flicker "
        "does not end placement. Requires --shaped_reward_mode != off.",
    )
    parser.add_argument(
        "--end_actor_after_handover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Once the tape is handed to arm1 (phase_max >= PHASE_GRASP1=3), force the frozen "
        "VLA to drive the remainder of the episode regardless of phase_mode / "
        "actor_task_phases. Requires --shaped_reward_mode != off; no-op otherwise.",
    )
    parser.add_argument("--warmup_episodes", type=int, default=20)
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--max_episode_steps", type=int, default=1800)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)

    # RL token model config (must match Stage 1).
    parser.add_argument("--encoder_layers", type=int, default=4)
    parser.add_argument("--encoder_heads", type=int, default=8)
    parser.add_argument(
        "--encoder_ff_dim",
        type=int,
        default=_RL_TOKEN_CFG_DEFAULTS.encoder_ff_dim,
        help="Encoder FFN hidden dim (nn.TransformerEncoderLayer dim_feedforward). "
        "Must match Stage 1: checkpoint linear1.weight is [ff_dim, d_model] "
        "(default 8192; use 4096 only if that Stage 1 run used dim_feedforward=4096).",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state_dim", type=int, default=16,
                        help="Proprioceptive state dimension. Set to 14 for the MuJoCo bimanual env.")
    parser.add_argument("--wandb_enabled", action="store_true")
    parser.add_argument("--project_name", type=str, default="openpi_rlt")
    parser.add_argument("--vla_only", action="store_true",
                        help="Bypass actor entirely: always execute frozen VLA reference actions. "
                             "No RL updates are performed. Use to verify VLA outputs in the sim env.")
    parser.add_argument("--log_video_interval", type=int, default=1,
                        help="Log rollout video + action comparison every N episodes (0 = disable).")
    parser.add_argument("--video_fps", type=int, default=10,
                        help="Frames per second for logged rollout videos.")

    # Rollout smoothing (same family as groundTruthEval temporal ensembling / EMA overlap).
    parser.add_argument(
        "--action_smoothing",
        type=str,
        default="none",
        choices=["none", "temporal_ensembling", "ema_overlap"],
        help="none: infer every rl_chunk_length, no overlap smoothing (default). "
        "temporal_ensembling / ema_overlap: re-infer every --inference_frequency steps.",
    )
    parser.add_argument(
        "--inference_frequency",
        type=int,
        default=-1,
        help="When --action_smoothing is not none: steps between VLA+actor calls; "
        "must be <= rl_chunk_length. Default: min(20, rl_chunk_length). Ignored when smoothing is none.",
    )
    parser.add_argument(
        "--te_k",
        type=float,
        default=0.1,
        help="Temporal ensembling decay coefficient (larger = slower decay of older chunks).",
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.75,
        help="Weight on newest chunk in overlap EMA (ema_overlap mode).",
    )

    # Phased control (VLA -> RLT actor -> VLA).
    parser.add_argument(
        "--phase_mode",
        type=str,
        default="off",
        choices=["off", "chunk", "task_phase"],
        help="off: actor drives the whole post-warmup episode (current default). "
        "chunk: VLA drives [0, x) and [y, max_steps); actor drives [x, y) post-warmup "
        "(time-based, --vla_phase_end_step / --rl_phase_end_step). "
        "task_phase: actor drives chunks where the shaped-reward phase_max is in "
        "--actor_task_phases; VLA otherwise. Requires --shaped_reward_mode != off.",
    )
    parser.add_argument(
        "--vla_phase_end_step",
        type=int,
        default=0,
        help="x: env step at which control switches VLA -> actor (chunk mode rounds up to the "
        "next chunk boundary). Default 0 (no leading VLA phase).",
    )
    parser.add_argument(
        "--rl_phase_end_step",
        type=int,
        default=-1,
        help="y: env step at which control switches actor -> VLA. -1 (default) means "
        "max_episode_steps (no trailing VLA phase).",
    )
    parser.add_argument(
        "--actor_task_phases",
        type=int,
        nargs="+",
        default=[1],
        help="task_phase mode: phase_max values for which the RLT actor drives the next chunk. "
        "Default [1] = drive once the tape is off the table (phase_max>=1 in lift_only mode). "
        "Use e.g. '--actor_task_phases 2 3' for the multi-phase 'sticky' reward when only the "
        "handover phases should be RLT-controlled.",
    )
    parser.add_argument(
        "--phase_replay_strategy",
        type=str,
        default="rl_window_only",
        choices=["rl_window_only", "all"],
        help="rl_window_only (default): only insert transitions in [x, y) into the replay buffer "
        "and force done=1.0 at the boundary so the critic does not bootstrap past it. "
        "all: insert every transition (outside the window executed action is VLA).",
    )
    parser.add_argument(
        "--phase_reset_smoothing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset the action-smoothing runtime (TE / EMA) at each phase boundary so VLA and "
        "actor chunks aren't blended together at the seam.",
    )

    # Environment selection.
    parser.add_argument("--env", type=str, default="robosuite",
                        choices=["robosuite", "mujoco"],
                        help="Which environment backend to use.")
    parser.add_argument(
        "--robosuite_controller_cfg",
        type=str,
        default="robosuite/environments/custom/configs/panda_joint_ctrl_slow.json",
        help="Robosuite composite controller JSON (must match high-accuracy eval; "
        "the non-_slow config uses much higher gains and will not track the same "
        "absolute joint actions the VLA was trained on).",
    )
    parser.add_argument(
        "--robosuite_image_size",
        type=int,
        default=None,
        help="If set, resize env RGB to this square size before the policy. "
        "Default None = full camera resolution (same as groundTruthEval; policy does resize_with_pad).",
    )
    parser.add_argument(
        "--robosuite_tape_offsets_json",
        type=str,
        default=None,
        help="Path to handover index JSON (yellow_x/y, duct_x/y per entry) — same file as "
        "groundTruthEval offsets_json_path. Strongly recommended: the built-in 64-combo grid "
        "can differ from the exact training/eval XY pairs.",
    )
    parser.add_argument(
        "--match_ground_truth_eval_rollout",
        action="store_true",
        help="Align rollout hyperparams with groundTruthEval / plot_gte_vs_rlt2_rollout_actions. "
        "Sets rl_chunk_length = VLA action horizon (full chunk per VLA call). "
        "Substyle from --match_ground_truth_eval_mode: temporal = TE + infer every 20 (default); "
        "none = no TE/EMA, re-infer every full horizon (same as plot --smoothing none). "
        "Actor/critic are built for this chunk length from scratch each run (no resume). "
        "Without this flag, defaults stay rl_chunk_length=10 and action_smoothing=none — "
        "only the first 10 of 50 VLA steps per call — so warm-up will not match GTE.",
    )
    parser.add_argument(
        "--match_ground_truth_eval_mode",
        type=str,
        default="temporal",
        choices=["temporal", "none"],
        help="Used with --match_ground_truth_eval_rollout: temporal = temporal_ensembling + "
        "inference_frequency=20 + te_k=0.1; none = action_smoothing none (infer every horizon).",
    )
    parser.add_argument(
        "--robosuite_contact_solref",
        type=float,
        nargs=2,
        metavar=("TIME", "DAMP"),
        default=None,
        help="Robosuite: set all geom_solref to [timeconst, dampratio]. "
        "Default when omitted: [0.01, 1.0] (groundTruthEval contact_solref).",
    )
    parser.add_argument(
        "--robosuite_contact_solimp",
        type=float,
        nargs=3,
        metavar=("DMIN", "DMAX", "WIDTH"),
        default=None,
        help="Robosuite: set geom_solimp[:, :3] to [dmin, dmax, width]. "
        "Default when omitted: [0.95, 0.99, 0.001] (groundTruthEval contact_solimp).",
    )
    parser.add_argument(
        "--robosuite_fixed_layout_index",
        type=int,
        default=None,
        help="If set with --robosuite_tape_offsets_json, always use this index into the combo list "
        "(same ordering as GTE's JSON) so episode 0 matches a chosen layout (e.g. 0 = first entry).",
    )
    parser.add_argument(
        "--gripper_action_log",
        type=str,
        default=None,
        help="Append executed gripper cmds (after hold+sanitize) per env step as CSV (dims 7 and 15).",
    )
    parser.add_argument(
        "--sim_states_dir",
        type=str,
        default=None,
        help=(
            "Directory of pre-captured sim state snapshots produced by "
            "misc_scripts/capture_sim_states.py (contains index.json, per-layout .npys or states.npz, "
            "and model.xml (required). When set, each episode reset restores the sim to the mid-demo state "
            "for the current tape layout. "
            "Use the same robosuite controller JSON as capture (--controller-cfg), "
            "and matching wrist-camera mode (default: wrist cams on; old captures "
            "with --no-wrist-cameras require --robosuite_no_wrist_cameras on Stage 2)."
        ),
    )
    parser.add_argument(
        "--robosuite_no_wrist_cameras",
        action="store_true",
        help=(
            "Build FrankaRobosuiteTapeHandover with agentview only (no wrist RGB). "
            "Enable this only if your sim_states were captured with capture_sim_states.py "
            "--no-wrist-cameras; otherwise leave unset so observations match the VLA."
        ),
    )
    parser.add_argument(
        "--vla_train_augmentation_at_infer",
        action="store_true",
        help="If set, keep RandomImageAugmentation + GaussianActionNoise in the policy pipeline "
        "(matches raw TrainConfig). Default off: strip them for sim (recommended).",
    )

    # --- PyRoki handover oracle (replaces RLT actor during the handover phase) ---
    parser.add_argument(
        "--handover_controller",
        type=str,
        default="rlt",
        choices=["rlt", "pyroki"],
        help="rlt (default): GaussianActor drives the handover phase. "
        "pyroki: blocking PyRoki IK macro (adapted from "
        "mike/dependencies/robosuite/test_scripts/handover_step.py) drives it instead. "
        "VLA still owns pickup/lift and placement. Implies --shaped_reward_mode "
        "lift_handover + --phase_mode task_phase + --actor_task_phases 1, and "
        "force-disables the async learner (no actor updates while the oracle drives).",
    )
    parser.add_argument(
        "--pyroki_x_shift",
        type=float,
        default=0.0,
        help="Handover geometry x-shift forwarded to PyrokiHandoverOracle "
        "(matches handover_step.py --x_shift).",
    )
    parser.add_argument(
        "--pyroki_y_shift",
        type=float,
        default=0.0,
        help="Handover geometry y-shift forwarded to PyrokiHandoverOracle "
        "(matches handover_step.py --y_shift).",
    )
    parser.add_argument(
        "--pyroki_angle_shift",
        type=float,
        default=0.0,
        help="Handover geometry angle-shift (rad about z) forwarded to "
        "PyrokiHandoverOracle (matches handover_step.py --angle_shift).",
    )
    parser.add_argument(
        "--pyroki_perturb_radius",
        type=float,
        default=0.0,
        help="Lateral waypoint perturbation radius (m) for the PyRoki macro. "
        "0.0 (default) = deterministic handover. Matches handover_step.py "
        "--perturb_radius.",
    )
    parser.add_argument(
        "--pyroki_skip_on_arm0_pick",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When True (default), if lift_handover detects arm0 was the picker, "
        "skip the PyRoki macro and fall back to the VLA path for that episode "
        "(the macro is hard-coded for arm1-led picks). Disable to force the "
        "macro and let it raise.",
    )
    # MuJoCo-specific args (ignored when --env robosuite).
    parser.add_argument("--task", type=str, default="tape_handover_random",
                        help="Task key from eye.mujoco.tasks.TASK_REGISTRY.")
    parser.add_argument("--mujoco_repo_path", type=str, default="/home/jeffreyaj/mujoco",
                        help="Path to the eye/mujoco repo root (added to sys.path).")
    parser.add_argument("--reward_mode", type=str, default="stage_shaped",
                        choices=["sparse", "stage_shaped"],
                        help="Reward function for the MuJoCo env.")
    parser.add_argument("--n_substeps", type=int, default=20,
                        help="Physics substeps per agent step in the MuJoCo env.")
    parser.add_argument("--vla_prompt", type=str,
                        default="Pick up the yellow tape, hand it over, and place it on the gray tape",
                        help="Language prompt passed to the frozen VLA. Should match the prompt "
                             "used during VLA fine-tuning (prompt_from_task in training config).")

    # --- Shaped reward (Robosuite tape-handover only) ---
    parser.add_argument("--shaped_reward_mode", type=str, default="off",
                        choices=["off", "lift_only", "lift_handover", "sticky", "dense_sticky"],
                        help="off: legacy sparse 1.0-on-success reward (default). "
                             "lift_only: single observable — phase=1 once tape z > table+lift_thresh; "
                             "milestone on first lift + sparse success terminal. Recommended pair "
                             "with --phase_mode task_phase --actor_task_phases 1. "
                             "lift_handover: lift + whichever arm first grasped while lifted "
                             "opens (phase 3); supports robot0- or robot1-led picks. Pair with "
                             "--phase_mode task_phase --actor_task_phases 1 to let the actor drive "
                             "lift→release and the VLA drive the place. "
                             "sticky: 5-phase taxonomy, pure step-function progress (no within-phase shaping). "
                             "dense_sticky: sticky + Ng-1999 potential-difference shaping inside each phase. "
                             "See openpi.rlt.shaped_reward.")
    parser.add_argument("--shaped_milestone_bonus", type=float, default=0.5,
                        help="Bonus emitted once when phase_max advances. Multiplied by the number "
                             "of phases skipped on a given step.")
    parser.add_argument("--shaped_success_bonus", type=float, default=1.0,
                        help="Sparse terminal bonus added on the success step.")
    parser.add_argument("--shaped_lift_thresh", type=float, default=None,
                        help="Yellow tape z above table top required to count as 'lifted'. "
                             "Defaults to ShapedRewardConfig.lift_thresh (~0.02 m).")

    return parser.parse_args()


def build_shaped_reward_cfg(args, *, discount: float | None = None) -> "ShapedRewardConfig | None":
    """Construct a ShapedRewardConfig from CLI args, or None when shaping is off.

    ``discount`` should be the RL discount (``Stage2Config.discount``) so the
    Ng-1999 potential-difference shaping (``r = γ Φ' − Φ``) is theoretically
    optimality-preserving. When ``None`` the dataclass default (0.99) is kept.

    Imported lazily so the MuJoCo path doesn't pull in robosuite-only modules.
    """
    if args.shaped_reward_mode == "off":
        return None
    from openpi.rlt.shaped_reward import ShapedRewardConfig

    kwargs = {
        "mode": args.shaped_reward_mode,
        "milestone_bonus": args.shaped_milestone_bonus,
        "success_bonus": args.shaped_success_bonus,
    }
    if discount is not None:
        kwargs["potential_discount"] = float(discount)
    if args.shaped_lift_thresh is not None:
        kwargs["lift_thresh"] = float(args.shaped_lift_thresh)
    return ShapedRewardConfig(**kwargs)


def build_tsh_pi_obs(
    obs_dict: dict[str, Any],
    action_horizon: int,
    env_action_dim: int,
) -> dict[str, Any]:
    """Build the raw observation dict expected by Pi05PolicyWrapper / TSHInputs.

    Matches ``groundTruthEval.Pi05PolicyWrapper.infer_batch`` (uint8 HWC images,
    raw proprio, placeholder actions for padding).
    """
    return {
        "state": np.asarray(obs_dict["state"], dtype=np.float32),
        "exo_image": obs_dict["exo_image"],
        "wrist_left_image": obs_dict["wrist_left_image"],
        "wrist_right_image": obs_dict["wrist_right_image"],
        "actions": np.zeros((action_horizon, env_action_dim), dtype=np.float32),
    }


def obs_dict_to_vla_observation(
    trained_policy: Policy,
    obs_dict: dict[str, Any],
    device: torch.device,
    action_horizon: int,
    env_action_dim: int,
) -> Observation:
    """Apply the same input transforms as ``Policy.infer`` (TSHInputs, Normalize, Resize, tokenize, …).

    Pi0.5 tokenization discretizes **normalized** state in [-1, 1]; building
    ``Observation`` manually with raw sim state was incorrect (see
    ``groundTruthEval`` + ``create_trained_policy`` pipeline).
    """
    pi_obs = build_tsh_pi_obs(obs_dict, action_horizon, env_action_dim)
    inputs = jax.tree.map(lambda x: x, pi_obs)
    inputs = trained_policy._input_transform(inputs)  # noqa: SLF001
    inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.asarray(x)).to(device)[None, ...],
        inputs,
    )
    return Observation.from_dict(inputs)


def infer_vla_actions_numpy(
    trained_policy: Policy,
    obs_dict: dict[str, Any],
    action_horizon: int,
    env_action_dim: int,
) -> np.ndarray:
    """Run full policy inference: same as ``groundTruthEval`` (includes Unnormalize)."""
    pi_obs = build_tsh_pi_obs(obs_dict, action_horizon, env_action_dim)
    out = trained_policy.infer(pi_obs)
    return np.asarray(out["actions"], dtype=np.float32)


def _phase_use_actor(
    t: int,
    *,
    phase_mode: str,
    x: int,
    y: int,
    is_warmup: bool,
    vla_only: bool,
    phase_max: int = 0,
    actor_task_phases: tuple[int, ...] = (),
    end_actor_after_handover: bool = False,
) -> bool:
    """Decide whether the actor (vs frozen VLA) should drive env step ``t``.

    Warm-up and ``--vla_only`` always force VLA. ``end_actor_after_handover``
    (Stage2Config flag) hands control back to the VLA once
    ``phase_max >= PHASE_GRASP1`` (arm1 has the tape) regardless of
    ``phase_mode`` / ``actor_task_phases``. Otherwise:

    - ``phase_mode == "off"``        : actor drives the whole post-warmup episode.
    - ``phase_mode == "chunk"``      : actor drives steps in ``[x, y)`` (time-based).
    - ``phase_mode == "task_phase"`` : actor drives whenever ``phase_max`` is in
      ``actor_task_phases`` (state-based; ``phase_max`` from the shaped-reward
      tracker, including ``lift_only`` mode).
    """
    if vla_only or is_warmup:
        return False
    if end_actor_after_handover and int(phase_max) >= PHASE_GRASP1:
        return False
    if phase_mode == "off":
        return True
    if phase_mode == "task_phase":
        return int(phase_max) in set(actor_task_phases)
    return x <= t < y


def _resolve_phase_window(
    *,
    phase_mode: str,
    vla_phase_end_step: int,
    rl_phase_end_step: int,
    max_episode_steps: int,
    rl_chunk_length: int,
    vla_only: bool,
) -> tuple[int, int]:
    """Validate phase args and return ``(x, y)`` clamped to ``[0, max_episode_steps]``.

    Raises ``ValueError`` on invalid configurations and emits warnings for the soft cases (e.g. the
    RL window being shorter than ``rl_chunk_length`` so no chunk can be inserted into the replay
    buffer).
    """
    if phase_mode == "off":
        return 0, max_episode_steps
    if phase_mode == "task_phase":
        # Window is determined dynamically per-episode from phase_max — no static
        # x/y to validate. Return the trivial full range so any code that still
        # reads (phase_x, phase_y) (e.g. legacy logging) sees a sane default.
        if vla_only:
            logger.warning(
                "phase_mode='task_phase' combined with --vla_only is a no-op; --vla_only "
                "forces VLA execution.",
            )
        return 0, max_episode_steps
    x = int(vla_phase_end_step)
    y = max_episode_steps if int(rl_phase_end_step) < 0 else int(rl_phase_end_step)
    if not (0 <= x <= y <= max_episode_steps):
        raise ValueError(
            f"phase window must satisfy 0 <= x <= y <= max_episode_steps, "
            f"got x={x}, y={y}, max_episode_steps={max_episode_steps}."
        )
    if vla_only:
        logger.warning(
            "phase_mode=%r combined with --vla_only is a no-op; --vla_only forces VLA execution.",
            phase_mode,
        )
    if (y - x) < rl_chunk_length:
        logger.warning(
            "phase window [%d, %d) is shorter than rl_chunk_length=%d; the replay buffer will not "
            "receive any transitions from this run (RL actor will never train).",
            x, y, rl_chunk_length,
        )
    return x, y


def compute_td_target(
    batch: dict[str, torch.Tensor],
    actor: GaussianActor,
    target_critic: TwinQCritic,
    discount: float,
    rl_chunk_length: int,
    *,
    target_noise_std: float = 0.0,
    target_noise_clip: float = 0.0,
) -> torch.Tensor:
    """Compute the TD target Q-value (Eq. 3) with optional TD3 target-policy smoothing.

    Q_hat = sum_{t'=1}^C gamma^{t'-1} r_{t'} + gamma^C * min(Q1', Q2')(x', a' + clip(eps))

    The clipped Gaussian noise on the target action is the TD3 trick from
    "Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto
    et al., 2018): smoothing the target Q surface around the actor's deterministic
    mean prevents the actor from exploiting a single sharp Q peak. Disable by
    leaving ``target_noise_std=0``.
    """
    rewards = batch["reward"]  # [B, C]
    next_z_rl = batch["next_z_rl"]  # [B, z_rl_dim]
    next_state = batch["next_state"]  # [B, state_dim]
    next_ref = batch["ref_action"]  # Use same ref for next (approximation).
    dones = batch["done"]  # [B]

    # Discounted sum of chunk rewards.
    C = rl_chunk_length
    gammas = discount ** torch.arange(C, device=rewards.device, dtype=torch.float32)
    discounted_rewards = (rewards * gammas.unsqueeze(0)).sum(dim=1)  # [B]

    # Next-state value via target critic (bootstrap with deterministic policy mean; less target noise).
    with torch.no_grad():
        next_ref_flat = next_ref.reshape(next_ref.shape[0], -1)
        next_mean = actor.forward_mean(next_z_rl, next_state, next_ref_flat)
        if target_noise_std > 0.0:
            noise = torch.randn_like(next_mean) * float(target_noise_std)
            if target_noise_clip > 0.0:
                noise = torch.clamp(
                    noise, -float(target_noise_clip), float(target_noise_clip),
                )
            next_mean = next_mean + noise
        next_q = target_critic.q_min(next_z_rl, next_state, next_mean)

    target = discounted_rewards + (discount ** C) * next_q * (1.0 - dones)
    return target


def update_critic(
    batch: dict[str, torch.Tensor],
    critic: TwinQCritic,
    actor: GaussianActor,
    target_critic: TwinQCritic,
    critic_optimizer: torch.optim.Optimizer,
    discount: float,
    rl_chunk_length: int,
    *,
    target_noise_std: float = 0.0,
    target_noise_clip: float = 0.0,
    grad_clip_norm: float = 0.0,
) -> float:
    """One critic update step. Returns critic loss."""
    target = compute_td_target(
        batch, actor, target_critic, discount, rl_chunk_length,
        target_noise_std=target_noise_std,
        target_noise_clip=target_noise_clip,
    )

    z_rl = batch["z_rl"]
    state = batch["state"]
    actions = batch["action"].reshape(z_rl.shape[0], -1)  # Flatten chunk.

    q1, q2 = critic(z_rl, state, actions)
    critic_loss = ((q1 - target.detach()) ** 2 + (q2 - target.detach()) ** 2).mean()

    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    if grad_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=float(grad_clip_norm))
    critic_optimizer.step()

    return critic_loss.item()


def update_actor(
    batch: dict[str, torch.Tensor],
    actor: GaussianActor,
    critic: TwinQCritic,
    actor_optimizer: torch.optim.Optimizer,
    bc_reg_weight: float,
    ref_action_dropout: float,
    *,
    grad_clip_norm: float = 0.0,
) -> tuple[float, float]:
    """One actor update step. Returns (actor_loss, bc_loss).

    Uses the Gaussian **mean** for both Q and BC terms so gradients anchor μ to the
    VLA reference instead of chasing noisy samples (reduces jitter vs Eq. 5 with samples).

    Reference-action dropout zeros the ref *fed to the MLP* for a fraction of
    the batch (so the actor learns to act on (z_rl, s_p) alone) but keeps the
    residual base — and the BC target — anchored to the unmasked VLA reference.
    """
    z_rl = batch["z_rl"]
    state = batch["state"]
    ref_actions = batch["ref_action"].reshape(z_rl.shape[0], -1)  # [B, C*d]

    # Reference action dropout: zero out ref for a random fraction of the batch.
    B = z_rl.shape[0]
    dropout_mask = torch.rand(B, 1, device=z_rl.device) > ref_action_dropout
    ref_actions_masked = ref_actions * dropout_mask.float()

    # Pass unmasked ref as the residual base so the bound and BC target stay sane.
    action_mean = actor.forward_mean(
        z_rl, state, ref_actions_masked, ref_for_residual=ref_actions,
    )

    # Actor loss: -Q(x, μ) + beta * ||μ - a_tilde||^2
    q_value = critic.q_min(z_rl, state, action_mean)
    bc_loss = ((action_mean - ref_actions) ** 2).mean()
    actor_loss = -q_value.mean() + bc_reg_weight * bc_loss

    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    if grad_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=float(grad_clip_norm))
    actor_optimizer.step()

    return actor_loss.item(), bc_loss.item()


def evaluate(
    env: RLEnvironment,
    trained_policy: Policy,
    extractor: VLAEmbeddingExtractor,
    encoder: RLTokenEncoder,
    actor: GaussianActor,
    num_episodes: int,
    cfg: Stage2Config,
    device: torch.device,
    vla_action_horizon: int,
    *,
    vla_only: bool = False,
    actor_lock: threading.Lock | None = None,
    phase_x: int = 0,
    phase_y: int | None = None,
    handover_controller: str = "rlt",
    pyroki_oracle_kwargs: dict[str, Any] | None = None,
    pyroki_skip_on_arm0_pick: bool = True,
) -> dict[str, float]:
    """Evaluate with the same rollout cadence / smoothing as training.

    When ``vla_only`` is True, execute the frozen VLA chunk (same as the training loop in
    warmup / ``--vla_only`` mode). Otherwise use the actor mean. Previously evaluate() always
    routed through the actor, so periodic eval under ``--vla_only`` did not measure the VLA.

    When ``cfg.phase_mode != "off"``, control is gated by ``[phase_x, phase_y)`` exactly as in
    the training rollout loop, including resetting the action smoother at each phase boundary
    (when ``cfg.phase_reset_smoothing``).

    When ``handover_controller == "pyroki"``, the actor branch is replaced by a single
    blocking PyRoki macro per episode (same logic as the training rollout). On
    HandoverPickerMismatchError the macro is skipped and control falls through to the
    actor (or VLA under ``vla_only``).
    """
    actor.eval()
    successes = 0
    total_rewards = 0.0
    total_steps = 0
    infer_every = cfg.infer_every()
    C, d = cfg.rl_chunk_length, cfg.action_dim
    if phase_y is None:
        phase_y = cfg.max_episode_steps

    for _ in range(num_episodes):
        obs_dict = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        smooth = ActionSmoothingRuntime.start_episode(
            cfg.action_smoothing,
            te_k=cfg.te_k,
            ema_alpha=cfg.ema_alpha,
        )
        action_chunk = np.zeros((C, d), dtype=np.float32)
        prev_use_actor: bool | None = None
        current_phase_max = 0
        tape_drop_streak = 0
        pyroki_macro_attempted = False

        while not done and ep_steps < cfg.max_episode_steps:
            should_run_pyroki_macro = (
                handover_controller == "pyroki"
                and not vla_only
                and not pyroki_macro_attempted
                and _phase_use_actor(
                    ep_steps,
                    phase_mode=cfg.phase_mode,
                    x=phase_x,
                    y=phase_y,
                    is_warmup=False,
                    vla_only=False,
                    phase_max=current_phase_max,
                    actor_task_phases=cfg.actor_task_phases,
                    end_actor_after_handover=cfg.end_actor_after_handover,
                )
            )
            if should_run_pyroki_macro:
                from openpi.rlt.pyroki_handover import HandoverPickerMismatchError

                pyroki_macro_attempted = True
                try:
                    macro_result = env.run_pyroki_handover_macro(
                        oracle_kwargs=pyroki_oracle_kwargs or {},
                        check_picker_arm=pyroki_skip_on_arm0_pick,
                    )
                except HandoverPickerMismatchError as exc:
                    logger.warning("eval: PyRoki macro skipped — %s", exc)
                    macro_result = None
                if macro_result is not None:
                    obs_dict = macro_result["obs"]
                    ep_reward += float(macro_result["reward"])
                    ep_steps += int(macro_result["inner_steps"])
                    current_phase_max = max(
                        current_phase_max,
                        int(macro_result["info"].get("phase_max", 0)),
                    )
                    if macro_result["done"]:
                        done = True
                        continue
                    smooth = ActionSmoothingRuntime.start_episode(
                        cfg.action_smoothing,
                        te_k=cfg.te_k,
                        ema_alpha=cfg.ema_alpha,
                    )
                    prev_use_actor = True
                    remainder = ep_steps % infer_every
                    if remainder != 0:
                        ep_steps += (infer_every - remainder)
                    continue

            if ep_steps % infer_every == 0:
                use_actor_now = _phase_use_actor(
                    ep_steps,
                    phase_mode=cfg.phase_mode,
                    x=phase_x,
                    y=phase_y,
                    is_warmup=False,
                    vla_only=vla_only,
                    phase_max=current_phase_max,
                    actor_task_phases=cfg.actor_task_phases,
                    end_actor_after_handover=cfg.end_actor_after_handover,
                )
                if (
                    cfg.phase_mode != "off"
                    and cfg.phase_reset_smoothing
                    and prev_use_actor is not None
                    and prev_use_actor != use_actor_now
                ):
                    smooth = ActionSmoothingRuntime.start_episode(
                        cfg.action_smoothing,
                        te_k=cfg.te_k,
                        ema_alpha=cfg.ema_alpha,
                    )

                ref_actions_np = infer_vla_actions_numpy(
                    trained_policy, obs_dict, vla_action_horizon, d,
                )
                if not use_actor_now:
                    action_chunk = ref_actions_np[:C, :d].copy()
                else:
                    observation = obs_dict_to_vla_observation(
                        trained_policy, obs_dict, device, vla_action_horizon, d,
                    )
                    with torch.no_grad():
                        z_rl = extractor.extract_rl_token(observation, encoder)
                        ref_chunk = torch.from_numpy(
                            ref_actions_np[:C, :d],
                        ).float().unsqueeze(0).to(device)
                        ref_flat = ref_chunk.reshape(1, -1)
                        state_t = torch.from_numpy(obs_dict["state"]).float().unsqueeze(0).to(device)
                        if actor_lock is not None:
                            with actor_lock:
                                action_mean = actor.forward_mean(z_rl, state_t, ref_flat)
                        else:
                            action_mean = actor.forward_mean(z_rl, state_t, ref_flat)
                    action_chunk = action_mean[0].cpu().numpy().reshape(C, d)
                smooth.on_new_chunk(action_chunk, C)
                prev_use_actor = use_actor_now

            step_in_cycle = ep_steps % infer_every
            a_exec = smooth.next_executable_action(action_chunk, step_in_cycle)
            obs_dict, reward, done, info = env.step(a_exec)
            ep_reward += reward
            ep_steps += 1
            current_phase_max = max(current_phase_max, int(info.get("phase_max", 0)))
            tape_drop_term, tape_drop_streak = tape_drop_episode_should_end(
                shaped_reward_mode=cfg.shaped_reward_mode,
                end_on_tape_drop=cfg.end_on_tape_drop,
                phase_now=int(info.get("phase", 0)),
                phase_max_seen=current_phase_max,
                streak=tape_drop_streak,
            )
            if tape_drop_term:
                done = True

        if env.task_completed():
            successes += 1
        total_rewards += ep_reward
        total_steps += ep_steps

    actor.train()
    return {
        "eval/success_rate": successes / max(num_episodes, 1),
        "eval/mean_reward": total_rewards / max(num_episodes, 1),
        "eval/mean_steps": total_steps / max(num_episodes, 1),
    }


def save_stage2_checkpoint(
    actor: GaussianActor,
    critic: TwinQCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    episode: int,
    save_dir: pathlib.Path,
) -> None:
    """Save actor-critic checkpoint."""
    ckpt_dir = save_dir / str(episode)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    safetensors.torch.save_model(actor, str(ckpt_dir / "actor.safetensors"))
    safetensors.torch.save_model(critic, str(ckpt_dir / "critic.safetensors"))
    torch.save(
        {
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "episode": episode,
        },
        str(ckpt_dir / "training_state.pt"),
    )
    logger.info(f"Saved Stage 2 checkpoint at episode {episode}")


def _annotate_rollout_exo_frame(
    rgb_u8: np.ndarray,
    *,
    r_tot: float,
    r_step: float,
    phase_max: int,
    use_actor: bool = False,
    ep_steps: int = 0,
    phase: int | None = None,
    g0_span: float | None = None,
    g1_span: float | None = None,
    yellow_lift_height: float | None = None,
    triggered_by: str | None = None,
) -> np.ndarray:
    """Return a copy of the exo-camera RGB frame with reward/phase/controller overlay.

    The controller line is colour-coded: green = RLT actor driving, yellow = VLA driving.
    This makes it easy to verify that phase-gated control (task_phase mode) switches at
    the right moment — e.g. that RLT only takes over once phase_max transitions from 0→1
    in lift_only mode.

    Optional sensor diagnostics (``g0_span``, ``g1_span``, ``yellow_lift_height``) are
    drawn when provided; they're the raw signals the shaped-reward phase detector
    consumes, so when a phase transition happens "for no reason" you can read off
    exactly what the gripper / tape were doing on that frame. ``triggered_by``
    annotates whether this frame was sampled by the regular cadence, by a
    phase-max change, or by episode termination — useful when debugging short-lived
    phases that would otherwise be invisible at the regular sampling interval.
    """
    img = np.asarray(rgb_u8, dtype=np.uint8).copy()
    if img.ndim != 3 or img.shape[2] != 3:
        return img

    try:
        import cv2
    except ImportError:
        logger.warning(
            "opencv not available — rollout video frames will not show reward overlay",
        )
        return img

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(float(bgr.shape[1]) / 900.0 * 0.7, 1.15))
    thickness_fg = max(1, int(round(scale * 2)))
    thickness_bg = thickness_fg + 2
    y = int(26 * scale + 8)
    x = 10

    # White lines: reward and phase info.
    phase_str = f"phase_max={phase_max}"
    if phase is not None and int(phase) != int(phase_max):
        phase_str += f" (phase={int(phase)})"
    white_lines = [
        f"r_tot={r_tot:.4f} r_step={r_step:.4f}",
        f"{phase_str}  step={ep_steps}",
    ]
    if g0_span is not None or g1_span is not None or yellow_lift_height is not None:
        sensor_parts: list[str] = []
        if g0_span is not None:
            sensor_parts.append(f"g0={g0_span:.3f}")
        if g1_span is not None:
            sensor_parts.append(f"g1={g1_span:.3f}")
        if yellow_lift_height is not None:
            sensor_parts.append(f"lift={yellow_lift_height:+.3f}")
        white_lines.append("  ".join(sensor_parts))
    for line in white_lines:
        cv2.putText(bgr, line, (x, y), font, scale, (0, 0, 0), thickness_bg, cv2.LINE_AA)
        cv2.putText(bgr, line, (x, y), font, scale, (255, 255, 255), thickness_fg, cv2.LINE_AA)
        y += int(28 * scale + 8)

    # Colour-coded controller line: green = RLT actor, yellow = VLA.
    ctrl_label = "[RLT ACTOR]" if use_actor else "[VLA]"
    if triggered_by is not None:
        ctrl_label = f"{ctrl_label}  ({triggered_by})"
    ctrl_color_bgr = (0, 200, 0) if use_actor else (0, 200, 255)  # green or yellow
    cv2.putText(bgr, ctrl_label, (x, y), font, scale, (0, 0, 0), thickness_bg, cv2.LINE_AA)
    cv2.putText(bgr, ctrl_label, (x, y), font, scale, ctrl_color_bgr, thickness_fg, cv2.LINE_AA)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _write_video_h264(frames: list[np.ndarray], path: str, fps: int) -> bool:
    """Write RGB frames to an H.264/yuv420p MP4 — the only format browsers reliably play.

    Tries imageio+ffmpeg first (best quality), then falls back to cv2.
    Returns True on success.
    """
    try:
        import imageio
        writer = imageio.get_writer(
            path, fps=fps, codec="libx264",
            output_params=["-pix_fmt", "yuv420p", "-crf", "23"],
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        return True
    except Exception:
        pass

    try:
        import cv2
        h, w = frames[0].shape[:2]
        # Ensure even dimensions (H.264 requirement).
        h, w = h - h % 2, w - w % 2
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for frame in frames:
            out.write(cv2.cvtColor(frame[:h, :w], cv2.COLOR_RGB2BGR))
        out.release()
        return out.isOpened()
    except Exception:
        return False


def log_rollout_wandb(
    frames: list[np.ndarray],
    rl_actions: np.ndarray,
    ref_actions: np.ndarray,
    episode: int,
    fps: int = 10,
    video_save_dir: pathlib.Path | None = None,
) -> None:
    """Log rollout video and per-dimension VLA-vs-RL action comparison to wandb.

    Args:
        frames: List of uint8 HWC RGB images (one per chunk + initial reset frame).
            The training loop burns in ``r_tot`` / ``r_step`` / ``phase_max``
            overlays (4 decimals on rewards).
        rl_actions: [T, action_dim] array of RL actor actions.
        ref_actions: [T, action_dim] array of VLA reference actions.
        episode: Current episode index (used as wandb step and in plot title).
        fps: Video frame rate.
        video_save_dir: If set, videos are also saved here for local viewing.
    """
    log_dict: dict = {}

    # --- Rollout video (exo camera) ---
    if frames:
        # Determine output path — prefer a persistent local file so it's viewable
        # even without wandb, and so we can control the encoding.
        if video_save_dir is not None:
            video_save_dir.mkdir(parents=True, exist_ok=True)
            video_path = str(video_save_dir / f"episode_{episode:05d}.mp4")
            tmp = None
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            video_path = tmp.name
            tmp.close()

        ok = _write_video_h264(frames, video_path, fps)
        if ok:
            logger.info(f"Rollout video saved: {video_path}")
            log_dict["rollout/video"] = wandb.Video(video_path)
        else:
            logger.warning("Failed to encode rollout video — skipping.")

    # --- Per-dimension action comparison: VLA ref vs RL actor ---
    if rl_actions.shape[0] > 0:
        T = rl_actions.shape[0]
        action_dim = rl_actions.shape[1]
        t = np.arange(T)

        ncols = 4
        nrows = (action_dim + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.2), squeeze=False)

        for dim in range(action_dim):
            ax = axes[dim // ncols][dim % ncols]
            ax.plot(t, ref_actions[:T, dim], label="VLA ref", color="steelblue", linewidth=1.2)
            ax.plot(t, rl_actions[:, dim], label="RL actor", color="tomato",
                    linewidth=1.2, linestyle="--")
            ax.set_title(f"dim {dim}", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.set_xlim(0, T - 1)
            if dim == 0:
                ax.legend(fontsize=6, loc="upper right")

        for idx in range(action_dim, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(f"VLA vs RL actions — episode {episode}", fontsize=10)
        fig.tight_layout()
        log_dict["rollout/action_comparison"] = wandb.Image(fig)
        plt.close(fig)

    if log_dict:
        wandb.log(log_dict, step=episode)


def _load_rl_token_encoder_state_dict(ckpt_path: str, device: torch.device) -> dict[str, Any]:
    """Load a state dict for `RLTokenEncoder` from Stage 1 RL token checkpoints.

    Supports ``.safetensors`` (``save_model(RLTokenModule, ...)``) and PyTorch ``.pt``
    (e.g. ``torch.save(module.state_dict(), ...)``). Strips an ``encoder.`` prefix
    when the file contains a full ``RLTokenModule`` checkpoint so weights match
    a standalone ``RLTokenEncoder``.
    """
    path = pathlib.Path(ckpt_path)
    if path.suffix.lower() == ".safetensors":
        raw: dict[str, Any] = safetensors.torch.load_file(str(path), device=str(device))
    else:
        try:
            ckpt_obj = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            ckpt_obj = torch.load(path, map_location=device)
        if not isinstance(ckpt_obj, dict):
            raise TypeError(f"Expected a dict checkpoint at {path}, got {type(ckpt_obj)}")
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            raw = ckpt_obj["state_dict"]
        elif "encoder" in ckpt_obj and isinstance(ckpt_obj["encoder"], dict):
            raw = ckpt_obj["encoder"]
        else:
            raw = ckpt_obj
    if any(k.startswith("encoder.") for k in raw):
        raw = {k[len("encoder.") :]: v for k, v in raw.items() if k.startswith("encoder.")}
    return raw


def _load_actor_critic_warmstart(
    actor: GaussianActor,
    critic: TwinQCritic,
    init_path: str,
    device: torch.device,
) -> None:
    """Load actor + critic weights from a Stage 2-style checkpoint dir.

    The directory must contain ``actor.safetensors`` and ``critic.safetensors`` (the
    layout produced by both ``save_stage2_checkpoint`` here and
    ``scripts/pretrain_rlt_actor_critic.py``). Architectures are validated by
    :meth:`torch.nn.Module.load_state_dict` with ``strict=True`` — a hidden-dim or
    chunk-length mismatch between the checkpoint and the current run will raise a
    clear error.
    """
    ckpt_dir = pathlib.Path(init_path)
    if ckpt_dir.is_file():
        ckpt_dir = ckpt_dir.parent
    actor_path = ckpt_dir / "actor.safetensors"
    critic_path = ckpt_dir / "critic.safetensors"
    if not actor_path.is_file() or not critic_path.is_file():
        raise FileNotFoundError(
            f"--init_actor_critic_path {init_path}: expected actor.safetensors and "
            f"critic.safetensors under {ckpt_dir}."
        )
    actor_state = safetensors.torch.load_file(str(actor_path), device=str(device))
    critic_state = safetensors.torch.load_file(str(critic_path), device=str(device))
    actor.load_state_dict(actor_state, strict=True)
    critic.load_state_dict(critic_state, strict=True)
    logger.info(
        "Warm-started actor+critic from %s (architectures validated strict=True; "
        "optimizer state is NOT restored).",
        ckpt_dir,
    )


def train(args: argparse.Namespace) -> None:
    init_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import openpi.training.config as _config

    if args.match_ground_truth_eval_rollout:
        tc = _config.get_config(args.vla_config_name)
        h = int(tc.model.action_horizon)
        args.rl_chunk_length = h
        if args.match_ground_truth_eval_mode == "temporal":
            args.action_smoothing = "temporal_ensembling"
            args.inference_frequency = 20
            args.te_k = 0.1
            logger.info(
                "match_ground_truth_eval_rollout: rl_chunk_length=%d (VLA horizon), "
                "temporal_ensembling, inference_frequency=20, te_k=0.1",
                h,
            )
        else:
            args.action_smoothing = "none"
            # infer_every() uses rl_chunk_length when smoothing is none; value unused.
            args.inference_frequency = 20
            logger.info(
                "match_ground_truth_eval_rollout: rl_chunk_length=%d (VLA horizon), "
                "action_smoothing=none (re-infer every full horizon, matches GTE TE-off path)",
                h,
            )

    # Same global contact parameters as groundTruthEval.EvalConfig main() defaults.
    if args.env == "robosuite":
        if args.robosuite_contact_solref is None:
            args.robosuite_contact_solref = [0.01, 1.0]
        if args.robosuite_contact_solimp is None:
            args.robosuite_contact_solimp = [0.95, 0.99, 0.001]

    if args.action_smoothing == "none":
        infer_freq = args.rl_chunk_length
    else:
        infer_freq = args.inference_frequency
        if infer_freq < 0:
            infer_freq = min(20, args.rl_chunk_length)
        if infer_freq < 1 or infer_freq > args.rl_chunk_length:
            raise ValueError(
                f"For action_smoothing={args.action_smoothing!r}, inference_frequency must be in "
                f"[1, rl_chunk_length={args.rl_chunk_length}], got {infer_freq}."
            )

    cfg = Stage2Config(
        rl_chunk_length=args.rl_chunk_length,
        action_dim=args.action_dim,
        stride=args.stride,
        action_smoothing=args.action_smoothing,
        inference_frequency=infer_freq,
        te_k=args.te_k,
        ema_alpha=args.ema_alpha,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_num_layers=args.actor_num_layers,
        actor_fixed_std=args.actor_fixed_std,
        actor_stochastic_rollout=args.actor_stochastic_rollout,
        actor_lr=args.actor_lr,
        actor_delta_max=args.actor_delta_max,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_num_layers=args.critic_num_layers,
        critic_lr=args.critic_lr,
        discount=args.discount,
        tau=args.tau,
        bc_reg_weight=args.bc_reg_weight,
        ref_action_dropout=args.ref_action_dropout,
        utd_ratio=args.utd_ratio,
        critic_updates_per_actor=args.critic_updates_per_actor,
        target_noise_std=args.target_noise_std,
        target_noise_clip=args.target_noise_clip,
        grad_clip_norm=args.grad_clip_norm,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        replay_only_successful_episodes=args.replay_only_successful,
        end_on_tape_drop=args.end_on_tape_drop,
        end_actor_after_handover=args.end_actor_after_handover,
        warmup_episodes=args.warmup_episodes,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_interval=args.save_interval,
        state_dim=args.state_dim,
        phase_mode=args.phase_mode,
        vla_phase_end_step=args.vla_phase_end_step,
        rl_phase_end_step=args.rl_phase_end_step,
        phase_replay_strategy=args.phase_replay_strategy,
        phase_reset_smoothing=args.phase_reset_smoothing,
        actor_task_phases=tuple(int(p) for p in args.actor_task_phases),
        shaped_reward_mode=args.shaped_reward_mode,
    )
    if cfg.phase_mode == "task_phase" and args.shaped_reward_mode == "off":
        raise ValueError(
            "--phase_mode task_phase requires --shaped_reward_mode != off so phase_max is "
            "computed (use 'lift_only' for the simple z-threshold gate)."
        )

    # PyRoki handover oracle prerequisites. The oracle replaces the RLT actor
    # during the handover window, so it needs (a) shaped reward in
    # `lift_handover` mode (so phase_max ticks 0 -> 1 on lift, then 1 -> 3 once
    # the macro completes), (b) `phase_mode=task_phase` with `actor_task_phases`
    # containing 1 (so the gate fires exactly during the handover), and (c) no
    # async learner (no actor updates make sense while a fixed oracle drives).
    if args.handover_controller == "pyroki":
        if args.shaped_reward_mode != "lift_handover":
            raise ValueError(
                "--handover_controller pyroki requires --shaped_reward_mode "
                "lift_handover so phase_max ticks 0 -> 1 -> 3 around the macro. "
                f"Got --shaped_reward_mode {args.shaped_reward_mode!r}."
            )
        if cfg.phase_mode != "task_phase":
            raise ValueError(
                "--handover_controller pyroki requires --phase_mode task_phase so "
                "the oracle is invoked exactly during the handover window. "
                f"Got --phase_mode {cfg.phase_mode!r}."
            )
        if 1 not in set(cfg.actor_task_phases):
            raise ValueError(
                "--handover_controller pyroki requires --actor_task_phases to include 1 "
                "(handover is gated on phase_max == 1 in lift_handover mode). "
                f"Got --actor_task_phases {list(cfg.actor_task_phases)}."
            )
        if args.async_learning:
            logger.warning(
                "--handover_controller pyroki: force-disabling async learner. "
                "The oracle drives the handover, so the actor receives no on-policy "
                "transitions and no off-policy updates would be meaningful.",
            )
            args.async_learning = False
        if args.vla_only:
            logger.warning(
                "--handover_controller pyroki combined with --vla_only is unusual: "
                "--vla_only forces VLA execution and the macro will never fire.",
            )
    # The drop / handover signals come from the shaped-reward phase tracker. If the
    # user disabled shaped reward, both flags are silent no-ops.
    if cfg.end_on_tape_drop and args.shaped_reward_mode == "off":
        logger.warning(
            "--end_on_tape_drop is set but --shaped_reward_mode=off. "
            "phase_max is always 0 in this mode, so the drop detector will never fire.",
        )
    # PHASE_GRASP1 (=3, "arm1 has the tape") is emitted by `sticky` /
    # `dense_sticky` (explicit arm1-grasp detector) and by `lift_handover`
    # (jumps 1 -> 3 the moment arm0 fully opens after lift; the physical
    # premise is that gravity would pull the tape down otherwise). `lift_only`
    # jumps 0 -> 1 -> 4 (success) and never passes through 2/3, so the
    # handover switch silently no-ops there.
    if cfg.end_actor_after_handover and args.shaped_reward_mode in ("off", "lift_only"):
        logger.warning(
            "--end_actor_after_handover is set but --shaped_reward_mode=%r does not emit "
            "PHASE_GRASP1 (=3). The handover->VLA switch will not fire. Use "
            "--shaped_reward_mode lift_handover (cheap) or sticky / dense_sticky "
            "(more granular) to enable it.",
            args.shaped_reward_mode,
        )
    phase_x, phase_y = _resolve_phase_window(
        phase_mode=cfg.phase_mode,
        vla_phase_end_step=cfg.vla_phase_end_step,
        rl_phase_end_step=cfg.rl_phase_end_step,
        max_episode_steps=cfg.max_episode_steps,
        rl_chunk_length=cfg.rl_chunk_length,
        vla_only=args.vla_only,
    )
    if cfg.phase_mode != "off":
        logger.info(
            "Phased control: mode=%s, VLA->actor at step %d, actor->VLA at step %d "
            "(replay=%s, reset_smoothing=%s).",
            cfg.phase_mode, phase_x, phase_y,
            cfg.phase_replay_strategy, cfg.phase_reset_smoothing,
        )
    logger.info(
        f"Rollout: action_smoothing={cfg.action_smoothing!r}, "
        f"infer_every={cfg.infer_every()} steps"
    )
    logger.info(
        "Updates: async_learning=%s, utd_ratio=%d (updates per env step), "
        "critic_updates_per_actor=%d, warmup_episodes=%d, "
        "replay_only_successful_warmup=%s, end_on_tape_drop=%s, "
        "end_actor_after_handover=%s",
        args.async_learning and not args.vla_only,
        cfg.utd_ratio,
        cfg.critic_updates_per_actor,
        cfg.warmup_episodes,
        cfg.replay_only_successful_episodes,
        cfg.end_on_tape_drop,
        cfg.end_actor_after_handover,
    )

    # --- Load frozen VLA (same path as groundTruthEval: policy transforms + weights) ---
    train_config = _config.get_config(args.vla_config_name)
    model_cfg = train_config.model
    if not isinstance(model_cfg, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype="bfloat16",
            action_dim=model_cfg.action_dim,
            action_horizon=model_cfg.action_horizon,
            max_token_len=model_cfg.max_token_len,
            pi05=getattr(model_cfg, "pi05", False),
        )
    train_config = dataclasses.replace(train_config, model=model_cfg)

    trained_policy = openpi_policy_config.create_trained_policy(
        train_config,
        args.vla_checkpoint_path,
        default_prompt=args.vla_prompt,
        pytorch_device=str(device),
        skip_train_data_input_transforms=not args.vla_train_augmentation_at_infer,
    )
    vla_model = trained_policy._model  # noqa: SLF001
    for p in vla_model.parameters():
        p.requires_grad_(False)
    vla_model.eval()
    logger.info(
        "Loaded frozen VLA via create_trained_policy (TSHInputs → Normalize → tokenize → …) "
        f"from {args.vla_checkpoint_path}"
    )

    vla_action_horizon = int(vla_model.config.action_horizon)
    extractor = VLAEmbeddingExtractor(vla_model, device)

    # --- Load frozen RL token encoder ---
    rl_token_config = RLTokenModelConfig(
        encoder_layers=args.encoder_layers,
        encoder_heads=args.encoder_heads,
        encoder_ff_dim=args.encoder_ff_dim,
    )
    encoder = RLTokenEncoder(rl_token_config).to(device)
    # Load only encoder weights from the Stage 1 checkpoint (.safetensors or .pt).
    enc_state = _load_rl_token_encoder_state_dict(args.rl_token_checkpoint_path, device)
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing:
        logger.warning(
            "RL token encoder: %d missing keys after load (showing up to 5): %s",
            len(missing),
            missing[:5],
        )
    if unexpected:
        logger.warning(
            "RL token encoder: %d unexpected keys after load (showing up to 5): %s",
            len(unexpected),
            unexpected[:5],
        )
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    logger.info(f"Loaded frozen RL token encoder from {args.rl_token_checkpoint_path}")

    # --- Build actor and critic ---
    action_chunk_dim = cfg.action_chunk_dim
    actor = GaussianActor(
        z_rl_dim=cfg.z_rl_dim,
        state_dim=cfg.state_dim,
        action_chunk_dim=action_chunk_dim,
        hidden_dim=cfg.actor_hidden_dim,
        num_layers=cfg.actor_num_layers,
        fixed_std=cfg.actor_fixed_std,
        delta_max=cfg.actor_delta_max,
    ).to(device)

    critic = TwinQCritic(
        z_rl_dim=cfg.z_rl_dim,
        state_dim=cfg.state_dim,
        action_chunk_dim=action_chunk_dim,
        hidden_dim=cfg.critic_hidden_dim,
        num_layers=cfg.critic_num_layers,
    ).to(device)

    if args.init_actor_critic_path:
        _load_actor_critic_warmstart(
            actor, critic, args.init_actor_critic_path, device,
        )

    # Target critic is a frozen deep copy of `critic`; build it AFTER any
    # warm-start load so the target mirrors the loaded weights.
    target_critic = create_target_critic(critic)

    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=cfg.actor_lr)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr)

    # Serializes concurrent reads (rollout inference) and writes (learner
    # optimizer.step) of actor parameters when async learning is enabled.
    actor_lock = threading.Lock()

    logger.info(f"Actor: {sum(p.numel() for p in actor.parameters()) / 1e3:.1f}K params")
    logger.info(f"Critic: {sum(p.numel() for p in critic.parameters()) / 1e3:.1f}K params")

    # --- Replay buffer ---
    replay_buffer = ReplayBuffer(
        capacity=cfg.buffer_capacity,
        chunk_length=cfg.rl_chunk_length,
        action_dim=cfg.action_dim,
        z_rl_dim=cfg.z_rl_dim,
        state_dim=cfg.state_dim,
    )

    # --- Environment ---
    if args.env == "mujoco":
        import sys
        # _stubs/ provides a minimal eye namespace (eye.camera constants +
        # eye.mujoco symlinked to the mujoco repo) so that imports resolve
        # without the full private eye package.
        sys.path.insert(0, os.path.join(args.mujoco_repo_path, "_stubs"))
        from eye.mujoco.rlt_env import MuJoCoRLTEnv
        from eye.mujoco.tasks import TASK_REGISTRY

        task_factory = TASK_REGISTRY[args.task]
        env = MuJoCoRLTEnv(
            task=task_factory(),
            reward_mode=args.reward_mode,
            n_substeps=args.n_substeps,
            max_episode_steps=cfg.max_episode_steps,
            seed=args.seed,
        )
        eval_env = MuJoCoRLTEnv(
            task=task_factory(),
            reward_mode=args.reward_mode,
            n_substeps=args.n_substeps,
            max_episode_steps=cfg.max_episode_steps,
            seed=args.seed + 1000,
        )
        logger.info(f"MuJoCo env: task={args.task}, reward_mode={args.reward_mode}, n_substeps={args.n_substeps}")
    else:
        from openpi.rlt.robosuite_env import RobosuiteRLTEnv

        tape_json = args.robosuite_tape_offsets_json
        contact_solref = args.robosuite_contact_solref
        contact_solimp = args.robosuite_contact_solimp
        layout_idx = args.robosuite_fixed_layout_index
        grip_log = args.gripper_action_log
        sim_states_dir = args.sim_states_dir
        # Train env uses the shaped reward (when enabled) so the actor sees
        # dense signal during exploration. Eval env keeps the legacy sparse
        # 0/1 reward so eval/{mean_reward, success_rate} stay comparable to
        # prior runs and to groundTruthEval.
        shaped_cfg = build_shaped_reward_cfg(args, discount=cfg.discount)
        if shaped_cfg is not None:
            logger.info(
                "Shaped reward enabled (mode=%s, milestone_bonus=%.3f, success_bonus=%.3f, "
                "potential_discount=%.3f). Eval env intentionally keeps sparse 0/1 reward "
                "for comparability.",
                shaped_cfg.mode, shaped_cfg.milestone_bonus, shaped_cfg.success_bonus,
                shaped_cfg.potential_discount,
            )
        env = RobosuiteRLTEnv(
            controller_cfg=args.robosuite_controller_cfg,
            image_size=args.robosuite_image_size,
            use_wrist_cameras=not args.robosuite_no_wrist_cameras,
            max_steps=cfg.max_episode_steps,
            seed=args.seed,
            tape_offsets_json=tape_json,
            contact_solref=contact_solref,
            contact_solimp=contact_solimp,
            tape_layout_index=layout_idx,
            gripper_action_log=grip_log,
            sim_states_dir=sim_states_dir,
            shaped_reward_cfg=shaped_cfg,
        )
        eval_env = RobosuiteRLTEnv(
            controller_cfg=args.robosuite_controller_cfg,
            image_size=args.robosuite_image_size,
            use_wrist_cameras=not args.robosuite_no_wrist_cameras,
            max_steps=cfg.max_episode_steps,
            seed=args.seed + 1000,
            tape_offsets_json=tape_json,
            contact_solref=contact_solref,
            contact_solimp=contact_solimp,
            tape_layout_index=layout_idx,
            gripper_action_log=None,
            sim_states_dir=sim_states_dir,
            shaped_reward_cfg=None,
        )
        if args.match_ground_truth_eval_rollout and tape_json is None:
            logger.warning(
                "match_ground_truth_eval_rollout: --robosuite_tape_offsets_json not set; "
                "RLT uses the built-in linspace grid, which can differ from groundTruthEval's "
                "handover_index.json layouts."
            )

    task_prompt = args.vla_prompt
    logger.info(f"VLA prompt (injected by policy if missing): '{task_prompt}'")

    # --- Checkpoint directory ---
    save_dir = pathlib.Path(args.checkpoint_base_dir) / "stage2" / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Wandb ---
    if args.wandb_enabled:
        wandb.init(name=args.exp_name, project=args.project_name, config=vars(args))
    else:
        wandb.init(mode="disabled")

    # --- Async learner (paper "Update" protocol) ---
    # When enabled, a background thread applies off-policy updates from the
    # replay buffer (Algorithm 1) at UTD=`utd_ratio` per env step, with
    # `critic_updates_per_actor` critic updates per actor update. The learner
    # is only started after warm-up and only when we actually need to train
    # (skipped under --vla_only).
    # Bind TD3 / grad-clip kwargs onto the update fns so the AsyncLearner can
    # call them with the existing signature it already understands. Same fns
    # (with the same baked-in kwargs) are used by the sync fallback below so
    # both paths get target-policy smoothing + grad clipping.
    update_critic_fn = functools.partial(
        update_critic,
        target_noise_std=cfg.target_noise_std,
        target_noise_clip=cfg.target_noise_clip,
        grad_clip_norm=cfg.grad_clip_norm,
    )
    update_actor_fn = functools.partial(
        update_actor,
        grad_clip_norm=cfg.grad_clip_norm,
    )
    logger.info(
        "Stabilizers: actor_delta_max=%.3f (residual+tanh bound), "
        "target_noise_std=%.3f, target_noise_clip=%.3f, grad_clip_norm=%.3f",
        cfg.actor_delta_max, cfg.target_noise_std, cfg.target_noise_clip, cfg.grad_clip_norm,
    )

    async_learner: AsyncLearner | None = None
    if args.async_learning and not args.vla_only:
        async_learner = AsyncLearner(
            replay_buffer=replay_buffer,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            actor_lock=actor_lock,
            update_critic_fn=update_critic_fn,
            update_actor_fn=update_actor_fn,
            device=device,
            batch_size=cfg.batch_size,
            discount=cfg.discount,
            rl_chunk_length=cfg.rl_chunk_length,
            bc_reg_weight=cfg.bc_reg_weight,
            ref_action_dropout=cfg.ref_action_dropout,
            tau=cfg.tau,
            utd_ratio=cfg.utd_ratio,
            critic_updates_per_actor=cfg.critic_updates_per_actor,
            min_buffer_size=args.learner_min_buffer_size or cfg.batch_size,
        )

    # --- Main training loop (Algorithm 1) ---
    total_env_steps = 0
    post_warmup_env_steps = 0  # Env steps counted toward UTD (paper: learning starts post-warmup).
    total_updates = 0
    log_video = args.log_video_interval > 0

    pbar = tqdm.tqdm(
        range(cfg.num_episodes),
        desc="Stage 2",
        unit="ep",
        dynamic_ncols=True,
    )

    infer_every = cfg.infer_every()

    for episode in pbar:
        is_warmup = args.vla_only or (episode < cfg.warmup_episodes)
        obs_dict = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        # Shaped-reward / phase-tracking accumulators. Always populated (the env
        # emits zeros when shaping is off), so logging code stays branch-free.
        ep_phase_max = 0
        ep_shaping_sum = 0.0
        ep_milestone_sum = 0.0

        # Collect per-step data for subsampled buffer insertion.
        ep_z_rls = []
        ep_states = []
        ep_actions = []
        ep_ref_actions = []
        ep_rewards = []
        ep_dones = []
        ep_frames: list[np.ndarray] = []  # exo camera frames for wandb video

        smooth = ActionSmoothingRuntime.start_episode(
            cfg.action_smoothing,
            te_k=cfg.te_k,
            ema_alpha=cfg.ema_alpha,
        )
        action_chunk = np.zeros((cfg.rl_chunk_length, cfg.action_dim), dtype=np.float32)
        z_rl_np = np.zeros((cfg.z_rl_dim,), dtype=np.float32)
        ref_chunk_np = np.zeros((cfg.rl_chunk_length, cfg.action_dim), dtype=np.float32)
        # Tracks the source of the previously-executed chunk (None at episode start) so we
        # can reset the smoothing runtime when control swaps VLA <-> actor at a phase boundary.
        prev_use_actor: bool | None = None
        ep_actor_steps = 0
        # Latest known phase_max for chunk-boundary gating (task_phase mode). Updated
        # from each env.step()'s info dict; stays 0 when shaped reward is off.
        current_phase_max = 0
        tape_drop_streak = 0
        ep_tape_drop_triggered = False

        # Per-step "did the actor drive this step?" flag, recorded so the replay
        # trim logic for task_phase mode knows which transitions were actor-controlled.
        ep_actor_drove: list[bool] = []

        # PyRoki handover oracle: at most one macro firing per episode. If the
        # macro raises HandoverPickerMismatchError (arm0-led pick) or
        # otherwise fails, this flag still gets set so we don't retry every
        # iteration; control falls through to the regular actor / VLA path.
        pyroki_macro_attempted = False
        pyroki_macro_succeeded = False

        # Track the last phase_max we drew a video frame for, so we can force
        # an additional capture on every transition. Without this, a brief
        # phase (e.g. lift_handover phase 1, which can last only a handful of
        # sim steps if the policy releases arm0 quickly after lift) is
        # systematically missed by the regular ``infer_every``-step sampler
        # and only the post-transition state ever appears in the video.
        last_video_phase_max = 0

        if log_video and "exo_image" in obs_dict:
            ep_frames.append(
                _annotate_rollout_exo_frame(
                    obs_dict["exo_image"],
                    r_tot=0.0,
                    r_step=0.0,
                    phase_max=0,
                    use_actor=False,
                    ep_steps=0,
                    triggered_by="reset",
                ),
            )

        while not done and ep_steps < cfg.max_episode_steps:
            # One video frame per VLA+actor cycle: captured *after* each chunk of
            # ``infer_every`` env.step calls so overlay matches cumulative reward.

            # --- PyRoki handover oracle (replaces the actor for the handover phase) ---
            # The macro consumes many inner sim steps in one logical call, so we
            # check this BEFORE the per-chunk inference block: if it fires we
            # short-circuit the rest of the iteration, advance ep_steps by the
            # number of inner sim steps consumed, and rely on _phase_use_actor
            # returning False on the next iteration (phase_max jumps 1 -> 3 post-
            # macro, so end_actor_after_handover hands control back to the VLA).
            should_run_pyroki_macro = (
                args.handover_controller == "pyroki"
                and not is_warmup
                and not args.vla_only
                and not pyroki_macro_attempted
                and _phase_use_actor(
                    ep_steps,
                    phase_mode=cfg.phase_mode,
                    x=phase_x,
                    y=phase_y,
                    is_warmup=False,
                    vla_only=False,
                    phase_max=current_phase_max,
                    actor_task_phases=cfg.actor_task_phases,
                    end_actor_after_handover=cfg.end_actor_after_handover,
                )
            )
            if should_run_pyroki_macro:
                from openpi.rlt.pyroki_handover import HandoverPickerMismatchError

                pyroki_macro_attempted = True
                try:
                    macro_result = env.run_pyroki_handover_macro(
                        oracle_kwargs=dict(
                            x_shift=args.pyroki_x_shift,
                            y_shift=args.pyroki_y_shift,
                            angle_shift=args.pyroki_angle_shift,
                            perturb_radius=args.pyroki_perturb_radius,
                            seed=args.seed + episode,
                        ),
                        check_picker_arm=args.pyroki_skip_on_arm0_pick,
                    )
                except HandoverPickerMismatchError as exc:
                    logger.warning(
                        "ep=%d step=%d: PyRoki macro skipped — %s",
                        episode, ep_steps, exc,
                    )
                    macro_result = None

                if macro_result is not None:
                    pyroki_macro_succeeded = True
                    obs_dict = macro_result["obs"]
                    macro_inner_steps = int(macro_result["inner_steps"])
                    macro_info = macro_result["info"]
                    macro_reward = float(macro_result["reward"])
                    macro_done = bool(macro_result["done"])

                    # Logging-only per-episode aggregates. Replay buffer is
                    # suppressed entirely for pyroki episodes (validated at
                    # startup), so we don't append to ep_z_rls / ep_actions /
                    # ep_ref_actions / ep_states (the macro doesn't produce
                    # 16D bimanual chunks anyway).
                    ep_rewards.append(macro_reward)
                    ep_dones.append(macro_done)
                    ep_actor_drove.append(True)
                    ep_reward += macro_reward
                    ep_shaping_sum += float(macro_info.get("shaping_reward", 0.0))
                    ep_milestone_sum += float(macro_info.get("milestone_reward", 0.0))
                    current_phase_max = max(
                        current_phase_max, int(macro_info.get("phase_max", 0)),
                    )
                    ep_phase_max = max(ep_phase_max, current_phase_max)
                    ep_actor_steps += 1  # one logical actor-driven "chunk"
                    ep_steps += macro_inner_steps
                    total_env_steps += macro_inner_steps

                    # Drop stale TE / EMA so the next VLA chunk starts clean
                    # instead of blending against pre-macro action_chunk values.
                    smooth = ActionSmoothingRuntime.start_episode(
                        cfg.action_smoothing,
                        te_k=cfg.te_k,
                        ema_alpha=cfg.ema_alpha,
                    )
                    prev_use_actor = True

                    if log_video and "exo_image" in obs_dict:
                        ep_frames.append(
                            _annotate_rollout_exo_frame(
                                obs_dict["exo_image"],
                                r_tot=float(ep_reward),
                                r_step=macro_reward,
                                phase_max=current_phase_max,
                                use_actor=True,
                                ep_steps=ep_steps,
                                phase=int(macro_info.get("phase", 0)),
                                g0_span=(
                                    float(macro_info.get("g0_span"))
                                    if "g0_span" in macro_info else None
                                ),
                                g1_span=(
                                    float(macro_info.get("g1_span"))
                                    if "g1_span" in macro_info else None
                                ),
                                yellow_lift_height=(
                                    float(macro_info.get("yellow_lift_height"))
                                    if "yellow_lift_height" in macro_info else None
                                ),
                                triggered_by="pyroki_macro",
                            ),
                        )
                        last_video_phase_max = current_phase_max

                    logger.info(
                        "ep=%d: PyRoki macro completed inner_steps=%d "
                        "reward=%.3f phase_max=%d done=%s skipped_reason=%s",
                        episode, macro_inner_steps, macro_reward,
                        current_phase_max, macro_done,
                        macro_info.get("oracle_skipped_reason"),
                    )

                    if macro_done:
                        done = True
                        continue

                    # Round ep_steps up to the next chunk boundary so the next
                    # iteration's `ep_steps % infer_every == 0` block fires
                    # cleanly and we don't try to execute a stale action_chunk
                    # at a non-zero step_in_cycle.
                    remainder = ep_steps % infer_every
                    if remainder != 0:
                        ep_steps += (infer_every - remainder)
                    continue

            if ep_steps % infer_every == 0:
                use_actor_now = _phase_use_actor(
                    ep_steps,
                    phase_mode=cfg.phase_mode,
                    x=phase_x,
                    y=phase_y,
                    is_warmup=is_warmup,
                    vla_only=args.vla_only,
                    phase_max=current_phase_max,
                    actor_task_phases=cfg.actor_task_phases,
                    end_actor_after_handover=cfg.end_actor_after_handover,
                )
                # Drop stale TE / EMA state at a phase boundary so VLA and actor chunks aren't
                # blended together at the seam.
                if (
                    cfg.phase_mode != "off"
                    and cfg.phase_reset_smoothing
                    and prev_use_actor is not None
                    and prev_use_actor != use_actor_now
                ):
                    smooth = ActionSmoothingRuntime.start_episode(
                        cfg.action_smoothing,
                        te_k=cfg.te_k,
                        ema_alpha=cfg.ema_alpha,
                    )

                observation = obs_dict_to_vla_observation(
                    trained_policy, obs_dict, device, vla_action_horizon, cfg.action_dim,
                )

                with torch.no_grad():
                    z_rl = extractor.extract_rl_token(observation, encoder)  # [1, 2048]
                    ref_actions_np = infer_vla_actions_numpy(
                        trained_policy, obs_dict, vla_action_horizon, cfg.action_dim,
                    )
                    ref_chunk = torch.from_numpy(
                        ref_actions_np[: cfg.rl_chunk_length, : cfg.action_dim],
                    ).float().to(device)

                state = obs_dict["state"]  # [16]

                ref_chunk_np = ref_chunk.cpu().numpy()

                if not use_actor_now:
                    action_chunk = ref_chunk_np
                else:
                    ref_flat = ref_chunk.reshape(1, -1)  # [1, C*d]
                    state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                    # actor_lock keeps inference consistent if the async learner
                    # is concurrently updating actor parameters.
                    with torch.no_grad(), actor_lock:
                        if cfg.actor_stochastic_rollout:
                            sampled, _ = actor(z_rl, state_t, ref_flat)
                            action_chunk = sampled[0].cpu().numpy().reshape(
                                cfg.rl_chunk_length, cfg.action_dim,
                            )
                        else:
                            mean_only = actor.forward_mean(z_rl, state_t, ref_flat)
                            action_chunk = mean_only[0].cpu().numpy().reshape(
                                cfg.rl_chunk_length, cfg.action_dim,
                            )
                    # Diagnostic: log actor activation and deviation from VLA so we can
                    # verify the actor is actually driving and producing distinct actions.
                    actor_vla_mad = float(np.mean(np.abs(action_chunk - ref_chunk_np)))
                    logger.info(
                        "ACTOR ACTIVE | ep=%d step=%d phase_max=%d "
                        "actor-VLA MAD=%.4f  (should be >0 if actor is doing something)",
                        episode, ep_steps, current_phase_max, actor_vla_mad,
                    )

                smooth.on_new_chunk(action_chunk, cfg.rl_chunk_length)
                z_rl_np = z_rl[0].cpu().numpy()
                prev_use_actor = use_actor_now

            step_in_cycle = ep_steps % infer_every
            state = obs_dict["state"]
            a_exec = smooth.next_executable_action(action_chunk, step_in_cycle)

            ep_z_rls.append(z_rl_np)
            ep_states.append(state.copy())
            ep_actions.append(a_exec)
            ep_ref_actions.append(ref_chunk_np[step_in_cycle])

            obs_dict, reward, done, info = env.step(a_exec)
            ep_steps += 1
            total_env_steps += 1
            current_phase_max = max(current_phase_max, int(info.get("phase_max", 0)))
            ep_phase_max = max(ep_phase_max, current_phase_max)
            tape_drop_term, tape_drop_streak = tape_drop_episode_should_end(
                shaped_reward_mode=cfg.shaped_reward_mode,
                end_on_tape_drop=cfg.end_on_tape_drop,
                phase_now=int(info.get("phase", 0)),
                phase_max_seen=current_phase_max,
                streak=tape_drop_streak,
            )
            if tape_drop_term:
                done = True
                ep_tape_drop_triggered = True
            ep_rewards.append(reward)
            ep_dones.append(done)
            ep_reward += reward
            ep_shaping_sum += float(info.get("shaping_reward", 0.0))
            ep_milestone_sum += float(info.get("milestone_reward", 0.0))
            ep_actor_drove.append(bool(prev_use_actor))
            if prev_use_actor:
                ep_actor_steps += 1

            phase_advanced = current_phase_max > last_video_phase_max
            if (
                log_video
                and "exo_image" in obs_dict
                # Sample one frame per inference cycle. Also force a frame
                # whenever ``phase_max`` advances — so brief phases (e.g.
                # lift_handover phase 1, which can last only a handful of sim
                # steps if arm0 releases quickly after lift) are guaranteed
                # to appear in the video instead of being skipped over by the
                # regular sampler. And one final frame on the terminal step
                # (success / tape-drop / timeout) so the success-step
                # ``r_step`` (where milestones + success_bonus fire) is
                # actually captured.
                and (ep_steps % infer_every == 0 or phase_advanced or done)
            ):
                triggered_by = (
                    "phase_advance" if phase_advanced
                    else ("done" if done else None)
                )
                ep_frames.append(
                    _annotate_rollout_exo_frame(
                        obs_dict["exo_image"],
                        r_tot=float(ep_reward),
                        r_step=float(reward),
                        phase_max=current_phase_max,  # running max — same value used by _phase_use_actor
                        use_actor=bool(prev_use_actor),
                        ep_steps=ep_steps,
                        phase=int(info.get("phase", 0)),
                        # Raw signals for debugging phase transitions; lift_handover
                        # uses g0/g1 span + first grasp arm (see info["lh_first_grasp_arm"]).
                        g0_span=float(info.get("g0_span")) if "g0_span" in info else None,
                        g1_span=float(info.get("g1_span")) if "g1_span" in info else None,
                        yellow_lift_height=(
                            float(info.get("yellow_lift_height"))
                            if "yellow_lift_height" in info
                            else None
                        ),
                        triggered_by=triggered_by,
                    ),
                )
                last_video_phase_max = current_phase_max
        # Optionally restrict the replay buffer to the RL window so the actor's seed and gradient
        # data both cover exactly the state distribution it will eventually control. Applies to
        # warm-up episodes too: the executed actions are VLA, but we only keep the actor-window slice.
        if cfg.phase_mode != "off" and cfg.phase_replay_strategy == "rl_window_only":
            if cfg.phase_mode == "task_phase":
                # Window is the contiguous run of actor-driven steps. ``phase_max`` is monotone
                # in an episode, so ep_actor_drove is at most (False*, True*, False*); find the
                # first/last True and trim to that span.
                actor_idx = [i for i, drove in enumerate(ep_actor_drove) if drove]
                if actor_idx:
                    lo, hi = actor_idx[0], actor_idx[-1] + 1
                else:
                    lo, hi = 0, 0
            else:
                lo = max(0, phase_x)
                hi = min(len(ep_z_rls), phase_y)
            if hi - lo >= cfg.rl_chunk_length:
                ep_z_rls = ep_z_rls[lo:hi]
                ep_states = ep_states[lo:hi]
                ep_actions = ep_actions[lo:hi]
                ep_ref_actions = ep_ref_actions[lo:hi]
                ep_rewards = ep_rewards[lo:hi]
                ep_dones = list(ep_dones[lo:hi])
                # Force terminal at the boundary so the critic does not bootstrap into the
                # VLA-controlled future. Episode-natural termination inside the window already
                # has done=True, so this is a no-op there.
                ep_dones[-1] = True
            else:
                # Window too short for a full chunk; skip insertion (warned at startup).
                ep_z_rls = []

        ep_success = bool(env.task_completed())
        ep_tape_dropped = ep_tape_drop_triggered
        replay_eligible = len(ep_z_rls) >= cfg.rl_chunk_length
        # Apply the success-only filter during warm-up only. After warm-up we
        # always feed the buffer so the actor sees real failures (incl. drops).
        warmup_success_filter_active = cfg.replay_only_successful_episodes and is_warmup
        # When the PyRoki oracle drove (or could have driven) this episode, do
        # not insert into the replay buffer: the macro doesn't produce per-step
        # 16D bimanual actions and we don't want the actor learning to imitate
        # an in-episode mix of VLA + oracle transitions.
        pyroki_suppress_replay = (
            args.handover_controller == "pyroki" and pyroki_macro_succeeded
        )
        insert_replay = replay_eligible and (
            not warmup_success_filter_active or ep_success
        ) and not pyroki_suppress_replay
        if insert_replay:
            replay_buffer.add_chunk_with_subsampling(
                z_rls=np.array(ep_z_rls),
                states=np.array(ep_states),
                actions=np.array(ep_actions),
                ref_actions=np.array(ep_ref_actions),
                rewards=np.array(ep_rewards),
                dones=np.array(ep_dones, dtype=np.float32),
                stride=cfg.stride,
            )

        # --- Off-policy updates (paper "Update" protocol) ---
        # Learning begins shortly after warm-up. UTD=`utd_ratio` is enforced
        # per **environment step** (not per episode), with
        # `critic_updates_per_actor` critic updates per actor update.
        critic_losses: list[float] = []
        actor_losses: list[float] = []
        bc_losses: list[float] = []
        achieved_utd = 0.0
        if not is_warmup:
            # Only post-warm-up env steps count toward the UTD budget so that
            # the learner doesn't have to burn through a huge backlog on its
            # first training episode.
            post_warmup_env_steps += ep_steps

            if async_learner is not None:
                # Kick the async learner off on the first training episode.
                if not async_learner.is_running:
                    async_learner.start()
                async_learner.record_env_steps(ep_steps)
                metrics = async_learner.pop_metrics()
                critic_losses = list(metrics["critic_losses"])  # type: ignore[arg-type]
                actor_losses = list(metrics["actor_losses"])  # type: ignore[arg-type]
                bc_losses = list(metrics["bc_losses"])  # type: ignore[arg-type]
                total_updates = int(metrics["critic_updates"]) + int(  # type: ignore[arg-type]
                    metrics["actor_updates"]  # type: ignore[arg-type]
                )
                achieved_utd = float(metrics["achieved_utd"])  # type: ignore[arg-type]
            elif replay_buffer.size >= cfg.batch_size:
                # Synchronous fallback: still UTD=5 per env step, but rollouts
                # and learning share the main thread (so rollouts block while
                # we update). Actor lock is acquired defensively for symmetry
                # with the async path.
                target_total_updates = cfg.utd_ratio * post_warmup_env_steps
                updates_this_step = max(0, target_total_updates - total_updates)
                for update_idx in range(updates_this_step):
                    batch = replay_buffer.sample(cfg.batch_size, device)
                    with actor_lock:
                        cl = update_critic_fn(
                            batch, critic, actor, target_critic,
                            critic_optimizer, cfg.discount, cfg.rl_chunk_length,
                        )
                    critic_losses.append(cl)
                    total_updates += 1
                    if total_updates % cfg.critic_updates_per_actor == 0:
                        with actor_lock:
                            al, bl = update_actor_fn(
                                batch, actor, critic, actor_optimizer,
                                cfg.bc_reg_weight, cfg.ref_action_dropout,
                            )
                        actor_losses.append(al)
                        bc_losses.append(bl)
                    soft_update_target(target_critic, critic, cfg.tau)
                achieved_utd = total_updates / max(1, post_warmup_env_steps)

        # Log training metrics every episode (including warmup) so the shaped-reward
        # signal is visible before the actor starts updating. Loss fields are only
        # populated post-warmup, so they're omitted naturally during warmup.
        log_dict = {
            "train/episode": episode,
            "train/ep_reward": ep_reward,
            "train/ep_steps": ep_steps,
            "train/buffer_size": replay_buffer.size,
            "train/total_env_steps": total_env_steps,
            "train/post_warmup_env_steps": post_warmup_env_steps,
            "train/total_updates": total_updates,
            "train/achieved_utd": achieved_utd,
            "train/success": float(ep_success),
            "train/is_warmup": float(is_warmup),
            "train/replay_skipped_failed": float(
                warmup_success_filter_active and replay_eligible and not ep_success,
            ),
            "train/tape_dropped": float(ep_tape_dropped),
            "shaped/phase_max": ep_phase_max,
            "shaped/ep_shaping_reward": ep_shaping_sum,
            "shaped/ep_milestone_reward": ep_milestone_sum,
        }
        if cfg.phase_mode != "off":
            log_dict["phase/actor_steps"] = ep_actor_steps
            log_dict["phase/rl_window_steps"] = max(
                0, min(phase_y, ep_steps) - max(phase_x, 0),
            )
        if critic_losses:
            log_dict["train/critic_loss"] = float(np.mean(critic_losses))
        if actor_losses:
            log_dict["train/actor_loss"] = float(np.mean(actor_losses))
        if bc_losses:
            log_dict["train/bc_loss"] = float(np.mean(bc_losses))
        wandb.log(log_dict, step=episode)

        # Update tqdm postfix with live metrics.
        if args.vla_only:
            phase = "vla_only"
        elif is_warmup:
            phase = "warm"
        else:
            phase = "train"
        pbar_postfix: dict = {
            "phase": phase,
            "rew": f"{ep_reward:.1f}",
            "buf": replay_buffer.size,
            "ok": int(env.task_completed()),
        }
        if critic_losses:
            pbar_postfix["crit"] = f"{np.mean(critic_losses):.3f}"
        if actor_losses:
            pbar_postfix["act"] = f"{np.mean(actor_losses):.3f}"
        pbar.set_postfix(pbar_postfix)

        # Log episode info.
        logger.info(
            f"Episode {episode} | reward={ep_reward:.2f} | steps={ep_steps} | "
            f"buffer={replay_buffer.size} | {phase} | "
            f"success={env.task_completed()} | "
            f"phase_max={ep_phase_max} | actor_steps={ep_actor_steps}/{ep_steps}"
        )

        # --- Rollout video + action comparison ---
        if log_video and args.log_video_interval > 0 and (episode % args.log_video_interval == 0):
            log_rollout_wandb(
                frames=ep_frames,
                rl_actions=np.array(ep_actions) if ep_actions else np.empty((0, cfg.action_dim)),
                ref_actions=np.array(ep_ref_actions) if ep_ref_actions else np.empty((0, cfg.action_dim)),
                episode=episode,
                fps=args.video_fps,
                video_save_dir=save_dir / "videos",
            )

        # --- Evaluation ---
        # In vla_only mode the actor is frozen/bypassed, but eval is still useful
        # to measure VLA success rate. In normal mode skip eval during warmup.
        eval_eligible = args.vla_only or (not is_warmup)
        if eval_eligible and (episode + 1) % cfg.eval_interval == 0:
            eval_metrics = evaluate(
                eval_env,
                trained_policy,
                extractor,
                encoder,
                actor,
                cfg.eval_episodes,
                cfg,
                device,
                vla_action_horizon,
                vla_only=args.vla_only,
                actor_lock=actor_lock,
                phase_x=phase_x,
                phase_y=phase_y,
                handover_controller=args.handover_controller,
                pyroki_oracle_kwargs=dict(
                    x_shift=args.pyroki_x_shift,
                    y_shift=args.pyroki_y_shift,
                    angle_shift=args.pyroki_angle_shift,
                    perturb_radius=args.pyroki_perturb_radius,
                    seed=args.seed + 9000 + episode,
                ),
                pyroki_skip_on_arm0_pick=args.pyroki_skip_on_arm0_pick,
            )
            wandb.log(eval_metrics, step=episode)
            logger.info(
                f"Eval @ episode {episode}: success={eval_metrics['eval/success_rate']:.2%} | "
                f"reward={eval_metrics['eval/mean_reward']:.2f}"
            )

        # --- Save checkpoint ---
        if (episode + 1) % cfg.save_interval == 0:
            # Pause the async learner so actor/critic/target_critic are in a
            # consistent state while we write the checkpoint. It resumes
            # immediately afterwards and retains its UTD budget counters.
            _save_with_learner_paused(
                async_learner,
                actor, critic, actor_optimizer, critic_optimizer,
                episode, save_dir,
            )

    # Final save + clean shutdown of the learner thread.
    _save_with_learner_paused(
        async_learner,
        actor, critic, actor_optimizer, critic_optimizer,
        cfg.num_episodes - 1, save_dir,
    )
    if async_learner is not None:
        async_learner.stop()
    wandb.finish()
    logger.info("Stage 2 training complete.")


def _save_with_learner_paused(
    async_learner: AsyncLearner | None,
    actor: GaussianActor,
    critic: TwinQCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    episode: int,
    save_dir: pathlib.Path,
) -> None:
    """Save a Stage-2 checkpoint with the async learner temporarily halted."""
    was_running = async_learner is not None and async_learner.is_running
    if was_running:
        async_learner.stop()
    try:
        save_stage2_checkpoint(
            actor, critic, actor_optimizer, critic_optimizer, episode, save_dir,
        )
    finally:
        if was_running:
            async_learner.start()


if __name__ == "__main__":
    train(parse_args())
