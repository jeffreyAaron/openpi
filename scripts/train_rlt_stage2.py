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
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
import tempfile
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
    MultiQCritic,
    create_target_critic,
    soft_update_target,
)
from openpi.rlt.config import RLTokenModelConfig, Stage2Config
from openpi.rlt.env_interface import RLEnvironment
from openpi.rlt.replay_buffer import DualReplayBuffer, ReplayBuffer, Transition
from openpi.rlt.rl_token import RLTokenEncoder
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

    # Stage 2 config overrides.
    parser.add_argument("--rl_chunk_length", type=int, default=10)
    parser.add_argument("--action_dim", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--ref_context_length", type=int, default=-1,
                        help="Length of VLA reference context fed to the actor. "
                             "-1 = same as rl_chunk_length (no extended context). "
                             "When > rl_chunk_length, the actor sees a longer VLA proposal so "
                             "it can 'compress' actions into faster motions (article).")
    parser.add_argument("--actor_hidden_dim", type=int, default=512,
                        help="Actor MLP width (article: 512).")
    parser.add_argument("--actor_num_layers", type=int, default=3,
                        help="Actor MLP hidden layer count (article: 3 = [512,512,512]).")
    parser.add_argument("--actor_fixed_std", type=float, default=0.03,
                        help="Gaussian std on flattened chunk when using stochastic rollout / actor.forward().")
    parser.add_argument("--actor_stochastic_rollout", action="store_true",
                        help="Add exploration noise during env rollout. Default: execute actor mean only (smoother).")
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_hidden_dim", type=int, default=512,
                        help="Critic MLP width (article: 512).")
    parser.add_argument("--critic_num_layers", type=int, default=3,
                        help="Critic MLP hidden layer count (article: 3 = [512,512,512]).")
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--num_critics", type=int, default=4,
                        help="Size of Q-ensemble (article: 4 after Q-spike mitigation).")
    parser.add_argument("--discount", type=float, default=0.985,
                        help="Article uses 0.985 for a 30s task @ 20Hz.")
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--bc_reg_weight", type=float, default=0.05,
                        help="BC weight on arm-joint actions (article: 0.05).")
    parser.add_argument("--gripper_bc_reg_weight", type=float, default=0.01,
                        help="BC weight on gripper dims (article: 0.01 = safer gripper exploration).")
    parser.add_argument("--jerk_reg_weight", type=float, default=0.01,
                        help="Weight on mean((mu[t+1]-mu[t])^2) in actor loss to encourage smooth actions.")
    parser.add_argument("--gradient_clip", type=float, default=20.0,
                        help="Max grad norm for actor and critic (article: halved 40->20 after Q-spike).")
    parser.add_argument("--chunk_smooth_window", type=int, default=1,
                        help="Apply a 1D uniform moving-average of this width along the time axis of the "
                             "actor's predicted action chunk at rollout time (default 1 = off). "
                             "Directly reduces step-to-step oscillation in the executed actions "
                             "independently of training progress. Try 3-7 to tame early-training noise.")
    parser.add_argument("--ref_action_dropout", type=float, default=0.2,
                        help="Approx. fraction of batch with ref actions zeroed during actor updates.")
    parser.add_argument("--utd_ratio", type=int, default=10,
                        help="Update-to-data ratio (article: 10 for [512,512,512]).")
    parser.add_argument("--critic_updates_per_actor", type=int, default=2)
    parser.add_argument("--buffer_capacity", type=int, default=100_000,
                        help="Online replay buffer capacity.")
    parser.add_argument("--demo_buffer_capacity", type=int, default=100_000,
                        help="Demo replay buffer capacity (warmup + prior runs).")
    parser.add_argument("--demo_fraction", type=float, default=0.5,
                        help="Fraction of each training batch drawn from the demo buffer.")
    parser.add_argument("--demo_buffer_path", type=str, default=None,
                        help="Optional .npz produced by a prior run; used to hydrate the demo buffer at startup.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--warmup_episodes", type=int, default=20)
    parser.add_argument(
        "--rlt_segment_start_step", type=int, default=None,
        help="Inclusive env-timestep X: RLT actor runs only on steps in [X, Y). "
             "Steps outside run the frozen VLA and are not added to the replay buffer. "
             "Default: falls back to --rlt_segment_start_step (0).",
    )
    parser.add_argument(
        "--rlt_segment_end_step", type=int, default=None,
        help="Exclusive env-timestep Y; -1 = run to end of episode. "
             "Default: falls back to --rlt_segment_end_step (-1).",
    )
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
        "Must match Stage 1. Checkpoint uses 4096 when linear1.weight has shape [4096, d_model].",
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

    return parser.parse_args()


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


def compute_td_target(
    batch: dict[str, torch.Tensor],
    actor: GaussianActor,
    target_critic: MultiQCritic,
    discount: float,
    rl_chunk_length: int,
) -> torch.Tensor:
    """Compute the TD target Q-value (Eq. 3).

    Q_hat = sum_{t'=1}^C gamma^{t'-1} r_{t'} + gamma^C * min_i(Q_i)(x', a') * (1 - done)

    ``batch["next_ref_action"]`` carries the VLA reference context at the next
    state (shape [B, H, d] where H = ref_context_length). When H > C this lets
    the actor's longer-context input differ between the current and next step,
    avoiding the previous approximation of reusing the current-step ref.
    """
    rewards = batch["reward"]  # [B, C]
    next_z_rl = batch["next_z_rl"]  # [B, z_rl_dim]
    next_state = batch["next_state"]  # [B, state_dim]
    next_ref = batch["next_ref_action"]  # [B, H, d] VLA ref at next state.
    dones = batch["done"]  # [B]

    C = rl_chunk_length
    gammas = discount ** torch.arange(C, device=rewards.device, dtype=torch.float32)
    discounted_rewards = (rewards * gammas.unsqueeze(0)).sum(dim=1)  # [B]

    with torch.no_grad():
        next_ref_flat = next_ref.reshape(next_ref.shape[0], -1)
        next_mean = actor.forward_mean(next_z_rl, next_state, next_ref_flat)
        next_q = target_critic.q_min(next_z_rl, next_state, next_mean)

    target = discounted_rewards + (discount ** C) * next_q * (1.0 - dones)
    return target


def update_critic(
    batch: dict[str, torch.Tensor],
    critic: MultiQCritic,
    actor: GaussianActor,
    target_critic: MultiQCritic,
    critic_optimizer: torch.optim.Optimizer,
    discount: float,
    rl_chunk_length: int,
    gradient_clip: float,
) -> float:
    """One critic update step over all N Q-heads. Returns mean critic loss."""
    target = compute_td_target(batch, actor, target_critic, discount, rl_chunk_length)
    target_det = target.detach()

    z_rl = batch["z_rl"]
    state = batch["state"]
    actions = batch["action"].reshape(z_rl.shape[0], -1)

    qs = critic(z_rl, state, actions)
    losses = [((q - target_det) ** 2).mean() for q in qs]
    critic_loss = torch.stack(losses).mean()

    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    if gradient_clip > 0:
        torch.nn.utils.clip_grad_norm_(critic.parameters(), gradient_clip)
    critic_optimizer.step()

    return critic_loss.item()


def update_actor(
    batch: dict[str, torch.Tensor],
    actor: GaussianActor,
    critic: MultiQCritic,
    actor_optimizer: torch.optim.Optimizer,
    bc_reg_weight: float,
    gripper_bc_reg_weight: float,
    gripper_action_indices: tuple[int, ...],
    action_dim: int,
    rl_chunk_length: int,
    jerk_reg_weight: float,
    ref_action_dropout: float,
    gradient_clip: float,
) -> tuple[float, float, float]:
    """One actor update step. Returns (actor_loss, bc_loss, jerk_loss).

    Actor loss (article):
        L = -Q(x, μ) + beta_arm * MSE_arm(μ, a_ref) + beta_gripper * MSE_gripper(μ, a_ref)
            + w_jerk * mean((μ[t+1] - μ[t])^2)

    The per-joint BC split (arm vs gripper) lets the gripper explore more
    aggressively than the arms (article: bc_beta=0.05 vs gripper 0.01). The
    jerk term offsets the reduced anchoring by penalizing non-smooth chunks.

    Uses the Gaussian mean (not stochastic samples) so gradients anchor μ
    directly to the VLA reference instead of chasing noise.
    """
    z_rl = batch["z_rl"]
    state = batch["state"]
    B = z_rl.shape[0]
    C = rl_chunk_length

    ref_actions_full = batch["ref_action"]  # [B, H, d]
    ref_actions_full_flat = ref_actions_full.reshape(B, -1)

    # Only the first C steps of the reference overlap with the executed chunk,
    # so BC is only computed there; the rest is extra actor-input context.
    ref_actions_bc = ref_actions_full[:, :C, :]  # [B, C, d]

    dropout_mask = torch.rand(B, 1, device=z_rl.device) > ref_action_dropout
    ref_actions_masked = ref_actions_full_flat * dropout_mask.float()

    action_mean_flat = actor.forward_mean(z_rl, state, ref_actions_masked)  # [B, C*d]
    action_mean = action_mean_flat.reshape(B, C, action_dim)

    gripper_idx = list(gripper_action_indices)
    arm_idx = [i for i in range(action_dim) if i not in gripper_idx]
    arm_bc = ((action_mean[..., arm_idx] - ref_actions_bc[..., arm_idx]) ** 2).mean()
    if gripper_idx:
        gripper_bc = ((action_mean[..., gripper_idx] - ref_actions_bc[..., gripper_idx]) ** 2).mean()
    else:
        gripper_bc = torch.zeros((), device=z_rl.device)
    bc_loss_weighted = bc_reg_weight * arm_bc + gripper_bc_reg_weight * gripper_bc
    bc_loss_raw = ((action_mean - ref_actions_bc) ** 2).mean()

    if C > 1 and jerk_reg_weight > 0:
        jerk_loss = ((action_mean[:, 1:] - action_mean[:, :-1]) ** 2).mean()
    else:
        jerk_loss = torch.zeros((), device=z_rl.device)

    q_value = critic.q_min(z_rl, state, action_mean_flat)
    actor_loss = -q_value.mean() + bc_loss_weighted + jerk_reg_weight * jerk_loss

    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    if gradient_clip > 0:
        torch.nn.utils.clip_grad_norm_(actor.parameters(), gradient_clip)
    actor_optimizer.step()

    return actor_loss.item(), bc_loss_raw.item(), jerk_loss.item()


def _resolve_ref_context_length(cfg: Stage2Config, vla_action_horizon: int) -> int:
    """Resolve the actor's VLA-reference context length, clamped to VLA horizon."""
    H = cfg.ref_context_length if cfg.ref_context_length > 0 else cfg.rl_chunk_length
    return max(cfg.rl_chunk_length, min(H, vla_action_horizon))


def _extract_ref_context(
    ref_actions_np: np.ndarray, ref_context_length: int, action_dim: int
) -> np.ndarray:
    """Right-pad VLA reference actions with zeros to ``ref_context_length`` rows."""
    H = ref_context_length
    take = min(H, ref_actions_np.shape[0])
    out = np.zeros((H, action_dim), dtype=np.float32)
    out[:take] = ref_actions_np[:take, :action_dim]
    return out


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
    chunk_smooth_window: int = 1,
) -> dict[str, float]:
    """Evaluate with the same rollout cadence / smoothing as training.

    Segment gating mirrors the training loop: the RLT actor runs only inside
    ``cfg.is_rlt_segment_step(ep_steps)`` (i.e. ``[rlt_segment_start_step, Y)``).
    When ``vla_only`` is True, always execute the frozen VLA chunk.
    """
    actor.eval()
    successes = 0
    total_rewards = 0.0
    total_steps = 0
    infer_every = cfg.infer_every()
    C, d = cfg.rl_chunk_length, cfg.action_dim
    H = _resolve_ref_context_length(cfg, vla_action_horizon)

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

        while not done and ep_steps < cfg.max_episode_steps:
            if ep_steps % infer_every == 0:
                ref_actions_np = infer_vla_actions_numpy(
                    trained_policy, obs_dict, vla_action_horizon, d,
                )
                use_vla = vla_only or not cfg.is_rlt_segment_step(ep_steps)
                if use_vla:
                    action_chunk = ref_actions_np[:C, :d].copy()
                else:
                    observation = obs_dict_to_vla_observation(
                        trained_policy, obs_dict, device, vla_action_horizon, d,
                    )
                    with torch.no_grad():
                        z_rl = extractor.extract_rl_token(observation, encoder)
                        ref_ctx_np = _extract_ref_context(ref_actions_np, H, d)
                        ref_ctx = torch.from_numpy(ref_ctx_np).float().unsqueeze(0).to(device)
                        ref_flat = ref_ctx.reshape(1, -1)
                        state_t = torch.from_numpy(obs_dict["state"]).float().unsqueeze(0).to(device)
                        action_mean = actor.forward_mean(z_rl, state_t, ref_flat)
                    action_chunk = _smooth_action_chunk(
                        action_mean[0].cpu().numpy().reshape(C, d), chunk_smooth_window,
                    )
                smooth.on_new_chunk(action_chunk, C)

            step_in_cycle = ep_steps % infer_every
            a_exec = smooth.next_executable_action(action_chunk, step_in_cycle)
            obs_dict, reward, done, info = env.step(a_exec)
            ep_reward += reward
            ep_steps += 1

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
    critic: MultiQCritic,
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
        frames: List of uint8 HWC images collected once per action chunk.
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


def _hydrate_demo_buffer(buffer: ReplayBuffer, demo_path: str) -> None:
    """Populate ``buffer`` from a ``.npz`` dump of prior-run transitions.

    Expected keys: z_rl, states, actions, ref_actions, rewards, next_z_rl,
    next_states, next_ref_actions, dones. Shapes must match ``buffer``'s arrays.
    Extra entries beyond capacity are silently dropped (ring buffer).
    """
    path = pathlib.Path(demo_path)
    if not path.exists():
        logger.warning(f"--demo_buffer_path {demo_path} not found; skipping hydrate.")
        return
    data = np.load(str(path))
    required = [
        "z_rl", "states", "actions", "ref_actions", "rewards",
        "next_z_rl", "next_states", "next_ref_actions", "dones",
    ]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(
            f"Demo buffer file {demo_path} missing keys: {missing}. "
            f"Expected: {required}"
        )
    n = int(data["z_rl"].shape[0])
    for i in range(n):
        transition = Transition(
            z_rl=data["z_rl"][i],
            state=data["states"][i],
            action_chunk=data["actions"][i],
            ref_action_chunk=data["ref_actions"][i],
            rewards=data["rewards"][i],
            next_z_rl=data["next_z_rl"][i],
            next_state=data["next_states"][i],
            next_ref_action_chunk=data["next_ref_actions"][i],
            done=bool(data["dones"][i]),
        )
        buffer.add(transition)
    logger.info(f"Hydrated demo buffer from {demo_path}: {buffer.size} transitions.")


def _smooth_action_chunk(chunk: np.ndarray, window: int) -> np.ndarray:
    """Apply a 1D uniform moving average along the time axis of a ``[T, d]`` chunk.

    Each action dimension is convolved independently with a box kernel of width
    ``window``. The ``mode='same'`` convolution preserves array length; boundary
    steps are averaged over fewer samples (numpy ``convolve`` zero-pad equivalent
    is avoided by using a reflected pad for edge steps via mode='full' + trim, but
    'same' is sufficient for smoothing purposes). ``window <= 1`` is a no-op.
    """
    if window <= 1 or chunk.shape[0] == 0:
        return chunk
    w = min(window, chunk.shape[0])
    kernel = np.ones(w, dtype=np.float32) / w
    return np.stack(
        [np.convolve(chunk[:, i], kernel, mode="same") for i in range(chunk.shape[1])],
        axis=1,
    ).astype(chunk.dtype)


def _use_vla_at_step(
    ep_steps: int,
    is_warmup: bool,
    vla_only: bool,
    rlt_segment_start_step: int,
    rlt_segment_end_step: int,
) -> bool:
    """Return True if the VLA (not the RLT actor) should produce the action chunk.

    Priority:
      1. ``vla_only`` or episode-level warm-up  → always VLA.
      2. ``ep_steps < rlt_segment_start_step``         → pre-RLT VLA segment.
      3. ``ep_steps >= rlt_segment_end_step >= 0``     → post-RLT VLA segment.
      4. Otherwise                               → RLT actor.
    """
    if vla_only or is_warmup:
        return True
    if ep_steps < rlt_segment_start_step:
        return True
    if rlt_segment_end_step >= 0 and ep_steps >= rlt_segment_end_step:
        return True
    return False


def _hybrid_step_phase(
    ep_steps: int,
    is_warmup: bool,
    vla_only: bool,
    rlt_segment_start_step: int,
    rlt_segment_end_step: int,
) -> str:
    """Human-readable phase label for the current step."""
    if vla_only:
        return "vla_only"
    if is_warmup:
        return "warm"
    if ep_steps < rlt_segment_start_step:
        return "vla_pre"
    if rlt_segment_end_step >= 0 and ep_steps >= rlt_segment_end_step:
        return "vla_post"
    return "rlt"


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

    # Resolve None defaults: None means "no segment restriction" → start=0, end=-1 (disabled).
    segment_start = args.rlt_segment_start_step if args.rlt_segment_start_step is not None else 0
    segment_end = args.rlt_segment_end_step if args.rlt_segment_end_step is not None else -1

    cfg = Stage2Config(
        rl_chunk_length=args.rl_chunk_length,
        action_dim=args.action_dim,
        stride=args.stride,
        ref_context_length=args.ref_context_length,
        action_smoothing=args.action_smoothing,
        inference_frequency=infer_freq,
        te_k=args.te_k,
        ema_alpha=args.ema_alpha,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_num_layers=args.actor_num_layers,
        actor_fixed_std=args.actor_fixed_std,
        actor_stochastic_rollout=args.actor_stochastic_rollout,
        actor_lr=args.actor_lr,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_num_layers=args.critic_num_layers,
        critic_lr=args.critic_lr,
        num_critics=args.num_critics,
        discount=args.discount,
        tau=args.tau,
        bc_reg_weight=args.bc_reg_weight,
        gripper_bc_reg_weight=args.gripper_bc_reg_weight,
        ref_action_dropout=args.ref_action_dropout,
        utd_ratio=args.utd_ratio,
        critic_updates_per_actor=args.critic_updates_per_actor,
        gradient_clip=args.gradient_clip,
        jerk_reg_weight=args.jerk_reg_weight,
        buffer_capacity=args.buffer_capacity,
        demo_buffer_capacity=args.demo_buffer_capacity,
        demo_fraction=args.demo_fraction,
        batch_size=args.batch_size,
        warmup_episodes=args.warmup_episodes,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_interval=args.save_interval,
        state_dim=args.state_dim,
        rlt_segment_start_step=segment_start,
        rlt_segment_end_step=segment_end,
    )
    logger.info(
        f"Rollout: action_smoothing={cfg.action_smoothing!r}, "
        f"infer_every={cfg.infer_every()} steps"
    )
    logger.info(
        f"RLT segment: [{cfg.rlt_segment_start_step}, "
        f"{cfg.segment_end_step()}) env steps"
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
    ref_context_length = _resolve_ref_context_length(cfg, vla_action_horizon)
    ref_context_chunk_dim = ref_context_length * cfg.action_dim
    if ref_context_length != (cfg.ref_context_length if cfg.ref_context_length > 0 else cfg.rl_chunk_length):
        logger.info(
            f"ref_context_length clamped to [rl_chunk_length={cfg.rl_chunk_length}, "
            f"vla_action_horizon={vla_action_horizon}] → {ref_context_length}"
        )
    actor = GaussianActor(
        z_rl_dim=cfg.z_rl_dim,
        state_dim=cfg.state_dim,
        action_chunk_dim=action_chunk_dim,
        ref_context_chunk_dim=ref_context_chunk_dim,
        hidden_dim=cfg.actor_hidden_dim,
        num_layers=cfg.actor_num_layers,
        fixed_std=cfg.actor_fixed_std,
    ).to(device)

    critic = MultiQCritic(
        z_rl_dim=cfg.z_rl_dim,
        state_dim=cfg.state_dim,
        action_chunk_dim=action_chunk_dim,
        hidden_dim=cfg.critic_hidden_dim,
        num_layers=cfg.critic_num_layers,
        num_critics=cfg.num_critics,
    ).to(device)

    target_critic = create_target_critic(critic)

    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=cfg.actor_lr)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr)

    logger.info(f"Actor: {sum(p.numel() for p in actor.parameters()) / 1e3:.1f}K params")
    logger.info(
        f"Critic ({cfg.num_critics} heads): "
        f"{sum(p.numel() for p in critic.parameters()) / 1e3:.1f}K params"
    )

    # --- Replay buffers (demo + online; article's two-buffer setup) ---
    replay_buffer = DualReplayBuffer(
        demo_capacity=cfg.demo_buffer_capacity,
        online_capacity=cfg.buffer_capacity,
        chunk_length=cfg.rl_chunk_length,
        action_dim=cfg.action_dim,
        z_rl_dim=cfg.z_rl_dim,
        state_dim=cfg.state_dim,
        ref_context_length=ref_context_length,
    )
    if args.demo_buffer_path is not None:
        _hydrate_demo_buffer(replay_buffer.demo_buffer, args.demo_buffer_path)

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

    # --- Main training loop (Algorithm 1) ---
    total_env_steps = 0
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

        while not done and ep_steps < cfg.max_episode_steps:
            # One video frame per VLA+actor cycle (same cadence as pre-smoothing).
            if log_video and ep_steps % infer_every == 0 and "exo_image" in obs_dict:
                ep_frames.append(obs_dict["exo_image"].copy())

            if ep_steps % infer_every == 0:
                observation = obs_dict_to_vla_observation(
                    trained_policy, obs_dict, device, vla_action_horizon, cfg.action_dim,
                )

                with torch.no_grad():
                    z_rl = extractor.extract_rl_token(observation, encoder)  # [1, 2048]
                    ref_actions_np = infer_vla_actions_numpy(
                        trained_policy, obs_dict, vla_action_horizon, cfg.action_dim,
                    )
                    # Executed chunk (first C of the VLA proposal).
                    ref_chunk_np_full = ref_actions_np[
                        : cfg.rl_chunk_length, : cfg.action_dim
                    ].astype(np.float32)
                    # Extended context fed to the actor (first H of the VLA proposal).
                    ref_ctx_np = _extract_ref_context(
                        ref_actions_np, ref_context_length, cfg.action_dim,
                    )
                    ref_ctx_t = torch.from_numpy(ref_ctx_np).float().unsqueeze(0).to(device)

                state = obs_dict["state"]

                # Segment gating: RLT actor only inside [X, Y), VLA elsewhere.
                # Warmup and --vla_only always use VLA.
                use_vla = is_warmup or args.vla_only or not cfg.is_rlt_segment_step(ep_steps)

                if use_vla:
                    action_chunk = ref_chunk_np_full.copy()
                else:
                    state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                    ref_flat = ref_ctx_t.reshape(1, -1)
                    with torch.no_grad():
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
                        action_chunk = _smooth_action_chunk(action_chunk, args.chunk_smooth_window)

                smooth.on_new_chunk(action_chunk, cfg.rl_chunk_length)
                z_rl_np = z_rl[0].cpu().numpy()
                ref_chunk_np = ref_chunk_np_full

            step_in_cycle = ep_steps % infer_every
            state = obs_dict["state"]
            a_exec = smooth.next_executable_action(action_chunk, step_in_cycle)

            ep_z_rls.append(z_rl_np)
            ep_states.append(state.copy())
            ep_actions.append(a_exec)
            ep_ref_actions.append(ref_chunk_np[step_in_cycle])

            obs_dict, reward, done, info = env.step(a_exec)
            ep_rewards.append(reward)
            ep_dones.append(done)
            ep_reward += reward
            ep_steps += 1
            total_env_steps += 1

        # --- Segment-scoped buffer insertion with stride-N subsampling ---
        # Only keep the slice of the episode that lies in [rlt_segment_start_step, Y),
        # where Y = cfg.segment_end_step(). Transitions whose chunk ends at or past
        # Y are marked terminal for RLT bootstrapping even if the env episode continues.
        seg_start = cfg.rlt_segment_start_step
        seg_end = min(cfg.segment_end_step(), len(ep_z_rls))
        seg_len = max(0, seg_end - seg_start)
        if seg_len >= cfg.rl_chunk_length:
            z_rls_arr = np.array(ep_z_rls[seg_start:seg_end])
            states_arr = np.array(ep_states[seg_start:seg_end])
            actions_arr = np.array(ep_actions[seg_start:seg_end])
            ref_arr = np.array(ep_ref_actions[seg_start:seg_end])
            rewards_arr = np.array(ep_rewards[seg_start:seg_end])
            dones_arr = np.array(ep_dones[seg_start:seg_end], dtype=np.float32)
            # Segment end (relative to the slice) == seg_len; any chunk ending
            # there is terminal for RLT bootstrapping.
            if is_warmup:
                replay_buffer.add_chunk_with_subsampling_to_demo(
                    z_rls=z_rls_arr,
                    states=states_arr,
                    actions=actions_arr,
                    ref_actions=ref_arr,
                    rewards=rewards_arr,
                    dones=dones_arr,
                    stride=cfg.stride,
                    segment_end_step=seg_len,
                )
            else:
                replay_buffer.add_chunk_with_subsampling_to_online(
                    z_rls=z_rls_arr,
                    states=states_arr,
                    actions=actions_arr,
                    ref_actions=ref_arr,
                    rewards=rewards_arr,
                    dones=dones_arr,
                    stride=cfg.stride,
                    segment_end_step=seg_len,
                )

        # --- Off-policy updates ---
        critic_losses: list[float] = []
        actor_losses: list[float] = []
        bc_losses: list[float] = []
        jerk_losses: list[float] = []
        if not is_warmup and replay_buffer.size >= cfg.batch_size:
            for update_idx in range(cfg.utd_ratio):
                batch = replay_buffer.sample(
                    cfg.batch_size, device, demo_fraction=cfg.demo_fraction,
                )

                cl = update_critic(
                    batch, critic, actor, target_critic,
                    critic_optimizer, cfg.discount, cfg.rl_chunk_length,
                    cfg.gradient_clip,
                )
                critic_losses.append(cl)

                if (update_idx + 1) % cfg.critic_updates_per_actor == 0:
                    al, bl, jl = update_actor(
                        batch, actor, critic, actor_optimizer,
                        bc_reg_weight=cfg.bc_reg_weight,
                        gripper_bc_reg_weight=cfg.gripper_bc_reg_weight,
                        gripper_action_indices=cfg.gripper_action_indices,
                        action_dim=cfg.action_dim,
                        rl_chunk_length=cfg.rl_chunk_length,
                        jerk_reg_weight=cfg.jerk_reg_weight,
                        ref_action_dropout=cfg.ref_action_dropout,
                        gradient_clip=cfg.gradient_clip,
                    )
                    actor_losses.append(al)
                    bc_losses.append(bl)
                    jerk_losses.append(jl)

                soft_update_target(target_critic, critic, cfg.tau)
                total_updates += 1

            log_dict = {
                "train/episode": episode,
                "train/ep_reward": ep_reward,
                "train/ep_steps": ep_steps,
                "train/buffer_size": replay_buffer.size,
                "train/demo_buffer_size": replay_buffer.demo_size,
                "train/online_buffer_size": replay_buffer.online_size,
                "train/total_env_steps": total_env_steps,
                "train/critic_loss": np.mean(critic_losses),
                "train/success": float(env.task_completed()),
            }
            if actor_losses:
                log_dict["train/actor_loss"] = np.mean(actor_losses)
            if bc_losses:
                log_dict["train/bc_loss"] = np.mean(bc_losses)
            if jerk_losses:
                log_dict["train/jerk_loss"] = np.mean(jerk_losses)
            wandb.log(log_dict, step=episode)

        # Update tqdm postfix with live metrics.
        # phase reflects the policy source at the final step of the episode.
        # Use resolved cfg values (safe integers) rather than the nullable args.
        phase = _hybrid_step_phase(
            max(0, ep_steps - 1), is_warmup, args.vla_only,
            cfg.rlt_segment_start_step, cfg.rlt_segment_end_step,
        )
        # When hybrid bracketing is active, annotate with the full structure.
        hybrid_active = cfg.rlt_segment_start_step > 0 or cfg.rlt_segment_end_step >= 0
        if hybrid_active and not is_warmup and not args.vla_only:
            rlt_end_label = cfg.segment_end_step()
            phase_label = (
                f"vla[0:{cfg.rlt_segment_start_step}]→"
                f"rlt[{cfg.rlt_segment_start_step}:{rlt_end_label}]"
            )
            if cfg.rlt_segment_end_step >= 0:
                phase_label += f"→vla[{cfg.rlt_segment_end_step}:]"
        else:
            phase_label = phase
        pbar_postfix: dict = {
            "phase": phase_label,
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
            f"buffer={replay_buffer.size} | {phase_label} | "
            f"success={env.task_completed()}"
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
                chunk_smooth_window=args.chunk_smooth_window,
            )
            wandb.log(eval_metrics, step=episode)
            logger.info(
                f"Eval @ episode {episode}: success={eval_metrics['eval/success_rate']:.2%} | "
                f"reward={eval_metrics['eval/mean_reward']:.2f}"
            )

        # --- Save checkpoint ---
        if (episode + 1) % cfg.save_interval == 0:
            save_stage2_checkpoint(
                actor, critic, actor_optimizer, critic_optimizer, episode, save_dir,
            )

    # Final save.
    save_stage2_checkpoint(
        actor, critic, actor_optimizer, critic_optimizer, cfg.num_episodes - 1, save_dir,
    )
    wandb.finish()
    logger.info("Stage 2 training complete.")


if __name__ == "__main__":
    train(parse_args())
