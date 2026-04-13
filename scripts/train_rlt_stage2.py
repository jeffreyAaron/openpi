"""Stage 2: Online RL training with actor-critic on RL token representation.

Loads a frozen VLA + frozen RL token encoder from Stage 1, then trains
lightweight actor and critic MLPs via off-policy TD3-style RL in a
Robosuite environment.

Usage:
    uv run scripts/train_rlt_stage2.py \
        --vla_config_name pi05_tsh \
        --vla_checkpoint_path /path/to/vla/checkpoint \
        --rl_token_checkpoint_path /path/to/stage1/checkpoint/rl_token.safetensors \
        --exp_name rlt_stage2

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
    TwinQCritic,
    create_target_critic,
    soft_update_target,
)
from openpi.rlt.config import RLTokenModelConfig, Stage2Config
from openpi.rlt.env_interface import RLEnvironment
from openpi.rlt.replay_buffer import ReplayBuffer, Transition
from openpi.rlt.rl_token import RLTokenEncoder
from openpi.rlt.vla_wrapper import VLAEmbeddingExtractor


logger = logging.getLogger(__name__)


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
    parser.add_argument("--actor_hidden_dim", type=int, default=256)
    parser.add_argument("--actor_num_layers", type=int, default=2)
    parser.add_argument("--actor_fixed_std", type=float, default=0.03,
                        help="Gaussian std on flattened chunk when using stochastic rollout / actor.forward().")
    parser.add_argument("--actor_stochastic_rollout", action="store_true",
                        help="Add exploration noise during env rollout. Default: execute actor mean only (smoother).")
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_hidden_dim", type=int, default=256)
    parser.add_argument("--critic_num_layers", type=int, default=2)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--bc_reg_weight", type=float, default=5.0,
                        help="Weight on MSE(actor mean, VLA ref chunk) in actor loss.")
    parser.add_argument("--ref_action_dropout", type=float, default=0.2,
                        help="Approx. fraction of batch with ref actions zeroed during actor updates.")
    parser.add_argument("--utd_ratio", type=int, default=5)
    parser.add_argument("--critic_updates_per_actor", type=int, default=2)
    parser.add_argument("--buffer_capacity", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--warmup_episodes", type=int, default=20)
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--max_episode_steps", type=int, default=1800)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)

    # RL token model config (must match Stage 1).
    parser.add_argument("--encoder_layers", type=int, default=4)
    parser.add_argument("--encoder_heads", type=int, default=8)

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
    target_critic: TwinQCritic,
    discount: float,
    rl_chunk_length: int,
) -> torch.Tensor:
    """Compute the TD target Q-value (Eq. 3).

    Q_hat = sum_{t'=1}^C gamma^{t'-1} r_{t'} + gamma^C * min(Q1', Q2')(x', a')
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
) -> float:
    """One critic update step. Returns critic loss."""
    target = compute_td_target(batch, actor, target_critic, discount, rl_chunk_length)

    z_rl = batch["z_rl"]
    state = batch["state"]
    actions = batch["action"].reshape(z_rl.shape[0], -1)  # Flatten chunk.

    q1, q2 = critic(z_rl, state, actions)
    critic_loss = ((q1 - target.detach()) ** 2 + (q2 - target.detach()) ** 2).mean()

    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_optimizer.step()

    return critic_loss.item()


def update_actor(
    batch: dict[str, torch.Tensor],
    actor: GaussianActor,
    critic: TwinQCritic,
    actor_optimizer: torch.optim.Optimizer,
    bc_reg_weight: float,
    ref_action_dropout: float,
) -> tuple[float, float]:
    """One actor update step. Returns (actor_loss, bc_loss).

    Uses the Gaussian **mean** for both Q and BC terms so gradients anchor μ to the
    VLA reference instead of chasing noisy samples (reduces jitter vs Eq. 5 with samples).
    """
    z_rl = batch["z_rl"]
    state = batch["state"]
    ref_actions = batch["ref_action"].reshape(z_rl.shape[0], -1)  # [B, C*d]

    # Reference action dropout: zero out ref for a random fraction of the batch.
    B = z_rl.shape[0]
    dropout_mask = torch.rand(B, 1, device=z_rl.device) > ref_action_dropout
    ref_actions_masked = ref_actions * dropout_mask.float()

    action_mean = actor.forward_mean(z_rl, state, ref_actions_masked)

    # Actor loss: -Q(x, μ) + beta * ||μ - a_tilde||^2
    q_value = critic.q_min(z_rl, state, action_mean)
    bc_loss = ((action_mean - ref_actions) ** 2).mean()
    actor_loss = -q_value.mean() + bc_reg_weight * bc_loss

    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
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
) -> dict[str, float]:
    """Evaluate with the same rollout cadence / smoothing as training.

    When ``vla_only`` is True, execute the frozen VLA chunk (same as the training loop in
    warmup / ``--vla_only`` mode). Otherwise use the actor mean. Previously evaluate() always
    routed through the actor, so periodic eval under ``--vla_only`` did not measure the VLA.
    """
    actor.eval()
    successes = 0
    total_rewards = 0.0
    total_steps = 0
    infer_every = cfg.infer_every()
    C, d = cfg.rl_chunk_length, cfg.action_dim

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
                if vla_only:
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
                        action_mean = actor.forward_mean(z_rl, state_t, ref_flat)
                    action_chunk = action_mean[0].cpu().numpy().reshape(C, d)
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
        critic_hidden_dim=args.critic_hidden_dim,
        critic_num_layers=args.critic_num_layers,
        critic_lr=args.critic_lr,
        discount=args.discount,
        tau=args.tau,
        bc_reg_weight=args.bc_reg_weight,
        ref_action_dropout=args.ref_action_dropout,
        utd_ratio=args.utd_ratio,
        critic_updates_per_actor=args.critic_updates_per_actor,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        warmup_episodes=args.warmup_episodes,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_interval=args.save_interval,
        state_dim=args.state_dim,
    )
    logger.info(
        f"Rollout: action_smoothing={cfg.action_smoothing!r}, "
        f"infer_every={cfg.infer_every()} steps"
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
    )
    encoder = RLTokenEncoder(rl_token_config).to(device)
    # Load only encoder weights from the Stage 1 checkpoint.
    safetensors.torch.load_model(encoder, args.rl_token_checkpoint_path, strict=False)
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
    ).to(device)

    critic = TwinQCritic(
        z_rl_dim=cfg.z_rl_dim,
        state_dim=cfg.state_dim,
        action_chunk_dim=action_chunk_dim,
        hidden_dim=cfg.critic_hidden_dim,
        num_layers=cfg.critic_num_layers,
    ).to(device)

    target_critic = create_target_critic(critic)

    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=cfg.actor_lr)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr)

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
        env = RobosuiteRLTEnv(
            controller_cfg=args.robosuite_controller_cfg,
            image_size=args.robosuite_image_size,
            max_steps=cfg.max_episode_steps,
            seed=args.seed,
            tape_offsets_json=tape_json,
            contact_solref=contact_solref,
            contact_solimp=contact_solimp,
            tape_layout_index=layout_idx,
            gripper_action_log=grip_log,
        )
        eval_env = RobosuiteRLTEnv(
            controller_cfg=args.robosuite_controller_cfg,
            image_size=args.robosuite_image_size,
            max_steps=cfg.max_episode_steps,
            seed=args.seed + 1000,
            tape_offsets_json=tape_json,
            contact_solref=contact_solref,
            contact_solimp=contact_solimp,
            tape_layout_index=layout_idx,
            gripper_action_log=None,
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
                    ref_chunk = torch.from_numpy(
                        ref_actions_np[: cfg.rl_chunk_length, : cfg.action_dim],
                    ).float().to(device)

                state = obs_dict["state"]  # [16]

                if is_warmup:
                    action_chunk = ref_chunk.cpu().numpy()
                else:
                    ref_flat = ref_chunk.reshape(1, -1)  # [1, C*d]
                    state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
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

                smooth.on_new_chunk(action_chunk, cfg.rl_chunk_length)
                z_rl_np = z_rl[0].cpu().numpy()
                ref_chunk_np = ref_chunk.cpu().numpy()

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

        # Insert episode data into replay buffer with stride-2 subsampling.
        if len(ep_z_rls) >= cfg.rl_chunk_length:
            replay_buffer.add_chunk_with_subsampling(
                z_rls=np.array(ep_z_rls),
                states=np.array(ep_states),
                actions=np.array(ep_actions),
                ref_actions=np.array(ep_ref_actions),
                rewards=np.array(ep_rewards),
                dones=np.array(ep_dones, dtype=np.float32),
                stride=cfg.stride,
            )

        # --- Off-policy updates ---
        critic_losses: list[float] = []
        actor_losses: list[float] = []
        bc_losses: list[float] = []
        if not is_warmup and replay_buffer.size >= cfg.batch_size:
            critic_losses = []
            actor_losses = []
            bc_losses = []

            for update_idx in range(cfg.utd_ratio):
                batch = replay_buffer.sample(cfg.batch_size, device)

                # Critic update.
                cl = update_critic(
                    batch, critic, actor, target_critic,
                    critic_optimizer, cfg.discount, cfg.rl_chunk_length,
                )
                critic_losses.append(cl)

                # Actor update (every critic_updates_per_actor critic steps).
                if (update_idx + 1) % cfg.critic_updates_per_actor == 0:
                    al, bl = update_actor(
                        batch, actor, critic, actor_optimizer,
                        cfg.bc_reg_weight, cfg.ref_action_dropout,
                    )
                    actor_losses.append(al)
                    bc_losses.append(bl)

                # Soft update target critic.
                soft_update_target(target_critic, critic, cfg.tau)
                total_updates += 1

            # Log training metrics.
            log_dict = {
                "train/episode": episode,
                "train/ep_reward": ep_reward,
                "train/ep_steps": ep_steps,
                "train/buffer_size": replay_buffer.size,
                "train/total_env_steps": total_env_steps,
                "train/critic_loss": np.mean(critic_losses),
                "train/success": float(env.task_completed()),
            }
            if actor_losses:
                log_dict["train/actor_loss"] = np.mean(actor_losses)
            if bc_losses:
                log_dict["train/bc_loss"] = np.mean(bc_losses)
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
