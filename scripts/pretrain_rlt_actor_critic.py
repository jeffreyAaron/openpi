"""Warm-start the Stage 2 actor-critic from a robosuite demo dataset.

Reads videos + joint .npz files from a directory tree with the same layout that
``scripts/dataset_conversion/handover_robosuite_to_lerobot.py`` consumes::

    <root>/<episode_name>/
        handover_agentview.mp4        -> exo_image
        handover_robot0_wrist.mp4     -> wrist_left_image  (robot0_is_left=True)
        handover_robot1_wrist.mp4     -> wrist_right_image
        handover_robot0_joints.npz    -> joint_positions, gripper_positions
        handover_robot1_joints.npz    -> joint_positions, gripper_positions

For each frame ``t`` the frozen VLA + frozen RL token encoder produce ``z_rl``
and a reference action chunk ``a_tilde``. Demonstration actions follow the
"next absolute pose" convention (``action[t] = state[t+1]``) used by the
LeRobot conversion script so they live in the same space the policy was
fine-tuned on.

The resulting (z_rl, state, action, ref_action, reward, done) per-step arrays
are pushed into a ``ReplayBuffer`` via ``add_chunk_with_subsampling`` with the
same chunk length / stride as Stage 2. The actor + twin Q critic are then
trained for ``--num_pretrain_updates`` steps with TD3+BC (the same
``update_critic`` and ``update_actor`` functions Stage 2 uses online), or with
pure BC when ``--bc_only`` is passed.

The checkpoint is written in the same layout Stage 2 expects::

    <checkpoint_base_dir>/stage2/<exp_name>/<step>/
        actor.safetensors
        critic.safetensors
        training_state.pt

so that ``train_rlt_stage2.py --init_actor_critic_path <ckpt_dir>`` loads it
verbatim to start fine-tuning online RL.

Example::

    cd /home/jeffreyaj/openpi && \\
    CUDA_VISIBLE_DEVICES=0 \\
    PYTHONPATH=/home/jeffreyaj/mike:/home/jeffreyaj/mike/dependencies/robosuite:src:. \\
    uv run python scripts/pretrain_rlt_actor_critic.py \\
        --dataset_dir /home/jeffreyaj/openpi/data/dataset_filtered_64_homing_removed \\
        --vla_checkpoint_path /home/jeffreyaj/openpi/checkpoints/pi05_tsh/tsh1_pytorch/40000/40000 \\
        --rl_token_checkpoint_path /home/jeffreyaj/openpi/checkpoints/rlt/stage1/rlt_stage1_handover_full_run/7650/rl_token.safetensors \\
        --exp_name rlt_actor_critic_warmstart \\
        --num_pretrain_updates 20000 \\
        --bc_reg_weight 50.0
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import sys
from typing import Any

import imageio.v2 as imageio
import jax
import numpy as np
import safetensors.torch
import torch
import tqdm
import wandb

# train_rlt_stage2.py lives next to us; reuse its helpers verbatim so the
# warm-started actor/critic are produced by exactly the same TD3+BC update
# logic that drives the online learner.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import openpi.models.pi0_config
from openpi.policies import policy_config as openpi_policy_config
import openpi.training.config as _config
from openpi.rlt.actor_critic import (
    GaussianActor,
    TwinQCritic,
    create_target_critic,
    soft_update_target,
)
from openpi.rlt.config import RLTokenModelConfig, Stage2Config
from openpi.rlt.replay_buffer import ReplayBuffer
from openpi.rlt.rl_token import RLTokenEncoder
from openpi.rlt.vla_wrapper import VLAEmbeddingExtractor
from train_rlt_stage2 import (  # noqa: E402  (sys.path tweak above)
    _load_rl_token_encoder_state_dict,
    infer_vla_actions_numpy,
    obs_dict_to_vla_observation,
    save_stage2_checkpoint,
    update_actor,
    update_critic,
)


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
    parser = argparse.ArgumentParser(
        description="Pretrain the RLT actor-critic from a robosuite demo dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Root directory containing one subdirectory per episode (each with "
        "handover_agentview.mp4, handover_robot{0,1}_wrist.mp4, handover_robot{0,1}_joints.npz).",
    )
    parser.add_argument("--vla_config_name", type=str, default="pi05_tsh")
    parser.add_argument("--vla_checkpoint_path", type=str, required=True)
    parser.add_argument(
        "--rl_token_checkpoint_path",
        type=str,
        required=True,
        help="Stage 1 checkpoint (.safetensors or .pt). The encoder weights from "
        "this checkpoint are used to compute z_rl over the demo frames.",
    )
    parser.add_argument("--exp_name", type=str, default="rlt_actor_critic_warmstart")
    parser.add_argument("--checkpoint_base_dir", type=str, default="./checkpoints/rlt")

    # Episode / dataset options.
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=-1,
        help="If >0, only ingest the first N episodes (alphabetical order). "
        "-1 = use all episodes under --dataset_dir.",
    )
    parser.add_argument(
        "--robot0_is_left",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True (default), robot0 maps to TSH 'left' (first 8 dims). Must match the "
        "convention used when fine-tuning the VLA (matches handover_robosuite_to_lerobot.py).",
    )
    parser.add_argument(
        "--gripper_qpos_max",
        type=float,
        default=0.08,
        help="Gripper finger qpos span used to normalize gripper to [-1, 1] for the policy state.",
    )
    parser.add_argument(
        "--success_terminal_reward",
        type=float,
        default=1.0,
        help="Reward placed on the last frame of every demo (assumed successful). Earlier frames "
        "get 0. Critic learns to value successful trajectories higher; actor BC dominates either way.",
    )

    # Chunk / replay-buffer settings (must match Stage 2 if you intend to fine-tune online).
    parser.add_argument("--rl_chunk_length", type=int, default=10)
    parser.add_argument("--action_dim", type=int, default=16)
    parser.add_argument("--state_dim", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--buffer_capacity", type=int, default=200_000)

    # Match Stage 2 actor/critic architecture (so weights are load-compatible).
    parser.add_argument("--actor_hidden_dim", type=int, default=256)
    parser.add_argument("--actor_num_layers", type=int, default=2)
    parser.add_argument("--actor_fixed_std", type=float, default=0.03)
    parser.add_argument("--actor_delta_max", type=float, default=1.0)
    parser.add_argument("--critic_hidden_dim", type=int, default=256)
    parser.add_argument("--critic_num_layers", type=int, default=2)

    # Optimization (TD3+BC).
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument(
        "--bc_reg_weight",
        type=float,
        default=50.0,
        help="Weight on MSE(actor mean, demo action chunk). Higher than Stage 2's online "
        "default (5.0) because pretraining is BC-dominated by design — there is no "
        "exploration data to fight the BC anchor.",
    )
    parser.add_argument("--ref_action_dropout", type=float, default=0.0,
                        help="Dropout on the reference fed to the actor's MLP. 0.0 during "
                        "pretraining keeps gradients deterministic.")
    parser.add_argument("--target_noise_std", type=float, default=0.2)
    parser.add_argument("--target_noise_clip", type=float, default=0.5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--critic_updates_per_actor", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_pretrain_updates", type=int, default=20_000,
                        help="Total critic update steps (actor steps every N=critic_updates_per_actor).")
    parser.add_argument(
        "--bc_only",
        action="store_true",
        help="Skip critic updates entirely and run pure behavior cloning (actor.forward_mean -> "
        "demo action, MSE). Critic weights are saved untrained — fine-tune them online from "
        "scratch on top of the BC-warmed actor.",
    )

    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb_enabled", action="store_true")
    parser.add_argument("--project_name", type=str, default="openpi_rlt")
    parser.add_argument("--vla_prompt", type=str,
                        default="Pick up the yellow tape, hand it over, and place it on the gray tape")
    parser.add_argument(
        "--vla_train_augmentation_at_infer",
        action="store_true",
        help="Keep RandomImageAugmentation + GaussianActionNoise in the policy pipeline. Default off.",
    )

    # RL token encoder architecture (must match Stage 1).
    parser.add_argument("--encoder_layers", type=int, default=4)
    parser.add_argument("--encoder_heads", type=int, default=8)
    _defaults = RLTokenModelConfig()
    parser.add_argument("--encoder_ff_dim", type=int, default=_defaults.encoder_ff_dim,
                        help="Encoder FFN hidden dim. Must match Stage 1.")

    return parser.parse_args()


def discover_episode_dirs(root: pathlib.Path) -> list[pathlib.Path]:
    """Same discovery convention as ``handover_robosuite_to_lerobot.py``."""
    marker = "handover_agentview.mp4"
    if (root / marker).is_file():
        return [root.resolve()]
    out: list[pathlib.Path] = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and (sub / marker).is_file():
            out.append(sub.resolve())
    if not out:
        raise FileNotFoundError(
            f"No '{marker}' under {root} or its immediate subdirectories."
        )
    return out


def _gripper_to_unit(qpos_2d: np.ndarray, *, span: float) -> np.ndarray:
    """Map robosuite finger qpos (N, 2) to [-1, +1] using the same formula as
    ``openpi.rlt.robosuite_env.format_proprio``: ``2 * sum(|qpos|)/span - 1`` (−1 open, +1 closed).

    Open: finger qpos sum ~0           -> -1.
    Closed: sum ~``span`` (e.g. 0.08)  -> +1.
    """
    g = np.asarray(qpos_2d, dtype=np.float32)
    if g.ndim != 2:
        raise ValueError(f"Expected gripper_positions (N, 2), got {g.shape}.")
    sums = np.sum(np.abs(g), axis=-1)
    return (2.0 * (sums / span) - 1.0).astype(np.float32)


def build_episode_state(
    r0: np.lib.npyio.NpzFile,
    r1: np.lib.npyio.NpzFile,
    *,
    robot0_is_left: bool,
    gripper_qpos_max: float,
) -> np.ndarray:
    """Return (T, 16) state matching ``format_proprio`` in robosuite_env.py."""
    j0 = np.asarray(r0["joint_positions"], dtype=np.float32)
    j1 = np.asarray(r1["joint_positions"], dtype=np.float32)
    g0 = _gripper_to_unit(r0["gripper_positions"], span=gripper_qpos_max)
    g1 = _gripper_to_unit(r1["gripper_positions"], span=gripper_qpos_max)
    if not (j0.shape == j1.shape and j0.shape[1] == 7):
        raise ValueError(
            f"Bad joint shapes r0={j0.shape}, r1={j1.shape}; expected (N, 7) each."
        )
    if g0.shape[0] != j0.shape[0] or g1.shape[0] != j0.shape[0]:
        raise ValueError("Gripper / joint frame counts disagree.")
    arm0 = np.concatenate([j0, g0[:, None]], axis=1)  # (N, 8)
    arm1 = np.concatenate([j1, g1[:, None]], axis=1)
    if robot0_is_left:
        return np.concatenate([arm0, arm1], axis=1)
    return np.concatenate([arm1, arm0], axis=1)


def _read_video_frames(path: pathlib.Path, expected_frames: int) -> np.ndarray:
    """Read all frames into a (T, H, W, 3) uint8 array, trimming to ``expected_frames``."""
    with imageio.get_reader(str(path)) as reader:
        frames: list[np.ndarray] = []
        for frame in reader:
            arr = np.asarray(frame, dtype=np.uint8)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[..., :3]
            frames.append(arr)
            if len(frames) >= expected_frames:
                break
    if len(frames) < expected_frames:
        # Defensive: dataset should already match joints length.
        logger.warning(
            "Video %s has %d frames but expected %d; trimming both sides.",
            path.name, len(frames), expected_frames,
        )
    return np.stack(frames, axis=0)


def _build_obs_dict(
    state_t: np.ndarray,
    exo_t: np.ndarray,
    wrist_left_t: np.ndarray,
    wrist_right_t: np.ndarray,
) -> dict[str, Any]:
    return {
        "state": state_t,
        "exo_image": exo_t,
        "wrist_left_image": wrist_left_t,
        "wrist_right_image": wrist_right_t,
    }


def ingest_episode_into_buffer(
    *,
    episode_dir: pathlib.Path,
    replay_buffer: ReplayBuffer,
    trained_policy,
    extractor: VLAEmbeddingExtractor,
    encoder: RLTokenEncoder,
    device: torch.device,
    rl_chunk_length: int,
    action_dim: int,
    stride: int,
    robot0_is_left: bool,
    gripper_qpos_max: float,
    vla_action_horizon: int,
    success_terminal_reward: float,
) -> tuple[int, int]:
    """Push subsampled chunk transitions from one demo episode into ``replay_buffer``.

    Returns ``(num_chunks_added, num_steps)``.
    """
    base_stem = "handover"
    exo_path = episode_dir / f"{base_stem}_agentview.mp4"
    w0_path = episode_dir / f"{base_stem}_robot0_wrist.mp4"
    w1_path = episode_dir / f"{base_stem}_robot1_wrist.mp4"
    j0_path = episode_dir / f"{base_stem}_robot0_joints.npz"
    j1_path = episode_dir / f"{base_stem}_robot1_joints.npz"
    for p in (exo_path, w0_path, w1_path, j0_path, j1_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing file in episode {episode_dir}: {p}")

    r0 = np.load(j0_path)
    r1 = np.load(j1_path)
    states = build_episode_state(
        r0, r1, robot0_is_left=robot0_is_left, gripper_qpos_max=gripper_qpos_max,
    )
    n = states.shape[0]
    if n < rl_chunk_length + 1:
        logger.warning(
            "%s has only %d frames (< rl_chunk_length+1 = %d); skipping.",
            episode_dir.name, n, rl_chunk_length + 1,
        )
        return 0, 0

    # Read all videos up front. Videos and joints are already aligned to the
    # same frame count by the original dataset (verified by ffprobe in
    # handover_robosuite_to_lerobot.py).
    exo_frames = _read_video_frames(exo_path, n)
    w0_frames = _read_video_frames(w0_path, n)
    w1_frames = _read_video_frames(w1_path, n)
    n = min(n, exo_frames.shape[0], w0_frames.shape[0], w1_frames.shape[0])
    exo_frames = exo_frames[:n]
    w0_frames = w0_frames[:n]
    w1_frames = w1_frames[:n]
    states = states[:n]

    if robot0_is_left:
        wrist_left_arr, wrist_right_arr = w0_frames, w1_frames
    else:
        wrist_left_arr, wrist_right_arr = w1_frames, w0_frames

    # Demo actions: a[t] = state[t+1]  (same convention as handover_robosuite_to_lerobot).
    T = n - 1
    actions = states[1:].copy()              # (T, 16)
    states_per_step = states[:-1].copy()     # (T, 16)

    # Per-step ref_actions / z_rls populated at chunk boundaries and held constant
    # for ``rl_chunk_length`` consecutive steps (mirrors the online rollout cadence
    # with action_smoothing="none").
    z_rls = np.zeros((T, replay_buffer.z_rl.shape[1]), dtype=np.float32)
    ref_actions = np.zeros((T, action_dim), dtype=np.float32)

    last_z_rl: np.ndarray | None = None
    last_ref_chunk: np.ndarray | None = None

    for t in range(T):
        if (t % rl_chunk_length) == 0 or last_z_rl is None:
            obs_dict = _build_obs_dict(
                states_per_step[t], exo_frames[t], wrist_left_arr[t], wrist_right_arr[t],
            )
            observation = obs_dict_to_vla_observation(
                trained_policy, obs_dict, device, vla_action_horizon, action_dim,
            )
            with torch.no_grad():
                z_rl = extractor.extract_rl_token(observation, encoder)  # [1, D]
                ref_chunk_np = infer_vla_actions_numpy(
                    trained_policy, obs_dict, vla_action_horizon, action_dim,
                )[:rl_chunk_length, :action_dim]
            last_z_rl = z_rl[0].cpu().numpy().astype(np.float32, copy=False)
            last_ref_chunk = ref_chunk_np.astype(np.float32, copy=False)

        z_rls[t] = last_z_rl
        # Step within the most recent VLA chunk; clamped to chunk_length-1 so even
        # an incomplete trailing chunk gets sensible reference targets.
        within = min(t % rl_chunk_length, rl_chunk_length - 1)
        ref_actions[t] = last_ref_chunk[within]

    rewards = np.zeros(T, dtype=np.float32)
    rewards[-1] = float(success_terminal_reward)
    dones = np.zeros(T, dtype=np.float32)
    dones[-1] = 1.0

    added = replay_buffer.add_chunk_with_subsampling(
        z_rls=z_rls,
        states=states_per_step,
        actions=actions,
        ref_actions=ref_actions,
        rewards=rewards,
        dones=dones,
        stride=stride,
    )
    return added, T


def update_actor_bc_only(
    batch: dict[str, torch.Tensor],
    actor: GaussianActor,
    actor_optimizer: torch.optim.Optimizer,
    *,
    grad_clip_norm: float = 0.0,
) -> tuple[float, float]:
    """Pure behavior-cloning update: minimize MSE(actor mean, demo action chunk).

    Returns (actor_loss, bc_loss). With the residual+tanh actor parametrization,
    this teaches the MLP to produce ``a_demo - a_ref`` (clamped to ``±delta_max``
    elementwise).
    """
    z_rl = batch["z_rl"]
    state = batch["state"]
    ref_actions = batch["ref_action"].reshape(z_rl.shape[0], -1)  # [B, C*d]
    demo_actions = batch["action"].reshape(z_rl.shape[0], -1)     # [B, C*d]

    action_mean = actor.forward_mean(z_rl, state, ref_actions)
    bc_loss = ((action_mean - demo_actions) ** 2).mean()

    actor_optimizer.zero_grad(set_to_none=True)
    bc_loss.backward()
    if grad_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=float(grad_clip_norm))
    actor_optimizer.step()

    return bc_loss.item(), bc_loss.item()


def update_actor_td3bc(
    batch: dict[str, torch.Tensor],
    actor: GaussianActor,
    critic: TwinQCritic,
    actor_optimizer: torch.optim.Optimizer,
    bc_reg_weight: float,
    ref_action_dropout: float,
    *,
    grad_clip_norm: float = 0.0,
) -> tuple[float, float]:
    """Pretrain-flavored actor update.

    Same shape as ``train_rlt_stage2.update_actor`` but the BC target is the
    **demonstration action chunk** (``batch['action']``), not the VLA reference,
    so the actor learns to imitate the demos while staying critic-informed.
    """
    z_rl = batch["z_rl"]
    state = batch["state"]
    ref_actions = batch["ref_action"].reshape(z_rl.shape[0], -1)
    demo_actions = batch["action"].reshape(z_rl.shape[0], -1)

    B = z_rl.shape[0]
    dropout_mask = torch.rand(B, 1, device=z_rl.device) > ref_action_dropout
    ref_actions_masked = ref_actions * dropout_mask.float()

    action_mean = actor.forward_mean(
        z_rl, state, ref_actions_masked, ref_for_residual=ref_actions,
    )

    q_value = critic.q_min(z_rl, state, action_mean)
    bc_loss = ((action_mean - demo_actions) ** 2).mean()
    actor_loss = -q_value.mean() + bc_reg_weight * bc_loss

    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    if grad_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=float(grad_clip_norm))
    actor_optimizer.step()

    return actor_loss.item(), bc_loss.item()


def main() -> None:
    args = parse_args()
    init_logging()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Episodes ---
    dataset_root = pathlib.Path(args.dataset_dir).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    episode_dirs = discover_episode_dirs(dataset_root)
    if args.max_episodes > 0:
        episode_dirs = episode_dirs[: args.max_episodes]
    logger.info("Pretraining on %d demo episodes from %s", len(episode_dirs), dataset_root)

    # --- VLA ---
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
    vla_action_horizon = int(vla_model.config.action_horizon)
    extractor = VLAEmbeddingExtractor(vla_model, device)
    logger.info("Loaded frozen VLA from %s", args.vla_checkpoint_path)

    # --- RL token encoder ---
    rl_token_config = RLTokenModelConfig(
        encoder_layers=args.encoder_layers,
        encoder_heads=args.encoder_heads,
        encoder_ff_dim=args.encoder_ff_dim,
    )
    encoder = RLTokenEncoder(rl_token_config).to(device)
    enc_state = _load_rl_token_encoder_state_dict(args.rl_token_checkpoint_path, device)
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing:
        logger.warning(
            "RL token encoder: %d missing keys after load (showing up to 5): %s",
            len(missing), missing[:5],
        )
    if unexpected:
        logger.warning(
            "RL token encoder: %d unexpected keys after load (showing up to 5): %s",
            len(unexpected), unexpected[:5],
        )
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    logger.info("Loaded frozen RL token encoder from %s", args.rl_token_checkpoint_path)

    # --- Actor / critic (architecture mirrors Stage 2) ---
    cfg = Stage2Config(
        rl_chunk_length=args.rl_chunk_length,
        action_dim=args.action_dim,
        stride=args.stride,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_num_layers=args.actor_num_layers,
        actor_fixed_std=args.actor_fixed_std,
        actor_lr=args.actor_lr,
        actor_delta_max=args.actor_delta_max,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_num_layers=args.critic_num_layers,
        critic_lr=args.critic_lr,
        discount=args.discount,
        tau=args.tau,
        bc_reg_weight=args.bc_reg_weight,
        ref_action_dropout=args.ref_action_dropout,
        target_noise_std=args.target_noise_std,
        target_noise_clip=args.target_noise_clip,
        grad_clip_norm=args.grad_clip_norm,
        critic_updates_per_actor=args.critic_updates_per_actor,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        state_dim=args.state_dim,
    )
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
    target_critic = create_target_critic(critic)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=cfg.actor_lr)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr)
    logger.info("Actor: %.1fK params | Critic: %.1fK params",
                sum(p.numel() for p in actor.parameters()) / 1e3,
                sum(p.numel() for p in critic.parameters()) / 1e3)

    # --- Replay buffer ---
    replay_buffer = ReplayBuffer(
        capacity=cfg.buffer_capacity,
        chunk_length=cfg.rl_chunk_length,
        action_dim=cfg.action_dim,
        z_rl_dim=cfg.z_rl_dim,
        state_dim=cfg.state_dim,
    )

    # --- Save dir / wandb ---
    save_dir = pathlib.Path(args.checkpoint_base_dir) / "stage2" / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.wandb_enabled:
        wandb.init(
            name=args.exp_name + "_pretrain",
            project=args.project_name,
            config=vars(args),
        )
    else:
        wandb.init(mode="disabled")

    # --- Phase 1: ingest demo episodes into the buffer ---
    total_steps = 0
    total_chunks = 0
    pbar = tqdm.tqdm(episode_dirs, desc="Ingest demos", unit="ep", dynamic_ncols=True)
    for ep_dir in pbar:
        try:
            n_chunks, n_steps = ingest_episode_into_buffer(
                episode_dir=ep_dir,
                replay_buffer=replay_buffer,
                trained_policy=trained_policy,
                extractor=extractor,
                encoder=encoder,
                device=device,
                rl_chunk_length=cfg.rl_chunk_length,
                action_dim=cfg.action_dim,
                stride=cfg.stride,
                robot0_is_left=args.robot0_is_left,
                gripper_qpos_max=args.gripper_qpos_max,
                vla_action_horizon=vla_action_horizon,
                success_terminal_reward=args.success_terminal_reward,
            )
        except Exception:
            logger.exception("Failed to ingest %s; skipping.", ep_dir)
            continue
        total_steps += n_steps
        total_chunks += n_chunks
        pbar.set_postfix({"buf": replay_buffer.size, "chunks": total_chunks})
    logger.info(
        "Ingestion complete: %d steps, %d chunks across %d episodes (buffer.size=%d).",
        total_steps, total_chunks, len(episode_dirs), replay_buffer.size,
    )
    if replay_buffer.size < cfg.batch_size:
        raise RuntimeError(
            f"Replay buffer too small ({replay_buffer.size}) for batch_size={cfg.batch_size}. "
            "Either lower --batch_size, lower --rl_chunk_length, or supply more demo episodes."
        )

    # --- Phase 2: offline pretraining (TD3+BC or BC-only) ---
    n_updates = int(args.num_pretrain_updates)
    pbar = tqdm.tqdm(total=n_updates, desc="Pretrain", unit="step", dynamic_ncols=True)
    actor_losses_recent: list[float] = []
    critic_losses_recent: list[float] = []
    bc_losses_recent: list[float] = []
    actor_step = 0
    for step in range(n_updates):
        batch = replay_buffer.sample(cfg.batch_size, device)

        if not args.bc_only:
            critic_loss = update_critic(
                batch, critic, actor, target_critic,
                critic_optimizer, cfg.discount, cfg.rl_chunk_length,
                target_noise_std=cfg.target_noise_std,
                target_noise_clip=cfg.target_noise_clip,
                grad_clip_norm=cfg.grad_clip_norm,
            )
            critic_losses_recent.append(critic_loss)
            soft_update_target(target_critic, critic, cfg.tau)
            do_actor = ((step + 1) % cfg.critic_updates_per_actor == 0)
        else:
            do_actor = True
            critic_loss = 0.0

        if do_actor:
            if args.bc_only:
                actor_loss, bc_loss = update_actor_bc_only(
                    batch, actor, actor_optimizer,
                    grad_clip_norm=cfg.grad_clip_norm,
                )
            else:
                actor_loss, bc_loss = update_actor_td3bc(
                    batch, actor, critic, actor_optimizer,
                    cfg.bc_reg_weight, cfg.ref_action_dropout,
                    grad_clip_norm=cfg.grad_clip_norm,
                )
            actor_losses_recent.append(actor_loss)
            bc_losses_recent.append(bc_loss)
            actor_step += 1

        if (step + 1) % 100 == 0 or step + 1 == n_updates:
            log_dict: dict[str, Any] = {
                "pretrain/step": step + 1,
                "pretrain/buffer_size": replay_buffer.size,
                "pretrain/actor_step": actor_step,
            }
            if critic_losses_recent:
                log_dict["pretrain/critic_loss"] = float(np.mean(critic_losses_recent))
            if actor_losses_recent:
                log_dict["pretrain/actor_loss"] = float(np.mean(actor_losses_recent))
            if bc_losses_recent:
                log_dict["pretrain/bc_loss"] = float(np.mean(bc_losses_recent))
            wandb.log(log_dict, step=step + 1)
            postfix: dict[str, Any] = {"buf": replay_buffer.size}
            if critic_losses_recent:
                postfix["crit"] = f"{np.mean(critic_losses_recent):.3f}"
            if bc_losses_recent:
                postfix["bc"] = f"{np.mean(bc_losses_recent):.4f}"
            pbar.set_postfix(postfix)
            actor_losses_recent.clear()
            critic_losses_recent.clear()
            bc_losses_recent.clear()

        if (step + 1) % args.save_interval == 0:
            save_stage2_checkpoint(
                actor, critic, actor_optimizer, critic_optimizer,
                step + 1, save_dir,
            )
        pbar.update(1)
    pbar.close()

    save_stage2_checkpoint(
        actor, critic, actor_optimizer, critic_optimizer,
        n_updates, save_dir,
    )
    wandb.finish()
    logger.info(
        "Pretraining complete. Latest checkpoint: %s",
        save_dir / str(n_updates),
    )
    logger.info(
        "To fine-tune online RL on top of this warm-start, pass:\n"
        "    --init_actor_critic_path %s",
        save_dir / str(n_updates),
    )


if __name__ == "__main__":
    main()
