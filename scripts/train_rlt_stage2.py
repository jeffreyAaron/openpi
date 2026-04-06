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
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import time

import numpy as np
import safetensors.torch
import torch
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
from openpi.models.model import Observation
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
    parser.add_argument("--actor_fixed_std", type=float, default=0.1)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_hidden_dim", type=int, default=256)
    parser.add_argument("--critic_num_layers", type=int, default=2)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--bc_reg_weight", type=float, default=1.0)
    parser.add_argument("--ref_action_dropout", type=float, default=0.5)
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
    parser.add_argument("--wandb_enabled", action="store_true")
    parser.add_argument("--project_name", type=str, default="openpi_rlt")
    return parser.parse_args()


def obs_dict_to_observation(obs_dict: dict, prompt_tokens: torch.Tensor | None, prompt_mask: torch.Tensor | None, device: torch.device) -> Observation:
    """Convert an env observation dict to a batched Observation for the VLA."""
    images = {}
    image_masks = {}

    camera_map = {
        "exo_image": "base_0_rgb",
        "wrist_left_image": "left_wrist_0_rgb",
        "wrist_right_image": "right_wrist_0_rgb",
    }

    for env_key, model_key in camera_map.items():
        if env_key in obs_dict:
            img = obs_dict[env_key]
            # Convert uint8 HWC to float32 in [-1, 1], add batch dim.
            img_t = torch.from_numpy(img).float() / 255.0 * 2.0 - 1.0
            images[model_key] = img_t.unsqueeze(0).to(device)  # [1, H, W, 3]
            image_masks[model_key] = torch.tensor([True], device=device)

    state = torch.from_numpy(obs_dict["state"]).float().unsqueeze(0).to(device)

    return Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=prompt_tokens,
        tokenized_prompt_mask=prompt_mask,
    )


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

    # Next-state value via target critic.
    with torch.no_grad():
        next_ref_flat = next_ref.reshape(next_ref.shape[0], -1)
        next_actions, _ = actor(next_z_rl, next_state, next_ref_flat)
        next_q = target_critic.q_min(next_z_rl, next_state, next_actions)

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
    """One actor update step. Returns (actor_loss, bc_loss)."""
    z_rl = batch["z_rl"]
    state = batch["state"]
    ref_actions = batch["ref_action"].reshape(z_rl.shape[0], -1)  # [B, C*d]

    # Reference action dropout: zero out ref for a random fraction of the batch.
    B = z_rl.shape[0]
    dropout_mask = torch.rand(B, 1, device=z_rl.device) > ref_action_dropout
    ref_actions_masked = ref_actions * dropout_mask.float()

    # Sample actions from actor.
    sampled_actions, action_mean = actor(z_rl, state, ref_actions_masked)

    # Actor loss: -Q(x, a) + beta * ||a - a_tilde||^2  (Eq. 5)
    q_value = critic.q_min(z_rl, state, sampled_actions)
    bc_loss = ((sampled_actions - ref_actions) ** 2).mean()
    actor_loss = -q_value.mean() + bc_reg_weight * bc_loss

    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_optimizer.step()

    return actor_loss.item(), bc_loss.item()


def evaluate(
    env: RLEnvironment,
    extractor: VLAEmbeddingExtractor,
    encoder: RLTokenEncoder,
    actor: GaussianActor,
    num_episodes: int,
    rl_chunk_length: int,
    action_dim: int,
    max_steps: int,
    device: torch.device,
    prompt_tokens: torch.Tensor | None,
    prompt_mask: torch.Tensor | None,
) -> dict[str, float]:
    """Evaluate the current actor policy."""
    actor.eval()
    successes = 0
    total_rewards = 0.0
    total_steps = 0

    for _ in range(num_episodes):
        obs_dict = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done and ep_steps < max_steps:
            observation = obs_dict_to_observation(obs_dict, prompt_tokens, prompt_mask, device)

            with torch.no_grad():
                z_rl = extractor.extract_rl_token(observation, encoder)
                ref_actions = extractor.sample_reference_actions(observation)
                ref_chunk = ref_actions[0, :rl_chunk_length, :action_dim]  # [C, d]
                ref_flat = ref_chunk.reshape(1, -1)  # [1, C*d]

                state = torch.from_numpy(obs_dict["state"]).float().unsqueeze(0).to(device)
                _, action_mean = actor(z_rl, state, ref_flat)

            action_chunk = action_mean[0].cpu().numpy().reshape(rl_chunk_length, action_dim)
            obs_dict, rewards, done, info = env.step_chunk(action_chunk)
            ep_reward += rewards.sum()
            ep_steps += rl_chunk_length

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


def train(args: argparse.Namespace) -> None:
    init_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = Stage2Config(
        rl_chunk_length=args.rl_chunk_length,
        action_dim=args.action_dim,
        stride=args.stride,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_num_layers=args.actor_num_layers,
        actor_fixed_std=args.actor_fixed_std,
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
    )

    import openpi.training.config as _config

    # --- Load frozen VLA ---
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

    vla_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    vla_weight_path = os.path.join(args.vla_checkpoint_path, "model.safetensors")
    safetensors.torch.load_model(vla_model, vla_weight_path)
    for p in vla_model.parameters():
        p.requires_grad_(False)
    vla_model.eval()
    logger.info(f"Loaded frozen VLA from {vla_weight_path}")

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
    from openpi.rlt.robosuite_env import RobosuiteRLTEnv

    env = RobosuiteRLTEnv(max_steps=cfg.max_episode_steps, seed=args.seed)
    eval_env = RobosuiteRLTEnv(max_steps=cfg.max_episode_steps, seed=args.seed + 1000)

    # --- Prompt tokens (fixed for the task) ---
    # TODO: extract from data config or pass as argument.
    prompt_tokens = None
    prompt_mask = None

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

    for episode in range(cfg.num_episodes):
        is_warmup = episode < cfg.warmup_episodes
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

        while not done and ep_steps < cfg.max_episode_steps:
            observation = obs_dict_to_observation(obs_dict, prompt_tokens, prompt_mask, device)

            with torch.no_grad():
                z_rl = extractor.extract_rl_token(observation, encoder)  # [1, 2048]
                ref_actions_full = extractor.sample_reference_actions(observation)  # [1, H, 32]
                ref_chunk = ref_actions_full[0, :cfg.rl_chunk_length, :cfg.action_dim]  # [C, d]

            state = obs_dict["state"]  # [16]

            if is_warmup:
                # During warmup, execute VLA reference actions directly.
                action_chunk = ref_chunk.cpu().numpy()
            else:
                # Actor produces actions conditioned on ref chunk.
                ref_flat = ref_chunk.reshape(1, -1)  # [1, C*d]
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    sampled, _ = actor(z_rl, state_t, ref_flat)
                action_chunk = sampled[0].cpu().numpy().reshape(cfg.rl_chunk_length, cfg.action_dim)

            # Execute chunk in environment, collecting per-step data.
            z_rl_np = z_rl[0].cpu().numpy()
            ref_chunk_np = ref_chunk.cpu().numpy()

            for step_in_chunk in range(cfg.rl_chunk_length):
                ep_z_rls.append(z_rl_np)
                ep_states.append(state.copy())
                ep_actions.append(action_chunk[step_in_chunk])
                ep_ref_actions.append(ref_chunk_np[step_in_chunk])

                obs_dict, reward, done, info = env.step(action_chunk[step_in_chunk])
                state = obs_dict["state"]

                ep_rewards.append(reward)
                ep_dones.append(done)
                ep_reward += reward
                ep_steps += 1

                if done:
                    break

            total_env_steps += min(cfg.rl_chunk_length, ep_steps)

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

        # Log episode info.
        logger.info(
            f"Episode {episode} | reward={ep_reward:.2f} | steps={ep_steps} | "
            f"buffer={replay_buffer.size} | {'warmup' if is_warmup else 'training'} | "
            f"success={env.task_completed()}"
        )

        # --- Evaluation ---
        if not is_warmup and (episode + 1) % cfg.eval_interval == 0:
            eval_metrics = evaluate(
                eval_env, extractor, encoder, actor,
                cfg.eval_episodes, cfg.rl_chunk_length, cfg.action_dim,
                cfg.max_episode_steps, device, prompt_tokens, prompt_mask,
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
