"""Stage 1: Train the RL token encoder-decoder on demonstration data.

Trains a small encoder-decoder transformer to compress the frozen VLA's
internal embeddings into a compact RL token z_rl. Optionally fine-tunes
the VLA simultaneously (controlled by --joint_vla_finetune).

Usage:
    # RL token only (VLA already fine-tuned):
    uv run scripts/train_rlt_stage1.py \
        --vla_config_name pi05_tsh \
        --vla_checkpoint_path /path/to/finetuned/vla/checkpoint \
        --exp_name rlt_stage1

    # Joint RL token + VLA fine-tuning (from base VLA):
    uv run scripts/train_rlt_stage1.py \
        --vla_config_name pi05_tsh \
        --vla_checkpoint_path /path/to/base/vla/checkpoint \
        --joint_vla_finetune \
        --exp_name rlt_stage1_joint

    # Local LeRobot dataset (overrides the Hub repo_id baked into the named config):
    uv run scripts/train_rlt_stage1.py \
        --vla_config_name pi05_tsh \
        --lerobot_repo_id /path/to/lerobot_dataset_root \
        --vla_checkpoint_path /path/to/vla/checkpoint \
        --exp_name rlt_stage1
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
import time

import jax
import numpy as np
import safetensors.torch
import torch
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.rlt.config import RLTConfig, Stage1Config
from openpi.rlt.rl_token import RLTokenModule
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
    parser = argparse.ArgumentParser(description="RLT Stage 1: RL Token Training")
    parser.add_argument("--vla_config_name", type=str, default="pi05_tsh")
    parser.add_argument(
        "--lerobot_repo_id",
        type=str,
        default=None,
        help="LeRobot dataset: Hugging Face repo id (e.g. org/name) or absolute path to a local "
        "LeRobot dataset root. If set, overrides repo_id from the selected --vla_config_name.",
    )
    parser.add_argument("--vla_checkpoint_path", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="rlt_stage1")
    parser.add_argument("--checkpoint_base_dir", type=str, default="./checkpoints/rlt")

    # Stage 1 config overrides.
    parser.add_argument("--joint_vla_finetune", action="store_true")
    parser.add_argument("--vla_loss_weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_train_steps", type=int, default=10_000)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--save_interval", type=int, default=2_000)
    parser.add_argument("--gradient_clip", type=float, default=1.0)

    # RL token model config overrides.
    parser.add_argument("--encoder_layers", type=int, default=4)
    parser.add_argument("--decoder_layers", type=int, default=4)
    parser.add_argument("--encoder_heads", type=int, default=8)
    parser.add_argument("--decoder_heads", type=int, default=8)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_enabled", action="store_true")
    parser.add_argument("--project_name", type=str, default="openpi_rlt")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_checkpoint(
    rl_token_module: RLTokenModule,
    vla_model: openpi.models_pytorch.pi0_pytorch.PI0Pytorch | None,
    optimizer: torch.optim.Optimizer,
    step: int,
    save_dir: pathlib.Path,
) -> None:
    """Save RL token (and optionally VLA) checkpoint."""
    ckpt_dir = save_dir / str(step)
    tmp_dir = save_dir / f"tmp_{step}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    safetensors.torch.save_model(rl_token_module, str(tmp_dir / "rl_token.safetensors"))
    torch.save({"optimizer": optimizer.state_dict(), "step": step}, str(tmp_dir / "training_state.pt"))

    if vla_model is not None:
        safetensors.torch.save_model(vla_model, str(tmp_dir / "model.safetensors"))

    # Atomic rename.
    if ckpt_dir.exists():
        import shutil

        shutil.rmtree(ckpt_dir)
    tmp_dir.rename(ckpt_dir)
    logger.info(f"Saved checkpoint at step {step} to {ckpt_dir}")


def lr_schedule(step: int, warmup_steps: int, peak_lr: float) -> float:
    """Linear warmup then constant LR."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / (warmup_steps + 1)
    return peak_lr


def train(args: argparse.Namespace) -> None:
    init_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load VLA config and data ---
    train_config = _config.get_config(args.vla_config_name)
    if args.lerobot_repo_id is not None:
        repo = os.path.expanduser(args.lerobot_repo_id)
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(train_config.data, repo_id=repo),
        )
        logger.info("Using LeRobot repo_id override: %s", repo)
    # Override batch size.
    train_config = _config.TrainConfig(
        **{
            **{f.name: getattr(train_config, f.name) for f in train_config.__dataclass_fields__.values()},
            "batch_size": args.batch_size,
            "exp_name": args.exp_name,
        }
    )
    loader, data_config = _data.create_data_loader(train_config, framework="pytorch", shuffle=True), None

    # --- Build VLA model ---
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

    # Load VLA weights.
    vla_weight_path = os.path.join(args.vla_checkpoint_path, "model.safetensors")
    safetensors.torch.load_model(vla_model, vla_weight_path)
    logger.info(f"Loaded VLA weights from {vla_weight_path}")

    # Freeze VLA if not jointly fine-tuning.
    if not args.joint_vla_finetune:
        for p in vla_model.parameters():
            p.requires_grad_(False)
        vla_model.eval()
        logger.info("VLA frozen (RL token training only)")
    else:
        vla_model.train()
        logger.info("VLA will be jointly fine-tuned")

    extractor = VLAEmbeddingExtractor(vla_model, device)

    # --- Build RL token module ---
    from openpi.rlt.config import RLTokenModelConfig

    rl_token_config = RLTokenModelConfig(
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        encoder_heads=args.encoder_heads,
        decoder_heads=args.decoder_heads,
    )
    rl_token_module = RLTokenModule(rl_token_config).to(device)
    logger.info(
        f"RL token module: {sum(p.numel() for p in rl_token_module.parameters()) / 1e6:.1f}M params"
    )

    # --- Optimizer ---
    params_to_optimize = list(rl_token_module.parameters())
    if args.joint_vla_finetune:
        params_to_optimize += list(vla_model.parameters())
        logger.info(
            f"Total trainable params: {sum(p.numel() for p in params_to_optimize) / 1e6:.1f}M"
        )

    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=1e-4)

    # --- Checkpoint directory ---
    save_dir = pathlib.Path(args.checkpoint_base_dir) / "stage1" / args.exp_name
    if args.overwrite and save_dir.exists():
        import shutil

        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Wandb ---
    if args.wandb_enabled:
        wandb.init(name=args.exp_name, project=args.project_name, config=vars(args))
    else:
        wandb.init(mode="disabled")

    # --- Training loop ---
    global_step = 0
    pbar = tqdm.tqdm(total=args.num_train_steps, desc="Stage 1")
    start_time = time.time()

    while global_step < args.num_train_steps:
        for observation, actions in loader:
            if global_step >= args.num_train_steps:
                break

            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(torch.float32).to(device)

            # Update LR.
            current_lr = lr_schedule(global_step, args.warmup_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            # Extract VLA image embeddings.
            if args.joint_vla_finetune:
                img_embs, img_pad_mask = extractor.extract_image_embeddings(observation)
            else:
                with torch.no_grad():
                    img_embs, img_pad_mask = extractor.extract_image_embeddings(observation)

            # RL token reconstruction loss.
            z_rl, recon_loss = rl_token_module(img_embs, img_pad_mask)

            # Optional VLA fine-tuning loss.
            total_loss = recon_loss
            if args.joint_vla_finetune:
                vla_loss = vla_model(observation, actions).mean()
                total_loss = recon_loss + args.vla_loss_weight * vla_loss

            # Backward + optimize.
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=args.gradient_clip)
            optimizer.step()

            # Logging.
            if global_step % 100 == 0:
                log_dict = {
                    "stage1/recon_loss": recon_loss.item(),
                    "stage1/total_loss": total_loss.item(),
                    "stage1/lr": current_lr,
                    "stage1/z_rl_norm": z_rl.detach().norm(dim=-1).mean().item(),
                }
                if args.joint_vla_finetune:
                    log_dict["stage1/vla_loss"] = vla_loss.item()
                wandb.log(log_dict, step=global_step)

                elapsed = time.time() - start_time
                steps_per_sec = (global_step + 1) / elapsed
                logger.info(
                    f"Step {global_step} | recon_loss={recon_loss.item():.4f} | "
                    f"total_loss={total_loss.item():.4f} | lr={current_lr:.2e} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

            # Save checkpoint.
            if (global_step > 0 and global_step % args.save_interval == 0) or (
                global_step == args.num_train_steps - 1
            ):
                vla_to_save = vla_model if args.joint_vla_finetune else None
                save_checkpoint(rl_token_module, vla_to_save, optimizer, global_step, save_dir)

            global_step += 1
            pbar.update(1)

    pbar.close()
    wandb.finish()
    logger.info("Stage 1 training complete.")


if __name__ == "__main__":
    train(parse_args())
