"""Configuration dataclasses for RLT (RL Token) training."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class RLTokenModelConfig:
    """Architecture config for the RL token encoder-decoder transformer."""

    # VLA backbone embedding dimension (PaliGemma width).
    vla_embed_dim: int = 2048
    # Output dimension of z_rl.
    rl_token_dim: int = 2048

    # Encoder transformer.
    encoder_layers: int = 4
    encoder_heads: int = 8
    encoder_ff_dim: int = 8192  # 4x vla_embed_dim

    # Decoder transformer.
    decoder_layers: int = 4
    decoder_heads: int = 8
    decoder_ff_dim: int = 8192


@dataclass
class Stage1Config:
    """Stage 1: RL token training on demonstration data."""

    rl_token: RLTokenModelConfig = field(default_factory=RLTokenModelConfig)

    # Whether to jointly fine-tune the VLA alongside RL token training.
    # Set True if starting from a base (non-task-specific) VLA checkpoint.
    # Set False if the VLA is already fine-tuned on the target task.
    joint_vla_finetune: bool = False

    # Weight for VLA fine-tuning loss: L = L_ro + alpha * L_vla.
    vla_loss_weight: float = 0.1

    # Training hyperparameters.
    lr: float = 1e-4
    batch_size: int = 32
    num_train_steps: int = 10_000  # Paper uses 2k-10k.
    warmup_steps: int = 1_000
    save_interval: int = 2_000
    gradient_clip: float = 1.0


@dataclass
class Stage2Config:
    """Stage 2: Online RL with actor-critic on RL token representation."""

    # --- Action chunk ---
    rl_chunk_length: int = 10  # C: RL policy chunk length (< VLA horizon H=50).
    action_dim: int = 16  # d: per-timestep action dim (TSH bimanual Franka = 16).
    stride: int = 2  # Subsampling stride for replay buffer.

    # --- Rollout action smoothing (groundTruthEval-style; toggle with action_smoothing) ---
    # none: one VLA+actor call every rl_chunk_length steps, no overlap smoothing.
    # temporal_ensembling | ema_overlap: re-infer every inference_frequency <= rl_chunk_length steps.
    action_smoothing: str = "none"
    inference_frequency: int = 10  # Steps between VLA+actor calls when smoothing is enabled.
    te_k: float = 0.1  # Temporal ensembling decay (higher = favor newest chunk less).
    ema_alpha: float = 0.75  # New-chunk weight in overlap EMA (ema_overlap mode).

    # --- Actor network ---
    actor_hidden_dim: int = 256
    actor_num_layers: int = 2
    actor_fixed_std: float = 0.03  # Exploration std when stochastic rollout / forward() sampling.
    actor_stochastic_rollout: bool = False  # If True, rollout uses mean + noise; else mean only (less jitter).
    actor_lr: float = 3e-4

    # --- Critic network ---
    critic_hidden_dim: int = 256
    critic_num_layers: int = 2
    critic_lr: float = 3e-4

    # --- RL hyperparameters ---
    discount: float = 0.99
    tau: float = 0.005  # Target network soft update rate.
    bc_reg_weight: float = 5.0  # beta: BC toward VLA reference (mean MSE); higher = tighter anchor.
    ref_action_dropout: float = 0.2  # Fraction of batch with ref zeroed during actor update (generalization).
    utd_ratio: int = 5  # Update-to-data ratio.
    critic_updates_per_actor: int = 2  # Critic updates per actor update.

    # --- Replay buffer ---
    buffer_capacity: int = 100_000
    batch_size: int = 256
    # Warm-up uses the same rollout loop as post-warm-up; align chunk length + smoothing with GTE
    # via train_rlt_stage2.py --match_ground_truth_eval_rollout (see script help).
    warmup_episodes: int = 20  # Episodes of VLA-only rollouts before RL starts.

    # --- Dimensions (derived from VLA + robot) ---
    z_rl_dim: int = 2048
    state_dim: int = 16  # Proprioceptive state dimension.

    # --- Episode settings ---
    max_episode_steps: int = 1800  # Matches Robosuite default.
    num_episodes: int = 500
    eval_interval: int = 50  # Evaluate every N episodes.
    eval_episodes: int = 10
    save_interval: int = 100

    # --- Environment ---
    control_freq: int = 20  # Robosuite: 20 Hz.

    @property
    def action_chunk_dim(self) -> int:
        """Flattened action chunk dimension: C * d."""
        return self.rl_chunk_length * self.action_dim

    def infer_every(self) -> int:
        """Environment steps between VLA+actor forward passes during rollout."""
        if self.action_smoothing == "none":
            return self.rl_chunk_length
        return self.inference_frequency


@dataclass
class RLTConfig:
    """Top-level config for RLT training (both stages)."""

    # VLA config name (registered in openpi training configs).
    vla_config_name: str = "pi05_tsh"
    # Path to VLA checkpoint (PyTorch safetensors).
    vla_checkpoint_path: str = ""
    # Path to Stage 1 RL token encoder checkpoint (for Stage 2).
    rl_token_checkpoint_path: str = ""

    # Stage configs.
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)

    # Common settings.
    seed: int = 42
    device: str = "cuda"
    wandb_enabled: bool = True
    project_name: str = "openpi_rlt"
    exp_name: str = "rlt_experiment"
    checkpoint_base_dir: str = "./checkpoints/rlt"
