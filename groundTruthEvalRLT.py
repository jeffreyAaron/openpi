"""
Ground-truth eval for the **full RLT pipeline** (frozen VLA + frozen RL token
encoder + trained Gaussian residual actor).

This script reuses the entire parallel-worker / coordinator stack from
``groundTruthEval.py`` (sim_worker, run_eval_batch, TemporalEnsembler,
format_proprio, gripper-zero-hold, sanitize_bimanual_panda_action, JSON
position-noise sampling, video saving) and only adds:

    * ``RLTPolicyWrapper`` — combines:
        - the frozen π₀.₅ VLA (loaded the same way as ``Pi05PolicyWrapper``,
          via ``openpi.policies.policy_config.create_trained_policy``)
        - the frozen ``RLTokenEncoder`` (Stage 1 checkpoint)
        - the trained ``GaussianActor``  (Stage 2 checkpoint dir
          containing ``actor.safetensors``)

      For each observation it: (a) infers the VLA reference chunk via
      the trained policy (full input transform, normalize, tokenize,
      flow-matching, unnormalize), (b) extracts ``z_rl`` via
      ``VLAEmbeddingExtractor`` + encoder, (c) runs the actor's
      deterministic mean ``μ = ref + δ_max·tanh(MLP([z_rl, s, ref]))``
      sliced/reshaped to ``(rl_chunk_length, action_dim)``.

    * ``RLTEvalConfig`` — extends ``EvalConfig`` with RLT-specific paths
      and architecture knobs (must match Stage 1 / Stage 2 training).

    * ``run_evaluation_rlt`` / ``main`` — entry point with sensible defaults.

Why this is a separate script from ``groundTruthEval.py``:
    The base GTE has only ``policy_type ∈ {"act", "pi05"}`` and is loaded
    in many places (logs/run_full.sh, plot scripts, etc.). Adding a third
    policy type and the RLT model dependencies inline would couple GTE to
    ``openpi.rlt.*`` whether or not RLT is being evaluated. Keeping the
    RLT path in its own file mirrors the "RLT lives under
    ``openpi.rlt``" boundary already used by ``train_rlt_stage2.py``.

Usage example (matches ``logs/run_rlt2_finetune_from_warmstart.sh`` defaults:
``rl_chunk_length=10``, no smoothing, ``encoder_ff_dim=8192``)::

    python groundTruthEvalRLT.py  # edit fields in main() to taste

To match a Stage 1 checkpoint trained with ``--encoder_ff_dim 4096``
(see ``logs/run_full.sh``), set ``encoder_ff_dim=4096`` in
``RLTEvalConfig``. A mismatch will fail strict ``load_state_dict``.

Important alignment notes vs ``groundTruthEval.py`` defaults:
    - GTE's pi05 main() uses ``inference_frequency=20`` with
      ``use_temporal_ensembling=True``. RLT defaults below mirror **how
      the actor was trained** (``rl_chunk_length=10``, no smoothing,
      ``inference_frequency=10``) so the rollout cadence matches the
      replay buffer the actor optimized against. If you set
      ``use_temporal_ensembling=True``, the actor will see overlapping
      chunks even though it never trained against them — interpret
      results accordingly.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# Reuse everything from the base GTE: sim worker, coordinator, smoothing, IO.
# (groundTruthEval.py lives at the same workspace root as this file.)
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from groundTruthEval import (  # noqa: E402  (intentional after sys.path insert)
    BasePolicyWrapper,
    EvalConfig,
    EvalResult,
    SimConfig,
    get_visible_gpu_ids,
    run_eval_batch,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RLTEvalConfig(EvalConfig):
    """GTE config + RLT-specific paths and architecture knobs.

    All RLT-specific fields default to the values used in the user's most
    recent stage-2 training run (``logs/run_rlt2_finetune_from_warmstart.sh``).
    The architecture params **must match** the Stage 1 encoder and the Stage 2
    actor that was trained — strict ``load_state_dict`` will raise on any
    hidden-dim / chunk-length / encoder-ff mismatch.
    """

    # Override default (parent has policy_type="act").
    policy_type: str = "rlt"

    # --- Frozen VLA + Stage 1 encoder + Stage 2 actor checkpoints ---
    # Note: ``checkpoint_path`` (inherited from EvalConfig) is the path used
    # for the eval output dir name. We treat it as the ACTOR checkpoint dir
    # (the most "specific" of the three), so the output dir reflects which
    # actor was evaluated. The VLA / encoder paths are set explicitly below.
    vla_config_name: str = "pi05_tsh"
    vla_checkpoint_path: str = ""           # Frozen π₀.₅ VLA checkpoint
    rl_token_checkpoint_path: str = ""      # Stage 1 .safetensors / .pt
    actor_checkpoint_path: str = ""         # Stage 2 ckpt dir (has actor.safetensors)
    vla_prompt: str = "Pick up the yellow tape, hand it over, and place it on the gray tape"
    rlt_device: str = "cuda"
    # Mirror train_rlt_stage2.py --vla_train_augmentation_at_infer (default off
    # there too): keep RandomImageAugmentation / GaussianActionNoise out of the
    # eval pipeline by default so sim observations match those that drove RL updates.
    vla_train_augmentation_at_infer: bool = False

    # --- RL chunk / actor architecture (must match Stage 2 training) ---
    rl_chunk_length: int = 10
    action_dim: int = 16
    state_dim: int = 16
    actor_hidden_dim: int = 256
    actor_num_layers: int = 2
    actor_fixed_std: float = 0.03
    actor_delta_max: float = 1.0
    # Forward the actor's mean μ deterministically (same as how `evaluate()` in
    # train_rlt_stage2.py runs the actor at eval time). Set True only if you
    # want to inject the training-time exploration noise at eval.
    actor_stochastic_rollout: bool = False

    # --- RL token encoder architecture (must match Stage 1) ---
    z_rl_dim: int = 2048
    encoder_layers: int = 4
    encoder_heads: int = 8
    encoder_ff_dim: int = 8192


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------


class RLTPolicyWrapper(BasePolicyWrapper):
    """Frozen VLA + frozen RL token encoder + trained Gaussian residual actor.

    Per observation:
      1. ``trained_policy.infer(pi_obs)`` → VLA reference action chunk
         (already through TSHInputs → Normalize → tokenize → flow matching →
         Unnormalize, exactly the same path as ``Pi05PolicyWrapper``).
      2. ``obs_dict_to_vla_observation(...)`` → an ``Observation`` with
         the same input transforms applied (so tokenization sees the
         normalized state in [-1, 1]).
      3. ``VLAEmbeddingExtractor.extract_image_embeddings`` → image-token
         embeddings from the prefix encoder.
      4. ``RLTokenEncoder(img_embs, img_mask)`` → ``z_rl ∈ ℝ^{z_rl_dim}``.
      5. ``GaussianActor.forward_mean(z_rl, state, ref_flat)`` → action
         mean ``μ = ref + δ_max · tanh(MLP([z_rl, s, ref]))``,
         shape ``(1, C·d)``; reshape to ``(C, d)``.

    The wrapper exposes ``action_chunk_size = rl_chunk_length`` so GTE's
    coordinator and sim worker treat the actor's chunk as the unit of
    inference (just like Pi05's full 50-step horizon was for that wrapper).
    """

    def __init__(self, cfg: RLTEvalConfig) -> None:
        # Same path manipulation as Pi05PolicyWrapper for environments where
        # openpi is shipped as a vendored submodule.  Harmless when openpi is
        # already importable from the active environment's site-packages.
        repo_root = _THIS_DIR
        for sub in ("dependencies/openpi/src",
                    "dependencies/openpi/packages/openpi-client/src"):
            p = str(repo_root / sub)
            if Path(p).is_dir() and p not in sys.path:
                sys.path.insert(0, p)

        # Local imports so the (heavy) RLT stack is only loaded when this
        # wrapper is actually instantiated.
        import openpi.models.pi0_config
        from openpi.policies import policy_config as openpi_policy_config
        from openpi.training import config as _config
        from openpi.rlt.config import RLTokenModelConfig
        from openpi.rlt.rl_token import RLTokenEncoder
        from openpi.rlt.actor_critic import GaussianActor
        from openpi.rlt.vla_wrapper import VLAEmbeddingExtractor

        device = torch.device(cfg.rlt_device if torch.cuda.is_available() else "cpu")

        # ---- Frozen VLA -----------------------------------------------------
        train_config = _config.get_config(cfg.vla_config_name)
        # Force bfloat16 to match train_rlt_stage2 (otherwise OOM with default).
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

        self._vla_action_horizon: int = int(model_cfg.action_horizon)
        self._action_dim: int = cfg.action_dim
        self._rl_chunk_length: int = cfg.rl_chunk_length
        self._device = device

        if cfg.rl_chunk_length > self._vla_action_horizon:
            raise ValueError(
                f"rl_chunk_length ({cfg.rl_chunk_length}) > VLA action_horizon "
                f"({self._vla_action_horizon}); the actor's chunk cannot exceed "
                f"the VLA's reference chunk length."
            )

        self.trained_policy = openpi_policy_config.create_trained_policy(
            train_config,
            cfg.vla_checkpoint_path,
            default_prompt=cfg.vla_prompt,
            pytorch_device=str(device),
            skip_train_data_input_transforms=not cfg.vla_train_augmentation_at_infer,
        )
        vla_model = self.trained_policy._model  # noqa: SLF001
        for p in vla_model.parameters():
            p.requires_grad_(False)
        vla_model.eval()
        print(
            f"[RLTPolicyWrapper] frozen VLA loaded from {cfg.vla_checkpoint_path} "
            f"(action_horizon={self._vla_action_horizon}, train_aug_at_infer="
            f"{cfg.vla_train_augmentation_at_infer})"
        )

        # ---- Frozen RL token encoder (Stage 1) -----------------------------
        rl_token_config = RLTokenModelConfig(
            vla_embed_dim=cfg.z_rl_dim,
            rl_token_dim=cfg.z_rl_dim,
            encoder_layers=cfg.encoder_layers,
            encoder_heads=cfg.encoder_heads,
            encoder_ff_dim=cfg.encoder_ff_dim,
        )
        self.encoder = RLTokenEncoder(rl_token_config).to(device)
        enc_state = _load_rl_token_encoder_state_dict(
            cfg.rl_token_checkpoint_path, device,
        )
        missing, unexpected = self.encoder.load_state_dict(enc_state, strict=False)
        if missing:
            print(
                f"[RLTPolicyWrapper] WARNING — encoder missing {len(missing)} keys "
                f"after load (showing up to 5): {missing[:5]}"
            )
        if unexpected:
            print(
                f"[RLTPolicyWrapper] WARNING — encoder got {len(unexpected)} unexpected "
                f"keys after load (showing up to 5): {unexpected[:5]}"
            )
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        print(
            f"[RLTPolicyWrapper] RL token encoder loaded from {cfg.rl_token_checkpoint_path}"
        )

        self.extractor = VLAEmbeddingExtractor(vla_model, device)

        # ---- Trained actor (Stage 2) ---------------------------------------
        action_chunk_dim = cfg.rl_chunk_length * cfg.action_dim
        self.actor = GaussianActor(
            z_rl_dim=cfg.z_rl_dim,
            state_dim=cfg.state_dim,
            action_chunk_dim=action_chunk_dim,
            hidden_dim=cfg.actor_hidden_dim,
            num_layers=cfg.actor_num_layers,
            fixed_std=cfg.actor_fixed_std,
            delta_max=cfg.actor_delta_max,
        ).to(device)
        actor_state = _load_actor_state_dict(cfg.actor_checkpoint_path, device)
        # strict=True so a mismatch with Stage 2 architecture fails loudly.
        self.actor.load_state_dict(actor_state, strict=True)
        for p in self.actor.parameters():
            p.requires_grad_(False)
        self.actor.eval()
        print(
            f"[RLTPolicyWrapper] actor loaded from {cfg.actor_checkpoint_path} "
            f"(rl_chunk_length={cfg.rl_chunk_length}, hidden_dim={cfg.actor_hidden_dim}, "
            f"delta_max={cfg.actor_delta_max}, fixed_std={cfg.actor_fixed_std})"
        )

        self._cfg = cfg
        # Lazily-imported helpers so we don't pull jax / openpi at module-load
        # time on the GTE side.
        from scripts.train_rlt_stage2 import (
            build_tsh_pi_obs,
            obs_dict_to_vla_observation,
        )
        self._build_tsh_pi_obs = build_tsh_pi_obs
        self._obs_dict_to_vla_observation = obs_dict_to_vla_observation

    @property
    def action_chunk_size(self) -> int:
        return self._rl_chunk_length

    def infer_batch(self, obs_list: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Run RLT inference per observation; return ``(C, d)`` chunks."""
        results: List[np.ndarray] = []
        C, d = self._rl_chunk_length, self._action_dim
        H = self._vla_action_horizon
        device = self._device

        for obs in obs_list:
            # Same dict layout as Pi05PolicyWrapper / sim_worker.
            pi_obs_dict = {
                "state": obs["raw_proprio"].astype(np.float32),
                "exo_image": obs["exo_image"],
                "wrist_left_image": obs["wrist_left_image"],
                "wrist_right_image": obs["wrist_right_image"],
            }

            # 1. VLA reference chunk (denormalized, [H, d]).
            pi_obs = self._build_tsh_pi_obs(pi_obs_dict, H, d)
            ref_full = self.trained_policy.infer(pi_obs)["actions"].astype(np.float32)
            ref_chunk_np = ref_full[:C, :d]

            # 2. Build Observation for embedding extraction (same input
            #    transforms as the policy applies internally).
            observation = self._obs_dict_to_vla_observation(
                self.trained_policy, pi_obs_dict, device, H, d,
            )

            # 3. z_rl = encoder(prefix-encoder image embeddings).
            with torch.no_grad():
                z_rl = self.extractor.extract_rl_token(observation, self.encoder)  # [1, z_rl_dim]

                ref_t = torch.from_numpy(ref_chunk_np).float().to(device)  # [C, d]
                ref_flat = ref_t.reshape(1, -1)                            # [1, C*d]
                state_t = torch.from_numpy(
                    pi_obs_dict["state"]
                ).float().unsqueeze(0).to(device)                          # [1, state_dim]

                # 4. Actor: deterministic mean by default (matches eval path
                #    in train_rlt_stage2.evaluate()).
                if self._cfg.actor_stochastic_rollout:
                    sampled, _ = self.actor(z_rl, state_t, ref_flat)
                    chunk = sampled[0].cpu().numpy().reshape(C, d)
                else:
                    mean = self.actor.forward_mean(z_rl, state_t, ref_flat)
                    chunk = mean[0].cpu().numpy().reshape(C, d)

            results.append(chunk.astype(np.float32))
        return results


# ---------------------------------------------------------------------------
# Checkpoint loaders
# ---------------------------------------------------------------------------


def _load_rl_token_encoder_state_dict(
    ckpt_path: str, device: torch.device,
) -> Dict[str, Any]:
    """Load a state dict for `RLTokenEncoder` from Stage 1 RL-token checkpoints.

    Same behavior as ``train_rlt_stage2._load_rl_token_encoder_state_dict``:
    supports ``.safetensors`` and ``.pt``, strips a leading ``encoder.`` prefix
    when the file contains a full ``RLTokenModule`` checkpoint.
    """
    import safetensors.torch

    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"RL token encoder checkpoint not found: {path}")
    if path.suffix.lower() == ".safetensors":
        raw: Dict[str, Any] = safetensors.torch.load_file(str(path), device=str(device))
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


def _load_actor_state_dict(actor_path: str, device: torch.device) -> Dict[str, Any]:
    """Load actor weights from a Stage 2 checkpoint.

    Accepts either:
      * a directory containing ``actor.safetensors`` (the layout produced by
        ``save_stage2_checkpoint`` and ``pretrain_rlt_actor_critic.py``), or
      * the ``actor.safetensors`` file directly.
    """
    import safetensors.torch

    p = Path(actor_path)
    if p.is_dir():
        actor_file = p / "actor.safetensors"
    elif p.is_file():
        actor_file = p
    else:
        raise FileNotFoundError(f"actor checkpoint not found: {actor_path}")
    if not actor_file.is_file():
        raise FileNotFoundError(
            f"actor.safetensors not found under {actor_path} (looked at {actor_file})"
        )
    return safetensors.torch.load_file(str(actor_file), device=str(device))


# ---------------------------------------------------------------------------
# Entry point (mirrors groundTruthEval.run_evaluation, but for RLT)
# ---------------------------------------------------------------------------


def run_evaluation_rlt(eval_config: RLTEvalConfig) -> Path:
    """Run the full GTE batch with the RLT actor; return the output path.

    Parallels ``groundTruthEval.run_evaluation`` step for step:
      * Resolve visible GPUs.
      * Build the output dir from a stem of the *actor* checkpoint
        (so each Stage 2 checkpoint's eval lives in its own folder).
      * Save the eval config JSON.
      * Build the policy.
      * Validate ``inference_frequency <= action_chunk_size`` (here =
        ``rl_chunk_length``).
      * Sample tape positions with the same Gaussian-noise scheme
        constrained to the per-tape min/max range.
      * Run sims in parallel batches.
      * Aggregate and write ``summary.json``.
    """
    visible_gpu_ids = get_visible_gpu_ids()

    # Make the output dir name reflect the actor checkpoint (the variable
    # one across runs): treat ``actor_checkpoint_path`` as the identifying
    # path. Falls back to ``checkpoint_path`` (inherited) for compatibility.
    src_for_stem = eval_config.actor_checkpoint_path or eval_config.checkpoint_path
    src_path = Path(src_for_stem) if src_for_stem else Path("rlt_unspecified")
    if src_path.is_dir():
        # Stage-2 ckpts look like ``.../<exp>/<episode>/actor.safetensors``.
        # Use ``<exp>_<episode>`` as the stem so the output dir is descriptive.
        ckpt_stem = f"{src_path.parent.name}_{src_path.name}"
    else:
        ckpt_stem = src_path.stem or src_path.parent.name

    output_path = Path(eval_config.output_dir) / f"rlt_{ckpt_stem}_{int(time.time())}"
    output_path.mkdir(parents=True, exist_ok=True)

    if eval_config.gripper_action_log is not None:
        eval_config = dataclasses.replace(
            eval_config,
            gripper_action_log=str(Path(eval_config.gripper_action_log).resolve()),
        )

    config_dict = dataclasses.asdict(eval_config)
    with open(output_path / "eval_config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Eval config saved to {output_path / 'eval_config.json'}")

    print(
        f"Loading RLT policy:\n"
        f"  VLA            : {eval_config.vla_checkpoint_path}\n"
        f"  RL-token enc   : {eval_config.rl_token_checkpoint_path}\n"
        f"  Actor (stage2) : {eval_config.actor_checkpoint_path}\n"
    )
    policy = RLTPolicyWrapper(eval_config)
    print(f"Policy loaded. action_chunk_size = rl_chunk_length = {policy.action_chunk_size}")

    if eval_config.inference_frequency > policy.action_chunk_size:
        raise AssertionError(
            f"inference_frequency ({eval_config.inference_frequency}) must be "
            f"<= action_chunk_size ({policy.action_chunk_size}). For matching the "
            "training rollout, set inference_frequency == rl_chunk_length and "
            "use_temporal_ensembling=False."
        )

    with open(eval_config.offsets_json_path) as f:
        offsets_data = json.load(f)
    if eval_config.max_sims is not None:
        offsets_data = offsets_data[: eval_config.max_sims]

    noise_rng = np.random.default_rng(eval_config.noise_seed)
    yellow_x_range = (min(e["yellow_x"] for e in offsets_data),
                      max(e["yellow_x"] for e in offsets_data))
    yellow_y_range = (min(e["yellow_y"] for e in offsets_data),
                      max(e["yellow_y"] for e in offsets_data))
    duct_x_range = (min(e["duct_x"] for e in offsets_data),
                    max(e["duct_x"] for e in offsets_data))
    duct_y_range = (min(e["duct_y"] for e in offsets_data),
                    max(e["duct_y"] for e in offsets_data))

    sim_configs: List[SimConfig] = []
    for i, entry in enumerate(offsets_data):
        for _ in range(1000):
            yellow_x = entry["yellow_x"] + noise_rng.normal(0, eval_config.position_noise_std)
            yellow_y = entry["yellow_y"] + noise_rng.normal(0, eval_config.position_noise_std)
            duct_x = entry["duct_x"] + noise_rng.normal(0, eval_config.position_noise_std)
            duct_y = entry["duct_y"] + noise_rng.normal(0, eval_config.position_noise_std)
            yellow_x = float(np.clip(yellow_x, *yellow_x_range))
            yellow_y = float(np.clip(yellow_y, *yellow_y_range))
            duct_x = float(np.clip(duct_x, *duct_x_range))
            duct_y = float(np.clip(duct_y, *duct_y_range))
            dist = np.sqrt((yellow_x - duct_x) ** 2 + (yellow_y - duct_y) ** 2)
            if dist >= eval_config.min_tape_distance:
                break
        else:
            print(
                f"[WARNING] sim {i}: could not find valid placement after 1000 attempts, "
                f"using last sample (dist={dist:.4f}m)"
            )
        sim_configs.append(SimConfig(
            local_sim_id=i % eval_config.max_parallel_envs,
            global_sim_id=i,
            yellow_tape_offset=np.array([yellow_x, yellow_y, 0.0]),
            duct_tape_offset=np.array([duct_x, duct_y, 0.0]),
        ))

    all_results: List[EvalResult] = []
    start_time = time.time()
    for batch_start in range(0, len(sim_configs), eval_config.max_parallel_envs):
        batch = sim_configs[batch_start : batch_start + eval_config.max_parallel_envs]
        print(f"\nStarting sims {batch[0].global_sim_id} – {batch[-1].global_sim_id} ...")
        batch_results = run_eval_batch(batch, policy, eval_config, output_path, visible_gpu_ids)
        all_results.extend(batch_results)

    elapsed = time.time() - start_time
    all_results.sort(key=lambda r: r.global_sim_id)

    print("\n" + "=" * 72)
    print("RLT EVALUATION RESULTS")
    print(f"Total time: {elapsed:.1f}s")
    print("=" * 72)
    for r in all_results:
        print(f"  Sim {r.global_sim_id:3d}: success={r.success}  steps={r.num_steps}")

    num_success = sum(r.success for r in all_results)
    success_rate = num_success / len(all_results) if all_results else 0.0
    summary = {
        "summary": {
            "total_sims": len(all_results),
            "successful": num_success,
            "success_rate": success_rate,
            "elapsed_seconds": elapsed,
            "vla_checkpoint_path": str(eval_config.vla_checkpoint_path),
            "rl_token_checkpoint_path": str(eval_config.rl_token_checkpoint_path),
            "actor_checkpoint_path": str(eval_config.actor_checkpoint_path),
            "policy_type": eval_config.policy_type,
            "rl_chunk_length": eval_config.rl_chunk_length,
            "action_smoothing": (
                "temporal_ensembling"
                if eval_config.use_temporal_ensembling
                else "none_or_ema"
            ),
            "inference_frequency": eval_config.inference_frequency,
        },
        "individual_results": [r.to_dict() for r in all_results],
    }
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSuccess rate: {success_rate:.1%}  ({num_success}/{len(all_results)})")
    print(f"Results saved to {output_path}")
    return output_path


def main() -> None:
    # ---- Edit these fields to configure a run ----
    # Defaults below match logs/run_rlt2_finetune_from_warmstart.sh:
    #   rl_chunk_length=10, no smoothing, encoder_ff_dim=8192 (RLTokenModelConfig default).
    # The actor was trained with phase_mode="off", so the actor drives the
    # whole post-warmup episode — same setup we replicate here.
    eval_config = RLTEvalConfig(
        # ---- Three checkpoints required for the full pipeline ----
        vla_checkpoint_path=(
            "/home/jeffreyaj/openpi/checkpoints/pi05_tsh/tsh1_pytorch/40000/40000"
        ),
        rl_token_checkpoint_path=(
            "/home/jeffreyaj/openpi/checkpoints/rlt/stage1/"
            "rlt_stage1_handover_full_run/7650/rl_token.safetensors"
        ),
        actor_checkpoint_path=(
            # Stage 2 checkpoint dir (must contain actor.safetensors).
            # Switch to the rlt2_finetune_from_warmstart_full_run dir once
            # an episode checkpoint exists there.
            "/home/jeffreyaj/openpi/checkpoints/rlt/stage2/"
            "rlt2_bc0p5_drop0p8_fresh_chunk50/199"
        ),
        vla_config_name="pi05_tsh",
        vla_prompt="Pick up the yellow tape, hand it over, and place it on the gray tape",

        # ---- Match Stage 2 training architecture / chunk sizing ----
        rl_chunk_length=10,         # Default in train_rlt_stage2.py
        action_dim=16,
        state_dim=16,
        actor_hidden_dim=256,
        actor_num_layers=2,
        actor_fixed_std=0.03,
        actor_delta_max=1.0,
        actor_stochastic_rollout=False,  # Eval with deterministic mean.
        z_rl_dim=2048,
        encoder_layers=4,
        encoder_heads=8,
        encoder_ff_dim=8192,        # Stage 1 default; use 4096 if your stage1
                                    # was trained with --encoder_ff_dim 4096.

        # ---- Rollout cadence (no smoothing; match training default) ----
        inference_frequency=10,     # == rl_chunk_length
        use_temporal_ensembling=False,

        # ---- Sim physics + IO (same as groundTruthEval.main pi05 defaults) ----
        max_steps=1800,
        max_parallel_envs=16,
        contact_solref=[0.01, 1.0],
        contact_solimp=[0.95, 0.99, 0.001],
        save_videos=True,
        controller_cfg=(
            "/home/jeffreyaj/robosuite/robosuite/environments/custom/configs/"
            "panda_joint_ctrl.json"
        ),
        position_noise_std=0.05,
    )

    if not eval_config.vla_checkpoint_path:
        raise ValueError("Set vla_checkpoint_path before running.")
    if not eval_config.rl_token_checkpoint_path:
        raise ValueError("Set rl_token_checkpoint_path before running.")
    if not eval_config.actor_checkpoint_path:
        raise ValueError("Set actor_checkpoint_path before running.")

    run_evaluation_rlt(eval_config)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # Required for CUDA + multiprocessing
    main()
