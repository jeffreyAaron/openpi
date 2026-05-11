#!/usr/bin/env bash
# Smoke launcher for the PyRoki handover oracle (replaces the RLT actor during
# the handover phase). VLA still drives pickup + placement; the macro from
# `mike/dependencies/robosuite/test_scripts/handover_step.py` runs the moment
# `lift_handover` reports phase_max == 1, then control hands back to the VLA.
#
# Mirrors run_rlt2_miniLaunch_staged.sh but:
#   --handover_controller pyroki  (oracle replaces actor at phase 1)
#   --warmup_episodes 0           (no warmup; oracle is fixed)
#   --num_episodes 5              (short smoke run)
#   --eval_interval 5             (eval at the very end)
#
# Override RL token path explicitly:
#   RL_TOKEN_CKPT=/path/to/rl_token.pt ENCODER_FF_DIM=4096 ./logs/run_pyroki_handover.sh
# Otherwise prefers RL_TOKEN_CLIP (4096 FFN) below, then RL_TOKEN_LOCAL (8192 FFN).
# Set ENCODER_FF_DIM in the environment if your checkpoint does not match these defaults.
#
# Note: the actor + critic are still constructed (so we exercise the same code
# path as the RLT branch), but they're never updated — the validator at
# startup force-disables the async learner and the rollout suppresses replay
# inserts when the oracle drives.
set -euo pipefail
cd /home/jeffreyaj/openpi

export CUDA_VISIBLE_DEVICES=7
# `mike/dependencies/{robosuite,pyroki/src}` are appended internally by
# openpi.rlt.pyroki_handover, but exposing them on PYTHONPATH up front keeps
# `import pyroki` and `from api import ...` resolvable to subprocesses too
# (e.g. JAX warm-compile spawns a fresh interpreter).
export PYTHONPATH="/home/jeffreyaj/mike:/home/jeffreyaj/mike/dependencies/robosuite:/home/jeffreyaj/mike/dependencies/pyroki/src:${PWD}/src:."

RL_TOKEN_CLIP="/home/adalal/mike/dependencies/openpi/checkpoints/rlt/stage1/rlt_stage1_handover_clip/9999/rl_token.pt"
RL_TOKEN_LOCAL="/home/jeffreyaj/openpi/checkpoints/rlt/stage1/rlt_stage1_handover_full_run/7650/rl_token.safetensors"
if [[ -n "${RL_TOKEN_CKPT:-}" ]]; then
  encoder_ff_dim="${ENCODER_FF_DIM:-8192}"
elif [[ -f "$RL_TOKEN_CLIP" ]]; then
  RL_TOKEN_CKPT="$RL_TOKEN_CLIP"
  encoder_ff_dim="${ENCODER_FF_DIM:-4096}"
elif [[ -f "$RL_TOKEN_LOCAL" ]]; then
  echo "[run_pyroki_handover] Using local RL token: $RL_TOKEN_LOCAL" >&2
  RL_TOKEN_CKPT="$RL_TOKEN_LOCAL"
  encoder_ff_dim="${ENCODER_FF_DIM:-8192}"
else
  echo "Set RL_TOKEN_CKPT to a valid rl_token .pt/.safetensors checkpoint." >&2
  exit 1
fi

echo "[run_pyroki_handover] rl_token=$RL_TOKEN_CKPT encoder_ff_dim=$encoder_ff_dim" >&2
uv run python scripts/train_rlt_stage2.py \
  --env robosuite \
  --action_dim 16 \
  --state_dim 16 \
  --exp_name pyroki_handover_smoke \
  --vla_checkpoint_path /home/jeffreyaj/openpi/checkpoints/pi05_tsh/tsh1_pytorch/40000/40000 \
  --rl_token_checkpoint_path "$RL_TOKEN_CKPT" \
  --robosuite_controller_cfg /home/jeffreyaj/robosuite/robosuite/environments/custom/configs/panda_joint_ctrl.json \
  --robosuite_tape_offsets_json /home/kavishk/mike/data/reformatted_remove_homing/handover_index.json \
  --vla_prompt "Pick up the yellow tape, hand it over, and place it on the gray tape" \
  --max_episode_steps 1800 \
  --num_episodes 5 \
  --warmup_episodes 0 \
  --buffer_capacity 1024 \
  --batch_size 32 \
  --utd_ratio 5 \
  --critic_updates_per_actor 2 \
  --bc_reg_weight 20.0 \
  --ref_action_dropout 0.08 \
  --eval_interval 5 \
  --eval_episodes 1 \
  --save_interval 999 \
  --log_video_interval 1 \
  --video_fps 10 \
  --encoder_ff_dim "${encoder_ff_dim}" \
  --wandb_enabled \
  --shaped_reward_mode lift_handover \
  --phase_mode task_phase \
  --actor_task_phases 1 \
  --phase_replay_strategy rl_window_only \
  --handover_controller pyroki \
  --pyroki_x_shift 0.0 \
  --pyroki_y_shift 0.0 \
  --pyroki_angle_shift 0.0 \
  --pyroki_perturb_radius 0.0
