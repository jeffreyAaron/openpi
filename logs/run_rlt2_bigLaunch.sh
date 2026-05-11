#!/usr/bin/env bash
# Stage 2 RLT (tape handover): task_phase actor on phase 1 + replay window seeding fix
# (warmup VLA trajectories now fill replay for the same RL window — see train_rlt_stage2.py).
# Wandb: watch replay/chunks_added_episode, replay/inserted, train/buffer_size, train/replay_capacity_frac.
#
# Override paths if needed, e.g.:
#   export OPENPI_ROOT=/path/to/openpi
#   export TAPE_OFFSETS_JSON=/path/to/handover_index.json
#   export ROBOSUITE_CONTROLLER_CFG=/path/to/panda_joint_ctrl.json
#   export MIKE_ROOT=/path/to/mike   # optional; prepended to PYTHONPATH when set
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}}"
export CUDA_VISIBLE_DEVICES=1
#export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

PYTHONPATH="${OPENPI_ROOT}/src"
if [[ -n "${MIKE_ROOT:-}" ]]; then
  PYTHONPATH="${MIKE_ROOT}:${MIKE_ROOT}/dependencies/robosuite:${PYTHONPATH}"
fi
export PYTHONPATH

TAPE_OFFSETS_JSON="${TAPE_OFFSETS_JSON:-/home/kavishk/mike/data/reformatted_remove_homing/handover_index.json}"
# Default matches prior cluster layout; override if your robosuite checkout lives elsewhere.
ROBOSUITE_CONTROLLER_CFG="${ROBOSUITE_CONTROLLER_CFG:-/home/jeffreyaj/robosuite/robosuite/environments/custom/configs/panda_joint_ctrl.json}"

VLA_CKPT="${VLA_CKPT:-${OPENPI_ROOT}/checkpoints/pi05_tsh/tsh1_pytorch/40000/40000}"
RL_TOKEN_CKPT="${RL_TOKEN_CKPT:-${OPENPI_ROOT}/checkpoints/rlt/stage1/rlt_stage1_handover_full_run/7650/rl_token.safetensors}"
# warm 20
uv run python scripts/train_rlt_stage2.py \
  --match_ground_truth_eval_rollout \
  --match_ground_truth_eval_mode none \
  --vla_checkpoint_path "${VLA_CKPT}" \
  --rl_token_checkpoint_path "${RL_TOKEN_CKPT}" \
  --robosuite_tape_offsets_json "${TAPE_OFFSETS_JSON}" \
  --robosuite_controller_cfg "${ROBOSUITE_CONTROLLER_CFG}" \
  --robosuite_fixed_layout_index 0 \
  --exp_name rlt2_bigLaunch \
  --vla_prompt "Pick up the yellow tape, hand it over, and place it on the gray tape" \
  --warmup_episodes 50 \
  --num_episodes 500 \
  --batch_size 256 \
  --buffer_capacity 100000 \
  --utd_ratio 5 \
  --critic_updates_per_actor 2 \
  --actor_lr 3e-4 \
  --critic_lr 3e-4 \
  --bc_reg_weight 5.0 \
  --actor_stochastic_rollout \
  --eval_interval 50 \
  --eval_episodes 10 \
  --save_interval 100 \
  --log_video_interval 1 \
  --wandb_enabled \
  --shaped_reward_mode lift_handover \
  --phase_mode task_phase \
  --phase_replay_strategy rl_window_only \
  --actor_task_phases 1
