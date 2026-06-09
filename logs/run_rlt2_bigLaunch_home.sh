#!/usr/bin/env bash
# Stage 2 RLT stability preset + home-return success criterion.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}}"
#export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export CUDA_VISIBLE_DEVICES=3
PYTHONPATH="${OPENPI_ROOT}/src"
if [[ -n "${MIKE_ROOT:-}" ]]; then
  PYTHONPATH="${MIKE_ROOT}:${MIKE_ROOT}/dependencies/robosuite:${PYTHONPATH}"
fi
export PYTHONPATH

TAPE_OFFSETS_JSON="${TAPE_OFFSETS_JSON:-/home/kavishk/mike/data/reformatted_remove_homing/handover_index.json}"
ROBOSUITE_CONTROLLER_CFG="${ROBOSUITE_CONTROLLER_CFG:-/home/jeffreyaj/robosuite/robosuite/environments/custom/configs/panda_joint_ctrl.json}"

VLA_CKPT="${VLA_CKPT:-${OPENPI_ROOT}/checkpoints/pi05_tsh/tsh1_pytorch/40000/40000}"
RL_TOKEN_CKPT="${RL_TOKEN_CKPT:-${OPENPI_ROOT}/checkpoints/rlt/stage1/rlt_stage1_handover_full_run/7650/rl_token.safetensors}"

uv run python scripts/train_rlt_stage2.py \
  --match_ground_truth_eval_rollout \
  --match_ground_truth_eval_mode none \
  --vla_checkpoint_path "${VLA_CKPT}" \
  --rl_token_checkpoint_path "${RL_TOKEN_CKPT}" \
  --robosuite_tape_offsets_json "${TAPE_OFFSETS_JSON}" \
  --robosuite_controller_cfg "${ROBOSUITE_CONTROLLER_CFG}" \
  --robosuite_layout_cycle \
  --exp_name rlt2_bigLaunch_home \
  --vla_prompt "Pick up the yellow tape, hand it over, and place it on the gray tape" \
  --warmup_episodes 50 \
  --num_episodes 500 \
  --batch_size 256 \
  --buffer_capacity 100000 \
  --utd_ratio 3 \
  --critic_updates_per_actor 2 \
  --actor_lr 1e-4 \
  --critic_lr 3e-4 \
  --bc_reg_weight 10.0 \
  --eval_interval 50 \
  --eval_episodes 10 \
  --save_interval 100 \
  --log_video_interval 1 \
  --wandb_enabled \
  --shaped_reward_mode lift_handover \
  --shaped_lift_handover_shortcut_xy_thresh 0.12 \
  --shaped_lift_handover_shortcut_penalty -0.002 \
  --shaped_lift_handover_gate_success_bonus \
  --shaped_lift_handover_shortcut_success_bonus_frac 0.15 \
  --shaped_lift_handover_require_grasp1_for_task_success \
  --shaped_require_home_for_task_success \
  --shaped_home_joint_tolerance 0.20 \
  --shaped_home_success_bonus 0.50 \
  --phase_mode task_phase \
  --phase_replay_strategy rl_window_only \
  --actor_task_phases 1
