#!/usr/bin/env bash
set -euo pipefail
cd /home/jeffreyaj/openpi
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/jeffreyaj/mike:/home/jeffreyaj/mike/dependencies/robosuite:/home/jeffreyaj/openpi/src:.

exec uv run python scripts/train_rlt_stage2.py \
  --match_ground_truth_eval_rollout \
  --match_ground_truth_eval_mode none \
  --vla_checkpoint_path /home/jeffreyaj/openpi/checkpoints/pi05_tsh/tsh1_pytorch/40000/40000 \
  --rl_token_checkpoint_path /home/jeffreyaj/openpi/checkpoints/rlt/stage1/rlt_stage1_handover_full_run/7650/rl_token.safetensors \
  --robosuite_tape_offsets_json /home/kavishk/mike/data/reformatted_remove_homing/handover_index.json \
  --robosuite_controller_cfg /home/jeffreyaj/robosuite/robosuite/environments/custom/configs/panda_joint_ctrl.json \
  --robosuite_fixed_layout_index 0 \
  --exp_name rlt2_bigLaunch \
  --vla_prompt "Pick up the yellow tape, hand it over, and place it on the gray tape" \
  --warmup_episodes 20 \
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
  --wandb_enabled
