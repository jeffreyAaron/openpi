#!/usr/bin/env bash
# Launched from tmux session 2, window 3 (see scripts/temp.sh).
set -euo pipefail
cd /home/jeffreyaj/openpi
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=/home/jeffreyaj/mike:/home/jeffreyaj/mike/dependencies/robosuite:src:.
uv run python scripts/train_rlt_stage2.py \
  --env robosuite --action_dim 16 --state_dim 16 \
  --vla_checkpoint_path /home/jeffreyaj/openpi/checkpoints/pi05_tsh/tsh1_pytorch/40000/40000 \
  --rl_token_checkpoint_path /home/jeffreyaj/openpi/checkpoints/rlt/stage1/rlt_stage1_handover_full_run/7650/rl_token.safetensors \
  --robosuite_tape_offsets_json /home/kavishk/mike/data/reformatted_remove_homing/handover_index.json \
  --exp_name rlt2_finetune_from_warmstart_full_run \
  --warmup_episodes 20 --num_episodes 500 --max_episode_steps 1800 \
  --batch_size 256 --buffer_capacity 100000 \
  --utd_ratio 5 --critic_updates_per_actor 2 \
  --bc_reg_weight 5.0 \
  --eval_interval 50 --eval_episodes 10 --save_interval 100 \
  --log_video_interval 1 \
  --wandb_enabled
