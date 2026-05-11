#!/usr/bin/env bash
set -euo pipefail
cd /home/jeffreyaj/openpi

export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH="/home/jeffreyaj/mike:/home/jeffreyaj/mike/dependencies/robosuite:${PWD}/src:."

RL_TOKEN_CKPT="${RL_TOKEN_CKPT:-/home/adalal/mike/dependencies/openpi/checkpoints/rlt/stage1/rlt_stage1_handover_clip/9999/rl_token.pt}"

uv run python scripts/train_rlt_stage2.py \
  --env robosuite \
  --action_dim 16 --state_dim 16 \
  --exp_name rlt2_full_dropterm_handover_debug_new_reward \
  --vla_checkpoint_path /home/jeffreyaj/openpi/checkpoints/pi05_tsh/tsh1_pytorch/40000/40000 \
  --rl_token_checkpoint_path "${RL_TOKEN_CKPT}" \
  --robosuite_controller_cfg /home/jeffreyaj/robosuite/robosuite/environments/custom/configs/panda_joint_ctrl.json \
  --robosuite_tape_offsets_json /home/kavishk/mike/data/reformatted_remove_homing/handover_index.json \
  --vla_prompt "Pick up the yellow tape, hand it over, and place it on the gray tape" \
  --warmup_episodes 20 \
  --num_episodes 500 \
  --max_episode_steps 1800 \
  --buffer_capacity 100000 \
  --batch_size 256 \
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
  --encoder_ff_dim 4096 \
  --wandb_enabled \
  --shaped_reward_mode lift_handover \
  --phase_mode task_phase \
  --actor_task_phases 1 2 \
  --phase_replay_strategy rl_window_only \
  --replay_only_successful \
  --end_actor_after_handover