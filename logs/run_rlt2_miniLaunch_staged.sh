#!/usr/bin/env bash
# Mini Stage-2 run with temp.sh-style hyperparameters (EMA overlap, smaller buffer,
# high BC regularization) plus:
#   --shaped_reward_mode lift_handover  (lift + arm0 opens => phase 3 → VLA handback)
#   --phase_mode task_phase             (VLA until pickup, then RLT through handover)
#   --actor_task_phases 1               (RLT while phase_max==1; see shaped_reward.py)
#
# Override RL token path explicitly:
#   RL_TOKEN_CKPT=/path/to/rl_token.pt ./logs/run_rlt2_miniLaunch_staged.sh
# Otherwise prefers the clip ckpt below, then falls back to bundled local .safetensors.
set -euo pipefail
cd /home/jeffreyaj/openpi

export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH="/home/jeffreyaj/mike:/home/jeffreyaj/mike/dependencies/robosuite:${PWD}/src:."

RL_TOKEN_CLIP="/home/adalal/mike/dependencies/openpi/checkpoints/rlt/stage1/rlt_stage1_handover_clip/9999/rl_token.pt"
RL_TOKEN_LOCAL="/home/jeffreyaj/openpi/checkpoints/rlt/stage1/rlt_stage1_handover_full_run/7650/rl_token.safetensors"
if [[ -n "${RL_TOKEN_CKPT:-}" ]]; then
  :
elif [[ -f "$RL_TOKEN_CLIP" ]]; then
  RL_TOKEN_CKPT="$RL_TOKEN_CLIP"
elif [[ -f "$RL_TOKEN_LOCAL" ]]; then
  echo "[run_rlt2_miniLaunch_staged] Using local RL token: $RL_TOKEN_LOCAL" >&2
  RL_TOKEN_CKPT="$RL_TOKEN_LOCAL"
else
  echo "Set RL_TOKEN_CKPT to a valid rl_token .pt/.safetensors checkpoint." >&2
  exit 1
fi

exec uv run python scripts/train_rlt_stage2.py \
  --env robosuite \
  --action_dim 16 \
  --state_dim 16 \
  --exp_name rlt2_miniLaunch_staged \
  --vla_checkpoint_path /home/jeffreyaj/openpi/checkpoints/pi05_tsh/tsh1_pytorch/40000/40000 \
  --robosuite_controller_cfg /home/jeffreyaj/robosuite/robosuite/environments/custom/configs/panda_joint_ctrl.json \
  --robosuite_tape_offsets_json /home/kavishk/mike/data/reformatted_remove_homing/handover_index.json \
  --vla_prompt "Pick up the yellow tape, hand it over, and place it on the gray tape" \
  --max_episode_steps 512 \
  --num_episodes 24 \
  --warmup_episodes 3 \
  --buffer_capacity 4096 \
  --batch_size 32 \
  --utd_ratio 5 \
  --critic_updates_per_actor 2 \
  --bc_reg_weight 20.0 \
  --ref_action_dropout 0.08 \
  --eval_interval 999 \
  --eval_episodes 1 \
  --save_interval 999 \
  --log_video_interval 1 \
  --video_fps 10 \
  --encoder_ff_dim 4096 \
  --wandb_enabled \
  --shaped_reward_mode lift_handover \
  --phase_mode task_phase \
  --actor_task_phases 1 \
  --phase_replay_strategy rl_window_only
