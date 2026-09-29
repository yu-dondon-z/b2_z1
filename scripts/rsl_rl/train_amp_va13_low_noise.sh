#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/dong/b2_z1"
PYTHON_BIN="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"
TRAIN_ITERATIONS="${B2Z1_TRAIN_ITERATIONS:-300}"

cd "${PROJECT_DIR}"

# Restart from VA7 rather than continuing the unsafe VA10--VA12 pilots.
exec "${PYTHON_BIN}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-VA13-Low-Noise \
    --num_envs 1024 \
    --headless \
    --resume \
    --load_run 2026-09-24_09-50-29_va7_whole_body_from_scratch \
    --checkpoint model_7999.pt \
    --reset_optimizer \
    --reset_leg_policy_std 0.18 \
    --arm-teacher-weight 5.0 \
    --leg-teacher-weight 0.6 \
    --leg-teacher-yaw-release 0.10 \
    --max_iterations "${TRAIN_ITERATIONS}" \
    --run_name va13_low_noise_from7999 \
    agent.amp_cfg.amp_num_preload_transitions=20000 \
    agent.save_interval=25 \
    "$@"
