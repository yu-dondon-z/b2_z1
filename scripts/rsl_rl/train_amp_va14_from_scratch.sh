#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/dong/b2_z1"
PYTHON_BIN="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"
cd "${PROJECT_DIR}"

exec "${PYTHON_BIN}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-VA14-From-Scratch \
    --num_envs 1024 \
    --headless \
    --teacher-checkpoint "${PROJECT_DIR}/logs/rsl_rl/b2_z1_amp/2026-09-24_09-50-29_va7_whole_body_from_scratch/model_7999.pt" \
    --arm-teacher-weight 5.0 \
    --leg-teacher-weight 0.0 \
    --initial-arm-policy-std 0.15 \
    --max_iterations 8000 \
    --run_name va14_from_scratch \
    agent.resume=false \
    agent.amp_cfg.amp_num_preload_transitions=20000 \
    "$@"
