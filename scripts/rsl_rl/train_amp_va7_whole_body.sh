#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/dong/b2_z1"
PYTHON_BIN="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"

cd "${PROJECT_DIR}"

# New 18-DoF policy from scratch; VA3/VA4 checkpoints remain untouched.
exec "${PYTHON_BIN}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-VA7-Whole-Body \
    --num_envs 1024 \
    --headless \
    --max_iterations 8000 \
    --run_name va7_whole_body_from_scratch \
    agent.amp_cfg.amp_num_preload_transitions=20000 \
    "$@"
