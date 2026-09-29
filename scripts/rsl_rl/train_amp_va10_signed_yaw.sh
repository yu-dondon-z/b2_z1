#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/dong/b2_z1"
PYTHON_BIN="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"

cd "${PROJECT_DIR}"

# VA8/VA9 pilot weights are intentionally not used: their fixed-left-arc
# response drifted right. Resume from the proven VA7 arm baseline instead.
exec "${PYTHON_BIN}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-VA10-Signed-Yaw \
    --num_envs 1024 \
    --headless \
    --resume \
    --load_run 2026-09-24_09-50-29_va7_whole_body_from_scratch \
    --checkpoint model_7999.pt \
    --reset_optimizer \
    --arm-teacher-weight 2.0 \
    --max_iterations 300 \
    --run_name va10_signed_yaw_review_from7999 \
    agent.amp_cfg.amp_num_preload_transitions=20000 \
    "$@"
