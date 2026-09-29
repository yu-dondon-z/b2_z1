#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/dong/b2_z1"
PYTHON_BIN="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"

cd "${PROJECT_DIR}"

# Preserve VA3/model_3950. This is a separate, short continuation to verify
# whether suppressing cross-path TCP motion helps before any long training.
exec "${PYTHON_BIN}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-VA4-Smooth-Tool \
    --num_envs 1024 \
    --headless \
    --resume \
    --load_run 2026-09-23_10-04-36_va3_anti_kneeling_from_scratch \
    --checkpoint model_3950.pt \
    --reset_optimizer \
    --max_iterations 400 \
    --run_name va4_smooth_tool_from3950 \
    agent.amp_cfg.amp_num_preload_transitions=20000 \
    "$@"
