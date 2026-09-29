#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/dong/b2_z1"
PYTHON_BIN="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"

cd "${PROJECT_DIR}"

# Restart from VA7, not the rejected VA8 branch. The single 18-action actor
# retains its entire Z1 mapping while only B2 output rows are adapted.
exec "${PYTHON_BIN}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-VA9-Preserve-Arm \
    --num_envs 1024 \
    --headless \
    --resume \
    --load_run 2026-09-24_09-50-29_va7_whole_body_from_scratch \
    --checkpoint model_7999.pt \
    --reset_optimizer \
    --freeze-arm-actor \
    --max_iterations 800 \
    --run_name va9_preserve_arm_from7999 \
    agent.amp_cfg.amp_num_preload_transitions=20000 \
    "$@"
