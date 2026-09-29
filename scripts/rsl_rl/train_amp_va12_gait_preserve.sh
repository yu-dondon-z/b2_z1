#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/dong/b2_z1"
PYTHON_BIN="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"

cd "${PROJECT_DIR}"

# Restart from the stable VA7 actor. The Z1 teacher remains strong; the B2
# teacher preserves its straight trot but yields to signed yaw commands.
exec "${PYTHON_BIN}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-VA12-Gait-Preserve \
    --num_envs 1024 \
    --headless \
    --resume \
    --load_run 2026-09-24_09-50-29_va7_whole_body_from_scratch \
    --checkpoint model_7999.pt \
    --reset_optimizer \
    --arm-teacher-weight 5.0 \
    --leg-teacher-weight 0.5 \
    --leg-teacher-yaw-release 0.10 \
    --max_iterations 300 \
    --run_name va12_gait_preserve_from7999 \
    agent.amp_cfg.amp_num_preload_transitions=20000 \
    "$@"
