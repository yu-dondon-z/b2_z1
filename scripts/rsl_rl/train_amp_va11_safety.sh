#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/dong/b2_z1"
PYTHON_BIN="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"

cd "${PROJECT_DIR}"

# Restart from VA7/7999.  VA10 improved path error but lost stability; do not
# continue its weights.  VA7 is both the policy initialization and arm teacher.
exec "${PYTHON_BIN}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-VA11-Safety \
    --num_envs 1024 \
    --headless \
    --resume \
    --load_run 2026-09-24_09-50-29_va7_whole_body_from_scratch \
    --checkpoint model_7999.pt \
    --reset_optimizer \
    --arm-teacher-weight 5.0 \
    --max_iterations 300 \
    --run_name va11_safety_from7999 \
    agent.amp_cfg.amp_num_preload_transitions=20000 \
    "$@"
