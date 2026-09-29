#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/dong/b2_z1"
PYTHON_BIN="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"

cd "${PROJECT_DIR}"

# VA1 is intentionally trained from scratch: no --resume, --load_run or
# --checkpoint arguments are used. Extra Hydra overrides can be appended by
# the caller, for example: ./scripts/rsl_rl/train_amp_va1_truth_curve.sh --seed 7
exec "${PYTHON_BIN}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-VA1-Truth-Curve \
    --num_envs 1024 \
    --headless \
    --max_iterations 8000 \
    --run_name va1 \
    agent.amp_cfg.amp_num_preload_transitions=20000 \
    "$@"
