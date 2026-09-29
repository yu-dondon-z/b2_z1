#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/dong/b2_z1"
PYTHON_BIN="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"

cd "${PROJECT_DIR}"

# VA2 starts from scratch.  It intentionally does not accept an implicit
# resume checkpoint, so the failed stationary VA1 solution cannot leak in.
# Extra CLI/Hydra overrides may be appended by the caller.
exec "${PYTHON_BIN}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-VA2-Truth-Curve \
    --num_envs 1024 \
    --headless \
    --max_iterations 8000 \
    --run_name va2 \
    agent.amp_cfg.amp_num_preload_transitions=20000 \
    "$@"
