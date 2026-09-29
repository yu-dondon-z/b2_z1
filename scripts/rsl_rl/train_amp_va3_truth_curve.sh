#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/dong/b2_z1"
PYTHON_BIN="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"

cd "${PROJECT_DIR}"

# VA3 intentionally starts from scratch. Do not add --resume, --load_run or a
# checkpoint here: the VA2 low-posture local optimum must not enter this run.
# Extra CLI/Hydra overrides may be appended by the caller.
exec "${PYTHON_BIN}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-VA3-Truth-Curve \
    --num_envs 1024 \
    --headless \
    --max_iterations 8000 \
    --run_name va3_anti_kneeling_from_scratch \
    agent.amp_cfg.amp_num_preload_transitions=20000 \
    "$@"
