#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/dong/b2_z1"
PYTHON_BIN="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"

cd "${PROJECT_DIR}"

# Branch from VA4, not the rejected VA5. A learned nullspace objective avoids
# the unstable action modification seen in playback-only experiments.
exec "${PYTHON_BIN}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-VA6-Nullspace \
    --num_envs 1024 \
    --headless \
    --resume \
    --load_run 2026-09-23_16-27-15_va4_smooth_tool_from3950 \
    --checkpoint model_4349.pt \
    --reset_optimizer \
    --max_iterations 300 \
    --run_name va6_nullspace_from4349 \
    agent.amp_cfg.amp_num_preload_transitions=20000 \
    "$@"
