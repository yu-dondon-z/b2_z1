#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/dong/b2_z1"
PYTHON_BIN="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"

cd "${PROJECT_DIR}"

# VA5 branches from the tested VA4 checkpoint. Neither VA3 nor VA4 weights
# are modified. Screen the result in playback before a longer continuation.
exec "${PYTHON_BIN}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-VA5-Elbow-Wrist \
    --num_envs 1024 \
    --headless \
    --resume \
    --load_run 2026-09-23_16-27-15_va4_smooth_tool_from3950 \
    --checkpoint model_4349.pt \
    --reset_optimizer \
    --max_iterations 300 \
    --run_name va5_elbow_wrist_from4349 \
    agent.amp_cfg.amp_num_preload_transitions=20000 \
    "$@"
