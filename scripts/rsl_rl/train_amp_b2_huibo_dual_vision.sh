#!/usr/bin/env bash
# Train a new 73-observation / 19-action B2+Huibo actor. No Z1 checkpoint
# or Z1 action teacher is compatible with this morphology.
set -euo pipefail
HUIBO_PROJECT_DIR="/home/dong/b2_z1"
HUIBO_PYTHON="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"
cd "${HUIBO_PROJECT_DIR}"
exec "${HUIBO_PYTHON}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2huibo-AMP-DualVision \
    --num_envs 1024 --headless \
    --max_iterations 8000 --run_name huibo_fixed020_from_scratch \
    agent.amp_cfg.amp_num_preload_transitions=20000 \
    "$@"
