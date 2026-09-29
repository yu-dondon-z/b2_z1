#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/dong/b2_z1"
PYTHON_BIN="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"

cd "${PROJECT_DIR}"

# Fine-tune the validated VA7 arm policy; keep its weights and AMP style.
# --max_iterations is additional training, not an absolute checkpoint index.
exec "${PYTHON_BIN}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-VA8-Path \
    --num_envs 1024 \
    --headless \
    --resume \
    --load_run 2026-09-24_09-50-29_va7_whole_body_from_scratch \
    --checkpoint model_7999.pt \
    --reset_optimizer \
    --max_iterations 1000 \
    --run_name va8_base_path_from7999 \
    agent.amp_cfg.amp_num_preload_transitions=20000 \
    "$@"
