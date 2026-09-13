#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/dong/b2_z1"
PYTHON_BIN="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"
LOAD_RUN="2026-09-09_16-01-19_rear_vision_v3_gait_finetune_from9600"
CHECKPOINT="model_10000.pt"
CHECKPOINT_PATH="${PROJECT_DIR}/logs/rsl_rl/b2_z1_amp/${LOAD_RUN}/${CHECKPOINT}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"

exec "${PYTHON_BIN}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-Rear-Curve-Gait-Heading-V5-Finetune \
    --num_envs 1024 \
    --headless \
    --resume \
    --load_run "${LOAD_RUN}" \
    --checkpoint "${CHECKPOINT}" \
    --max_iterations 1200 \
    --run_name rear_curve_gait_heading_v5_from10000 \
    --reset_optimizer \
    agent.amp_cfg.amp_num_preload_transitions=20000 \
    "$@"
