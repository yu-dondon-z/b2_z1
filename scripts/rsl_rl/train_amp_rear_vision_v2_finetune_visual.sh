#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/dong/b2_z1"
PYTHON_BIN="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"
NUM_ENVS="${B2Z1_VIS_NUM_ENVS:-32}"
MAX_ITERATIONS="${B2Z1_VIS_MAX_ITERATIONS:-1000}"
LOAD_RUN="2026-09-04_09-35-51_rear_deployment_finetune_from7999"
CHECKPOINT="model_9400.pt"
CHECKPOINT_PATH="${PROJECT_DIR}/logs/rsl_rl/b2_z1_amp/${LOAD_RUN}/${CHECKPOINT}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"

exec "${PYTHON_BIN}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-Rear-Vision-V2-Finetune-Visual \
    --num_envs "${NUM_ENVS}" \
    --resume \
    --load_run "${LOAD_RUN}" \
    --checkpoint "${CHECKPOINT}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --run_name rear_vision_v2_2_visual_repair_from9400 \
    --reset_optimizer \
    "$@"
