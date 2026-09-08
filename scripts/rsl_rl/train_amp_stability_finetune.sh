#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/dong/b2_z1"
PYTHON_BIN="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"
LOAD_RUN="2026-08-05_09-16-19_deployment_finetune_v2_from11998"
CHECKPOINT="model_13997.pt"
CHECKPOINT_PATH="${PROJECT_DIR}/logs/rsl_rl/b2_z1_amp/${LOAD_RUN}/${CHECKPOINT}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"

exec "${PYTHON_BIN}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-Stability-Finetune \
    --num_envs 1024 \
    --headless \
    --resume \
    --load_run "${LOAD_RUN}" \
    --checkpoint "${CHECKPOINT}" \
    --max_iterations 1200 \
    --run_name stability_finetune_from13997 \
    --reset_optimizer \
    "$@"
