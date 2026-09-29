#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="/home/dong/b2_z1"
PYTHON_BIN="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"
cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-VA15-Curriculum \
    --num_envs 1024 --headless \
    --resume \
    --load_run 2026-09-28_10-26-49_va14_from_scratch \
    --checkpoint model_2500.pt \
    --reset_optimizer \
    --teacher-checkpoint "${PROJECT_DIR}/logs/rsl_rl/b2_z1_amp/2026-09-24_09-50-29_va7_whole_body_from_scratch/model_7999.pt" \
    --arm-teacher-weight 5.0 --leg-teacher-weight 0.0 \
    --max_iterations 5500 --run_name va15_curriculum_from2500 \
    agent.amp_cfg.amp_num_preload_transitions=20000 \
    "$@"
