#!/usr/bin/env bash
# Fixed-speed proxy stage, initialized from VA15. Live image validation is
# scripts/vision/play_va16_dual_vision.py; this is not perception training.
set -euo pipefail
VA16_PROJECT_DIR="/home/dong/b2_z1"
VA16_PYTHON="${B2Z1_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"
cd "${VA16_PROJECT_DIR}"
exec "${VA16_PYTHON}" scripts/rsl_rl/train_amp.py \
    --task Unitree-b2z1-AMP-VA16-DualVision \
    --num_envs 1024 --headless --resume \
    --load_run 2026-09-28_15-07-31_va15_curriculum_from2500 \
    --checkpoint model_5000.pt --reset_optimizer \
    --teacher-checkpoint "${VA16_PROJECT_DIR}/logs/rsl_rl/b2_z1_amp/2026-09-24_09-50-29_va7_whole_body_from_scratch/model_7999.pt" \
    --arm-teacher-weight 5.0 --leg-teacher-weight 0.0 \
    --max_iterations 2000 --run_name va16_fixed020_proxy \
    agent.amp_cfg.amp_num_preload_transitions=20000 \
    "$@"
