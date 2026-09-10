#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

exec /home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train_amp.py \
  --task Unitree-b2z1-AMP-Proprio-Omni-V4-Finetune \
  --num_envs 1024 \
  --resume \
  --load_run 2026-09-09_16-01-19_rear_vision_v3_gait_finetune_from9600 \
  --checkpoint model_10000.pt \
  --max_iterations 2000 \
  --run_name proprio_omni_v4_from_model10000 \
  --reset_optimizer \
  --headless \
  agent.amp_cfg.amp_num_preload_transitions=20000
