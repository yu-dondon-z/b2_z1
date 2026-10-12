#!/usr/bin/env bash
set -euo pipefail
task_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$task_repo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train_amp.py \
  --task Unitree-b2z1-AMP-VB8-Quiet-Load \
  --num_envs "${VB8_NUM_ENVS:-512}" --headless \
  --max_iterations "${VB8_ITERATIONS:-500}" \
  --resume --reset_optimizer \
  --load_run 2026-10-09_10-12-22_VB7_load_robust_long \
  --checkpoint model_10000.pt --run_name VB8_quiet_load_from10000 \
  agent.amp_cfg.amp_num_preload_transitions=20000 "$@"
