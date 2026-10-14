#!/usr/bin/env bash
set -euo pipefail
task_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$task_repo"
task_run=2026-10-13_10-35-50_VB10_clearance_fromVB9_10999
task_checkpoint=model_12300.pt
if [[ ! -f "logs/rsl_rl/b2_z1_amp/${task_run}/${task_checkpoint}" ]]; then
    echo "Missing screened training donor: ${task_run}/${task_checkpoint}" >&2
    exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train_amp.py \
  --task Unitree-b2z1-AMP-VB11-Ready --headless \
  --num_envs "${VB11_NUM_ENVS:-256}" --max_iterations "${VB11_ITERATIONS:-2000}" \
  --resume --reset_optimizer --load_run "$task_run" --checkpoint "$task_checkpoint" \
  --run_name VB11_ready_fromVB10_12300 \
  agent.amp_cfg.amp_num_preload_transitions=20000 "$@"
