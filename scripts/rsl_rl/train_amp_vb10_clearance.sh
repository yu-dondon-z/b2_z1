#!/usr/bin/env bash
set -euo pipefail
task_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$task_repo"
task_run=2026-10-13_09-13-05_VB9_pose_guide_fromVB7_10000
task_checkpoint=model_10999.pt
if [[ ! -f "logs/rsl_rl/b2_z1_amp/${task_run}/${task_checkpoint}" ]]; then
    echo "Missing screened VB9 checkpoint: ${task_run}/${task_checkpoint}" >&2
    exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train_amp.py \
  --task Unitree-b2z1-AMP-VB10-Clearance --headless \
  --num_envs "${VB10_NUM_ENVS:-256}" --max_iterations "${VB10_ITERATIONS:-1500}" \
  --resume --reset_optimizer --load_run "$task_run" --checkpoint "$task_checkpoint" \
  --run_name VB10_clearance_fromVB9_10999 \
  agent.amp_cfg.amp_num_preload_transitions=20000 "$@"
