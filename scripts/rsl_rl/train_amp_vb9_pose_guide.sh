#!/usr/bin/env bash
set -euo pipefail
task_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$task_repo"
stage="${1:-1}"
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
  1)
    task_name=Unitree-b2z1-AMP-VB9-Pose-Guide
    task_load_run=2026-10-09_10-12-22_VB7_load_robust_long
    task_checkpoint=model_10000.pt
    task_run_name=VB9_pose_guide_fromVB7_10000
    ;;
  2)
    task_name=Unitree-b2z1-AMP-VB9-Load-Stable
    task_load_run="${VB9_LOAD_RUN:?Set VB9_LOAD_RUN to a screened stage-A run}"
    task_checkpoint="${VB9_CHECKPOINT:?Set VB9_CHECKPOINT to a screened stage-A checkpoint}"
    task_run_name=VB9_load_stable_from_screened_pose
    ;;
  *) echo "Usage: bash scripts/rsl_rl/train_amp_vb9_pose_guide.sh [1|2] [Hydra overrides...]" >&2; exit 1;;
esac
if [[ ! -f "logs/rsl_rl/b2_z1_amp/${task_load_run}/${task_checkpoint}" ]]; then
    echo "Checkpoint missing; refusing training: ${task_load_run}/${task_checkpoint}" >&2
    exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train_amp.py \
  --task "$task_name" --num_envs "${VB9_NUM_ENVS:-256}" --headless \
  --max_iterations "${VB9_ITERATIONS:-1000}" --resume --reset_optimizer \
  --load_run "$task_load_run" --checkpoint "$task_checkpoint" --run_name "$task_run_name" \
  agent.amp_cfg.amp_num_preload_transitions=20000 "$@"
