#!/usr/bin/env bash
set -euo pipefail
task_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$task_repo"
exec /home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train_vb12_guarded.py \
  --num-envs "${VB12_NUM_ENVS:-256}" --iterations "${VB12_ITERATIONS:-500}" "$@"
