#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec /home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train_vb13_curriculum.py "$@"
