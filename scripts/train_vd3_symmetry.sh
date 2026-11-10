#!/usr/bin/env bash
# Warm-start VD3 10500 with leg mirror consistency; pass --resume for later runs.
set -euo pipefail

vd3_symmetry_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export VD3_WARM_START="${VD3_WARM_START:-/home/dong/rl_sar_b2_z1/outputs/VD3_SUPPORT_REVIEW_20261104/model_10500.pt}"

exec bash "$vd3_symmetry_root/scripts/train_vd3.sh" \
    --agent rsl_rl_symmetry_cfg_entry_point \
    --max_iterations 3000 --run_name vd3_leg_mirror_from_10500 \
    "$@"
