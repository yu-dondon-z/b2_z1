#!/usr/bin/env bash
# Adapt VD5 without resetting its learned ground-target curriculum.
set -euo pipefail
vd5_startup_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vd5_startup_source="${VD5_STARTUP_FROM:-$vd5_startup_root/logs/rsl_rl/b2_z1_vd5_world_seam_amp/2026-11-10_00-02-22_vd5_world_seam/model_3250.pt}"
vd5_startup_args=(--vd5-startup-from "$vd5_startup_source")
for vd5_startup_arg in "$@"; do
    if [[ "$vd5_startup_arg" == --resume || "$vd5_startup_arg" == --vd5-startup-from || "$vd5_startup_arg" == --vd5-startup-from=* ]]; then
        vd5_startup_args=()
    fi
done
exec bash "$vd5_startup_root/scripts/train_vd5.sh" \
    --task Unitree-b2z1-VD5-Startup-Velocity \
    --experiment_name b2_z1_vd5_startup_amp --run_name vd5_startup_80ref_20stand \
    --max_iterations 3000 "${vd5_startup_args[@]}" "$@"
