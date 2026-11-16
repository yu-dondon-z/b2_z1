#!/usr/bin/env bash
# Fixed completed VD12 run; does not select latest or alter the historical 2000 entry.
set -euo pipefail

vd12_final_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vd12_final_checkpoint="$vd12_final_root/logs/rsl_rl/b2_z1_vd12_geometry_guard_amp/2026-11-14_15-07-38_vd12_geometry_guard_long_20261114/model_5999.pt"
if [[ ! -s "$vd12_final_checkpoint" ]]; then
    printf '%s\n' "Missing completed VD12 checkpoint: $vd12_final_checkpoint" >&2
    exit 1
fi
exec bash "$vd12_final_root/scripts/play_vd12_checked.sh" \
    --checkpoint "$vd12_final_checkpoint" "$@"
