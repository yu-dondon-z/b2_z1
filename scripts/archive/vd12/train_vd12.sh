#!/usr/bin/env bash
# Historical launcher; original options retained, not the VD13 entry.
printf '[历史入口] train_vd12.sh：仅供旧版本复现；当前 VD13 见 docs/README.md。\n' >&2
set -euo pipefail

vd12_root="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/../../.." && pwd)"
vd12_initial="$vd12_root/logs/rsl_rl/b2_z1_vd11_motion_quality_amp/2026-11-13_17-56-49_vd11_motion_quality_long_20261113/model_5999.pt"
vd12_load=(--vd12-from "$vd12_initial")
for vd12_arg in "$@"; do
    if [[ "$vd12_arg" == --resume || "$vd12_arg" == --vd12-from || "$vd12_arg" == --vd12-from=* ]]; then
        vd12_load=()
    fi
done

# Only generated caches use temporary storage. Keep every durable checkpoint;
# --vd12-from= explicitly starts fresh, while --resume restores only same VD12.
vd12_cache="$(mktemp -d "${VD12_CACHE_ROOT:-/dev/shm}/vd12_train_XXXXXXXX")"
mkdir -p "$vd12_cache/cache" "$vd12_cache/tmp" "$vd12_cache/assets" "$vd12_cache/mpl"
export PYTHONPATH="$vd12_root/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$vd12_cache/cache" TMPDIR="$vd12_cache/tmp"
# VD12 inherits the measured-tube asset builder, not the original massless VD5.
export MPLCONFIGDIR="$vd12_cache/mpl" VD10_ASSET_CACHE="$vd12_cache/assets"

cd "$vd12_root"
exec "${VD12_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}" scripts/rsl_rl/train.py \
    --task Unitree-b2z1-VD12-Velocity --headless --num_envs 256 --max_iterations 6000 \
    --experiment_name b2_z1_vd12_geometry_guard_amp --run_name vd12_geometry_guard_long \
    --keep-last-checkpoints 0 --log-root "$vd12_root/logs/rsl_rl" \
    '--kit_args=--/persistent/isaac/asset_root/cloud=/home/dong/isaacsim_assets/Assets/Isaac/4.5' \
    agent.save_interval=100 "${vd12_load[@]}" "$@"
