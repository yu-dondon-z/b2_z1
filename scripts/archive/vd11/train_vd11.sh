#!/usr/bin/env bash
# Historical launcher; original options retained, not the VD13 entry.
printf '[历史入口] train_vd11.sh：仅供旧版本复现；当前 VD13 见 docs/README.md。\n' >&2
set -euo pipefail

vd11_root="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/../../.." && pwd)"
vd11_initial="$vd11_root/logs/rsl_rl/b2_z1_vd10_hollow_gun_amp/2026-11-13_14-31-27_vd10_safe_start_continue5000_20261113/model_5998.pt"
vd11_load=(--vd11-from "$vd11_initial")
for vd11_arg in "$@"; do
    if [[ "$vd11_arg" == --resume || "$vd11_arg" == --vd11-from || "$vd11_arg" == --vd11-from=* ]]; then
        vd11_load=()
    fi
done

# Only generated caches use temporary storage. Logs/checkpoints are persistent;
# keep_last=0 preserves every checkpoint. --vd11-from= explicitly starts fresh.
vd11_cache="$(mktemp -d "${VD11_CACHE_ROOT:-/dev/shm}/vd11_train_XXXXXXXX")"
mkdir -p "$vd11_cache/cache" "$vd11_cache/tmp" "$vd11_cache/assets" "$vd11_cache/mpl"
export PYTHONPATH="$vd11_root/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$vd11_cache/cache" TMPDIR="$vd11_cache/tmp"
# The inherited measured tube builder reads VD10_ASSET_CACHE in both tasks.
export MPLCONFIGDIR="$vd11_cache/mpl" VD10_ASSET_CACHE="$vd11_cache/assets"

cd "$vd11_root"
exec "${VD11_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}" scripts/rsl_rl/train.py \
    --task Unitree-b2z1-VD11-Velocity --headless --num_envs 256 --max_iterations 6000 \
    --experiment_name b2_z1_vd11_motion_quality_amp --run_name vd11_motion_quality_long \
    --keep-last-checkpoints 0 --log-root "$vd11_root/logs/rsl_rl" \
    '--kit_args=--/persistent/isaac/asset_root/cloud=/home/dong/isaacsim_assets/Assets/Isaac/4.5' \
    agent.save_interval=100 "${vd11_load[@]}" "$@"
