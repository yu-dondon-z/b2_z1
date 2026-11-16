#!/usr/bin/env bash
set -euo pipefail
vd13_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vd13_source="$vd13_root/logs/rsl_rl/b2_z1_vd12_geometry_guard_amp/2026-11-14_15-07-38_vd12_geometry_guard_long_20261114/model_5999.pt"
vd13_load=(--vd13-from "$vd13_source")
for vd13_arg in "$@"; do
    if [[ "$vd13_arg" == --resume || "$vd13_arg" == --vd13-from || "$vd13_arg" == --vd13-from=* ]]; then
        vd13_load=()
    fi
done
vd13_cache="$(mktemp -d "${VD13_CACHE_ROOT:-/dev/shm}/vd13_train_XXXXXXXX")"
mkdir -p "$vd13_cache/cache" "$vd13_cache/tmp" "$vd13_cache/assets" "$vd13_cache/mpl"
export PYTHONPATH="$vd13_root/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$vd13_cache/cache" TMPDIR="$vd13_cache/tmp" MPLCONFIGDIR="$vd13_cache/mpl"
export VD13_ASSET_CACHE="$vd13_cache/assets" VD10_ASSET_CACHE="$vd13_cache/unused_vd10"
cd "$vd13_root"
exec "${VD13_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}" scripts/rsl_rl/train.py \
    --task Unitree-b2z1-VD13-Velocity --headless --num_envs 256 --max_iterations 6000 \
    --experiment_name b2_z1_vd13_mount75_amp --run_name vd13_mount75_long \
    --logger tensorboard --keep-last-checkpoints 0 --log-root "$vd13_root/logs/rsl_rl" \
    '--kit_args=--/persistent/isaac/asset_root/cloud=/home/dong/isaacsim_assets/Assets/Isaac/4.5' \
    agent.save_interval=100 "${vd13_load[@]}" "$@"
