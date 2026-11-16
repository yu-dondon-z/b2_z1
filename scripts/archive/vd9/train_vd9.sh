#!/usr/bin/env bash
# Historical launcher; original options retained, not the VD13 entry.
printf '[历史入口] train_vd9.sh：仅供旧版本复现；当前 VD13 见 docs/README.md。\n' >&2
set -euo pipefail
vd9_root="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/../../.." && pwd)"
vd9_source="$vd9_root/logs/rsl_rl/b2_z1_vd5_startup_amp/2026-11-10_11-47-57_vd5_startup_80ref_20stand_continue1000/model_4498.pt"
vd9_load=(--vd9-from "$vd9_source")
vd9_run_name=vd9_physics_rate_from_vd5_stage1800
for vd9_arg in "$@"; do
    if [[ "$vd9_arg" == --resume || "$vd9_arg" == --vd9-from || "$vd9_arg" == --vd9-from=* || \
          "$vd9_arg" == --vd9-warm-start || "$vd9_arg" == --vd9-warm-start=* ]]; then
        vd9_load=()
    fi
    if [[ "$vd9_arg" == --vd9-warm-start || "$vd9_arg" == --vd9-warm-start=* ]]; then
        vd9_run_name=vd9_fresh_warm_start
    fi
done
vd9_cache="$(mktemp -d /dev/shm/vd9_train_XXXXXXXX)"
mkdir -p "$vd9_cache/cache" "$vd9_cache/tmp" "$vd9_cache/assets" "$vd9_cache/mpl"
export PYTHONPATH="$vd9_root/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$vd9_cache/cache" TMPDIR="$vd9_cache/tmp" MPLCONFIGDIR="$vd9_cache/mpl" VD3_ASSET_CACHE="$vd9_cache/assets"
cd "$vd9_root"
exec /home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
    --task Unitree-b2z1-VD9-Velocity --headless --num_envs 256 --max_iterations 1800 \
    --experiment_name b2_z1_vd9_physics_rate_amp --run_name "$vd9_run_name" \
    --keep-last-checkpoints 0 --log-root "$vd9_root/logs/rsl_rl" \
    '--kit_args=--/persistent/isaac/asset_root/cloud=/home/dong/isaacsim_assets/Assets/Isaac/4.5' \
    agent.save_interval=100 "${vd9_load[@]}" "$@"
