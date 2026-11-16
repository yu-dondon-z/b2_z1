#!/usr/bin/env bash
# Historical launcher; original options retained, not the VD13 entry.
printf '[历史入口] train_vd6.sh：仅供旧版本复现；当前 VD13 见 docs/README.md。\n' >&2
# Separate experiment; exact VD5 Startup optimizer/AMP/curriculum continuation.
set -euo pipefail
vd6_root="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/../../.." && pwd)"
vd6_source="${VD6_FROM:-$vd6_root/logs/rsl_rl/b2_z1_vd5_startup_amp/2026-11-10_11-47-57_vd5_startup_80ref_20stand_continue1000/model_4498.pt}"
vd6_load=(--vd6-from "$vd6_source")
for vd6_arg in "$@"; do
    if [[ "$vd6_arg" == --resume || "$vd6_arg" == --vd6-from || "$vd6_arg" == --vd6-from=* ]]; then
        vd6_load=()
    fi
done
vd6_cache="${VD6_CACHE_ROOT:-/dev/shm/vd6_train_cache_$(id -u)}"
mkdir -p "$vd6_cache/cache" "$vd6_cache/tmp" "$vd6_cache/assets" "$vd6_cache/mpl"
export PYTHONPATH="$vd6_root/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$vd6_cache/cache" TMPDIR="$vd6_cache/tmp"
export MPLCONFIGDIR="$vd6_cache/mpl" VD3_ASSET_CACHE="$vd6_cache/assets"
cd "$vd6_root"
exec /home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
    --task Unitree-b2z1-VD6-Velocity --headless --num_envs 256 --max_iterations 1000 \
    --experiment_name b2_z1_vd6_smooth_amp --run_name vd6_soft_landing_arm_continuity \
    --keep-last-checkpoints 5 --log-root "$vd6_root/logs/rsl_rl" \
    '--kit_args=--/persistent/isaac/asset_root/cloud=/home/dong/isaacsim_assets/Assets/Isaac/4.5' \
    "${vd6_load[@]}" agent.amp_cfg.amp_num_preload_transitions=200000 agent.save_interval=100 "$@"
