#!/usr/bin/env bash
# Start the isolated VD5 seam-to-base-frame training run.
set -euo pipefail

vd5_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vd5_python="${VD5_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"
vd5_cache_root="${VD5_CACHE_ROOT:-/dev/shm/vd5_train_cache_$(id -u)}"
vd5_checkpoint="${VD5_WARM_START:-$vd5_root/logs/rsl_rl/b2_z1_vd3_curve_amp/2026-11-05_09-57-36_vd3_leg_mirror_from_10500/model_2000.pt}"
vd5_asset_root="${VD5_ASSET_ROOT:-/home/dong/isaacsim_assets/Assets/Isaac/4.5}"

# Use the installed asset pack instead of resolving the ground over the network.
for vd5_asset in Isaac/Environments/Grid/default_environment.usd \
    Isaac/Props/UIElements/frame_prim.usd Isaac/Props/UIElements/arrow_x.usd; do
    if [[ ! -f "$vd5_asset_root/$vd5_asset" ]]; then
        printf 'Missing VD5 scene asset: %s\nSet VD5_ASSET_ROOT to a complete local Isaac asset pack.\n' \
            "$vd5_asset_root/$vd5_asset" >&2
        exit 1
    fi
done

mkdir -p -- "$vd5_cache_root/cache" "$vd5_cache_root/tmp" "$vd5_cache_root/matplotlib" \
    "$vd5_cache_root/assets" "$vd5_cache_root/cuda" "$vd5_cache_root/gl" "$vd5_cache_root/mesa"
export PYTHONPATH="$vd5_root/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$vd5_cache_root/cache" TMPDIR="$vd5_cache_root/tmp"
export MPLCONFIGDIR="$vd5_cache_root/matplotlib"
# The shared robot asset loader still uses the VD3 cache variable.
export VD3_ASSET_CACHE="$vd5_cache_root/assets"
export CUDA_CACHE_PATH="$vd5_cache_root/cuda" __GL_SHADER_DISK_CACHE_PATH="$vd5_cache_root/gl"
export MESA_SHADER_CACHE_DIR="$vd5_cache_root/mesa"

vd5_warm_args=(--vd5-warm-start "$vd5_checkpoint")
for vd5_arg in "$@"; do
    if [[ "$vd5_arg" == --resume || "$vd5_arg" == --vd5-startup-from || "$vd5_arg" == --vd5-startup-from=* ]]; then
        vd5_warm_args=()
    fi
done

cd -- "$vd5_root"
exec "$vd5_python" scripts/rsl_rl/train.py \
    --task Unitree-b2z1-VD5-Velocity --headless --num_envs 256 --max_iterations 20000 \
    --experiment_name b2_z1_vd5_world_seam_amp --run_name vd5_world_seam \
    --keep-last-checkpoints 5 --log-root "$vd5_root/logs/rsl_rl" \
    "--kit_args=--/persistent/isaac/asset_root/cloud=$vd5_asset_root" \
    "${vd5_warm_args[@]}" agent.amp_cfg.amp_num_preload_transitions=200000 agent.save_interval=250 "$@"
