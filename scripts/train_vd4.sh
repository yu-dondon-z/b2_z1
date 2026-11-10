#!/usr/bin/env bash
# Start a separate VD4 run; trailing arguments override the defaults.
set -euo pipefail

vd4_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vd4_python="${VD4_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"
vd4_cache_root="${VD4_CACHE_ROOT:-/dev/shm/vd4_train_cache_$(id -u)}"
vd4_checkpoint="${VD4_WARM_START:-$vd4_root/logs/rsl_rl/b2_z1_vd3_curve_amp/2026-11-04_09-44-59_vd3_zero_gun_flange_stability/model_18000.pt}"

mkdir -p -- "$vd4_cache_root/cache" "$vd4_cache_root/tmp" "$vd4_cache_root/matplotlib" \
    "$vd4_cache_root/assets" "$vd4_cache_root/cuda" "$vd4_cache_root/gl" "$vd4_cache_root/mesa"
export PYTHONPATH="$vd4_root/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$vd4_cache_root/cache" TMPDIR="$vd4_cache_root/tmp"
export MPLCONFIGDIR="$vd4_cache_root/matplotlib" VD4_ASSET_CACHE="$vd4_cache_root/assets"
export CUDA_CACHE_PATH="$vd4_cache_root/cuda" __GL_SHADER_DISK_CACHE_PATH="$vd4_cache_root/gl"
export MESA_SHADER_CACHE_DIR="$vd4_cache_root/mesa"

vd4_warm_args=(--vd4-warm-start "$vd4_checkpoint")
for vd4_arg in "$@"; do
    if [[ "$vd4_arg" == --resume ]]; then
        vd4_warm_args=()
    fi
done

cd -- "$vd4_root"
exec "$vd4_python" scripts/rsl_rl/train.py \
    --task Unitree-b2z1-VD4-Velocity --headless --num_envs 256 --max_iterations 20000 \
    --experiment_name b2_z1_vd4_front_calibrated_amp --run_name vd4_front_calibrated \
    --keep-last-checkpoints 5 --log-root "$vd4_root/logs/rsl_rl" \
    "${vd4_warm_args[@]}" agent.amp_cfg.amp_num_preload_transitions=200000 agent.save_interval=250 "$@"
