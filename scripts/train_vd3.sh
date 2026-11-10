#!/usr/bin/env bash
# Long training command; all trailing CLI arguments override the defaults below.
set -euo pipefail

vd3_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vd3_python="${VD3_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"
vd3_cache_root="${VD3_CACHE_ROOT:-/dev/shm/vd3_train_cache_$(id -u)}"
vd3_checkpoint="${VD3_WARM_START:-$vd3_root/logs/rsl_rl/b2_z1_vd2_curve_amp/2026-11-03_10-33-21_vd2_curriculum_resume/model_2050.pt}"

mkdir -p -- "$vd3_cache_root/cache" "$vd3_cache_root/tmp" "$vd3_cache_root/matplotlib" \
    "$vd3_cache_root/assets" "$vd3_cache_root/cuda" "$vd3_cache_root/gl" "$vd3_cache_root/mesa"
export PYTHONPATH="$vd3_root/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$vd3_cache_root/cache"
export TMPDIR="$vd3_cache_root/tmp"
export MPLCONFIGDIR="$vd3_cache_root/matplotlib"
export VD3_ASSET_CACHE="$vd3_cache_root/assets"
export CUDA_CACHE_PATH="$vd3_cache_root/cuda"
export __GL_SHADER_DISK_CACHE_PATH="$vd3_cache_root/gl"
export MESA_SHADER_CACHE_DIR="$vd3_cache_root/mesa"

vd3_warm_args=(--vd3-warm-start "$vd3_checkpoint")
for vd3_arg in "$@"; do
    if [[ "$vd3_arg" == --resume ]]; then
        vd3_warm_args=()
    fi
done

cd -- "$vd3_root"
exec "$vd3_python" scripts/rsl_rl/train.py \
    --task Unitree-b2z1-Velocity --headless --num_envs 256 --max_iterations 20000 \
    --experiment_name b2_z1_vd3_curve_amp --run_name vd3_no_tool_support \
    --keep-last-checkpoints 5 --log-root "$vd3_root/logs/rsl_rl" \
    "${vd3_warm_args[@]}" \
    agent.amp_cfg.amp_num_preload_transitions=200000 agent.save_interval=250 \
    "$@"
