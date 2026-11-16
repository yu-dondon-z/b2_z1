#!/usr/bin/env bash
set -euo pipefail
vd14_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vd14_source="$vd14_root/logs/rsl_rl/b2_z1_vd13_mount75_amp/2026-11-15_23-51-49_vd13_mount75_resume/model_6199.pt"
vd14_load=(--vd14-from "$vd14_source")
for vd14_arg in "$@"; do
    if [[ "$vd14_arg" == --resume || "$vd14_arg" == --vd14-from || "$vd14_arg" == --vd14-from=* ]]; then
        vd14_load=()
    fi
done
vd14_cache="$(mktemp -d "${VD14_CACHE_ROOT:-/dev/shm}/vd14_train_XXXXXXXX")"
mkdir -p "$vd14_cache/cache" "$vd14_cache/tmp" "$vd14_cache/assets" "$vd14_cache/mpl"
export PYTHONPATH="$vd14_root/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$vd14_cache/cache" TMPDIR="$vd14_cache/tmp" MPLCONFIGDIR="$vd14_cache/mpl"
# Same measured model as VD13; do not regenerate or edit the preserved URDF.
export VD13_ASSET_CACHE="$vd14_cache/assets" VD10_ASSET_CACHE="$vd14_cache/unused_vd10"
cd "$vd14_root"
exec "${VD14_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}" scripts/rsl_rl/train.py \
    --task Unitree-b2z1-VD14-Velocity --headless --num_envs 256 --max_iterations 6000 \
    --experiment_name b2_z1_vd14_posture_amp --run_name vd14_posture_long \
    --logger tensorboard --keep-last-checkpoints 0 --log-root "$vd14_root/logs/rsl_rl" \
    '--kit_args=--/persistent/isaac/asset_root/cloud=/home/dong/isaacsim_assets/Assets/Isaac/4.5' \
    agent.save_interval=100 "${vd14_load[@]}" "$@"
