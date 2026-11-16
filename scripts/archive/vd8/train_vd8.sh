#!/usr/bin/env bash
# Historical launcher; original options retained, not the VD13 entry.
printf '[历史入口] train_vd8.sh：仅供旧版本复现；当前 VD13 见 docs/README.md。\n' >&2
set -euo pipefail
vd8_root="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/../../.." && pwd)"
vd8_source="$vd8_root/logs/rsl_rl/b2_z1_vd5_startup_amp/2026-11-10_11-47-57_vd5_startup_80ref_20stand_continue1000/model_4498.pt"
vd8_load=(--vd8-from "$vd8_source")
for vd8_arg in "$@"; do
    if [[ "$vd8_arg" == --resume || "$vd8_arg" == --vd8-from || "$vd8_arg" == --vd8-from=* ]]; then vd8_load=(); fi
done
vd8_cache="$(mktemp -d /dev/shm/vd8_train_XXXXXXXX)"
mkdir -p "$vd8_cache/cache" "$vd8_cache/tmp" "$vd8_cache/assets" "$vd8_cache/mpl"
export PYTHONPATH="$vd8_root/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$vd8_cache/cache" TMPDIR="$vd8_cache/tmp" MPLCONFIGDIR="$vd8_cache/mpl" VD3_ASSET_CACHE="$vd8_cache/assets"
cd "$vd8_root"
exec /home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
    --task Unitree-b2z1-VD8-Velocity --headless --num_envs 256 --max_iterations 600 \
    --experiment_name b2_z1_vd8_roll_landing_amp --run_name vd8_roll_landing_initial_600iter \
    --keep-last-checkpoints 0 --log-root "$vd8_root/logs/rsl_rl" \
    '--kit_args=--/persistent/isaac/asset_root/cloud=/home/dong/isaacsim_assets/Assets/Isaac/4.5' \
    "${vd8_load[@]}" agent.amp_cfg.amp_num_preload_transitions=200000 agent.save_interval=100 "$@"
