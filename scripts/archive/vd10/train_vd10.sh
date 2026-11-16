#!/usr/bin/env bash
# Historical launcher; original options retained, not the VD13 entry.
printf '[历史入口] train_vd10.sh：仅供旧版本复现；当前 VD13 见 docs/README.md。\n' >&2
set -euo pipefail

vd10_root="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/../../.." && pwd)"
vd10_initial="$vd10_root/logs/rsl_rl/b2_z1_vd5_startup_amp/2026-11-10_11-47-57_vd5_startup_80ref_20stand_continue1000/model_4498.pt"
vd10_load=(--vd10-from "$vd10_initial")
for vd10_arg in "$@"; do
    if [[ "$vd10_arg" == --resume || "$vd10_arg" == --vd10-from || "$vd10_arg" == --vd10-from=* ]]; then
        vd10_load=()
    fi
done

# Isolate Isaac Lab's generated USD and cache from other variants. Logs and
# checkpoints always remain on persistent storage under the experiment name.
vd10_cache="$(mktemp -d /dev/shm/vd10_train_XXXXXXXX)"
mkdir -p "$vd10_cache/cache" "$vd10_cache/tmp" "$vd10_cache/assets" "$vd10_cache/mpl"
export PYTHONPATH="$vd10_root/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$vd10_cache/cache" TMPDIR="$vd10_cache/tmp"
export MPLCONFIGDIR="$vd10_cache/mpl" VD10_ASSET_CACHE="$vd10_cache/assets"

cd "$vd10_root"
exec /home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
    --task Unitree-b2z1-VD10-Velocity --headless --num_envs 256 --max_iterations 4000 \
    --experiment_name b2_z1_vd10_hollow_gun_amp --run_name vd10_hollow_gun_long \
    --keep-last-checkpoints 0 --log-root "$vd10_root/logs/rsl_rl" \
    '--kit_args=--/persistent/isaac/asset_root/cloud=/home/dong/isaacsim_assets/Assets/Isaac/4.5' \
    agent.save_interval=100 "${vd10_load[@]}" "$@"
