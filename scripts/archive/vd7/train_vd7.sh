#!/usr/bin/env bash
# Historical launcher; original options retained, not the VD13 entry.
printf '[历史入口] train_vd7.sh：仅供旧版本复现；当前 VD13 见 docs/README.md。\n' >&2
set -euo pipefail
vd7_root="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/../../.." && pwd)"
vd7_source="$vd7_root/logs/rsl_rl/b2_z1_vd5_startup_amp/2026-11-10_11-47-57_vd5_startup_80ref_20stand_continue1000/model_4498.pt"
vd7_load=(--vd7-from "$vd7_source")
for vd7_arg in "$@"; do
    if [[ "$vd7_arg" == --resume || "$vd7_arg" == --vd7-from || "$vd7_arg" == --vd7-from=* ]]; then vd7_load=(); fi
done
vd7_cache="$(mktemp -d /dev/shm/vd7_train_XXXXXXXX)"
mkdir -p "$vd7_cache/cache" "$vd7_cache/tmp" "$vd7_cache/assets" "$vd7_cache/mpl"
export PYTHONPATH="$vd7_root/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$vd7_cache/cache" TMPDIR="$vd7_cache/tmp" MPLCONFIGDIR="$vd7_cache/mpl" VD3_ASSET_CACHE="$vd7_cache/assets"
cd "$vd7_root"
exec /home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
    --task Unitree-b2z1-VD7-Velocity --headless --num_envs 256 --max_iterations 100 \
    --experiment_name b2_z1_vd7_tcp_amp --run_name vd7_tcp_highpass_worst_impact_100iter \
    --keep-last-checkpoints 0 --log-root "$vd7_root/logs/rsl_rl" \
    '--kit_args=--/persistent/isaac/asset_root/cloud=/home/dong/isaacsim_assets/Assets/Isaac/4.5' \
    "${vd7_load[@]}" agent.amp_cfg.amp_num_preload_transitions=200000 agent.save_interval=100 "$@"
