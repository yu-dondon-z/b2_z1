#!/usr/bin/env bash
set -euo pipefail

vd11_play_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vd11_play_cache="$(mktemp -d /dev/shm/vd11_play_XXXXXXXX)"
mkdir -p "$vd11_play_cache/cache" "$vd11_play_cache/tmp" "$vd11_play_cache/assets" "$vd11_play_cache/mpl"
export PYTHONPATH="$vd11_play_root/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$vd11_play_cache/cache" TMPDIR="$vd11_play_cache/tmp"
export MPLCONFIGDIR="$vd11_play_cache/mpl" VD10_ASSET_CACHE="$vd11_play_cache/assets"
export VD11_PLAY_RESULT_FILE="$vd11_play_cache/result.json"

cd "$vd11_play_root"
vd11_play_python="${VD11_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"
vd11_play_exit=0
"$vd11_play_python" \
    scripts/diagnostics/play_vd11_checked.py \
    '--kit_args=--/persistent/isaac/asset_root/cloud=/home/dong/isaacsim_assets/Assets/Isaac/4.5' "$@" || vd11_play_exit=$?
if (( vd11_play_exit != 0 )); then
    exit "$vd11_play_exit"
fi
for vd11_play_arg in "$@"; do
    if [[ "$vd11_play_arg" == --help || "$vd11_play_arg" == -h ]]; then
        exit 0
    fi
done
if [[ ! -s "$VD11_PLAY_RESULT_FILE" ]]; then
    printf '%s\n' 'VD11 Play did not finish a result; do not interpret Kit exit(0) as a pass.' >&2
    exit 1
fi
"$vd11_play_python" -c 'import json, os, sys; result=json.load(open(os.environ["VD11_PLAY_RESULT_FILE"])); sys.exit(result["exit_code"])'
