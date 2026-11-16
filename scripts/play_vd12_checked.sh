#!/usr/bin/env bash
set -euo pipefail

vd12_play_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vd12_play_cache="$(mktemp -d /dev/shm/vd12_play_XXXXXXXX)"
mkdir -p "$vd12_play_cache/cache" "$vd12_play_cache/tmp" "$vd12_play_cache/assets" "$vd12_play_cache/mpl"
export PYTHONPATH="$vd12_play_root/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$vd12_play_cache/cache" TMPDIR="$vd12_play_cache/tmp"
export MPLCONFIGDIR="$vd12_play_cache/mpl" VD10_ASSET_CACHE="$vd12_play_cache/assets"
export VD12_PLAY_RESULT_FILE="$vd12_play_cache/result.json"
cd "$vd12_play_root"
vd12_play_python="${VD12_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"
vd12_play_exit=0
"$vd12_play_python" scripts/diagnostics/play_vd12_checked.py \
    '--kit_args=--/persistent/isaac/asset_root/cloud=/home/dong/isaacsim_assets/Assets/Isaac/4.5' "$@" || vd12_play_exit=$?
if (( vd12_play_exit != 0 )); then
    exit "$vd12_play_exit"
fi
for vd12_play_arg in "$@"; do
    if [[ "$vd12_play_arg" == --help || "$vd12_play_arg" == -h ]]; then
        exit 0
    fi
done
if [[ ! -s "$VD12_PLAY_RESULT_FILE" ]]; then
    printf '%s\n' 'VD12 Play has no completed result; Kit exit(0) does not imply a pass.' >&2
    exit 1
fi
"$vd12_play_python" -c 'import json, os, sys; result=json.load(open(os.environ["VD12_PLAY_RESULT_FILE"])); sys.exit(result["exit_code"])'
