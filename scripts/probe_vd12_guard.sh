#!/usr/bin/env bash
set -euo pipefail

vd12_probe_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vd12_probe_cache="$(mktemp -d /dev/shm/vd12_probe_XXXXXXXX)"
mkdir -p "$vd12_probe_cache/cache" "$vd12_probe_cache/tmp" "$vd12_probe_cache/assets" "$vd12_probe_cache/mpl"
export PYTHONPATH="$vd12_probe_root/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$vd12_probe_cache/cache" TMPDIR="$vd12_probe_cache/tmp"
export MPLCONFIGDIR="$vd12_probe_cache/mpl" VD10_ASSET_CACHE="$vd12_probe_cache/assets"
export VD12_PROBE_RESULT_FILE="$vd12_probe_cache/result.json"
vd12_probe_python="${VD12_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"
cd "$vd12_probe_root"
vd12_probe_exit=0
"$vd12_probe_python" scripts/diagnostics/probe_vd12_guard.py \
    '--kit_args=--/persistent/isaac/asset_root/cloud=/home/dong/isaacsim_assets/Assets/Isaac/4.5' "$@" || vd12_probe_exit=$?
if (( vd12_probe_exit != 0 )); then
    exit "$vd12_probe_exit"
fi
for vd12_probe_arg in "$@"; do
    if [[ "$vd12_probe_arg" == --help || "$vd12_probe_arg" == -h ]]; then
        exit 0
    fi
done
if [[ ! -s "$VD12_PROBE_RESULT_FILE" ]]; then
    printf '%s\n' 'VD12 probe produced no result; Kit exit(0) alone is not a pass.' >&2
    exit 1
fi
"$vd12_probe_python" -c 'import json, os, sys; sys.exit(json.load(open(os.environ["VD12_PROBE_RESULT_FILE"]))["exit_code"])'
