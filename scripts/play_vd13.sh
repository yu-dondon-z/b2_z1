#!/usr/bin/env bash
# Local simulation only; fixed completed mount75 checkpoint, no export/hardware.
set -euo pipefail
vd13_play_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vd13_play_cache="$(mktemp -d /dev/shm/vd13_play_XXXXXXXX)"
mkdir -p "$vd13_play_cache/cache" "$vd13_play_cache/tmp" "$vd13_play_cache/assets" "$vd13_play_cache/mpl"
export PYTHONPATH="$vd13_play_root/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$vd13_play_cache/cache" TMPDIR="$vd13_play_cache/tmp"
export MPLCONFIGDIR="$vd13_play_cache/mpl" VD13_ASSET_CACHE="$vd13_play_cache/assets"
export VD10_ASSET_CACHE="$vd13_play_cache/unused_vd10"
export VD13_PLAY_RESULT_FILE="$vd13_play_cache/result.json"
cd "$vd13_play_root"
vd13_play_python="${VD13_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"
vd13_play_exit=0
"$vd13_play_python" -B scripts/diagnostics/play_vd13_checked.py \
  '--kit_args=--/persistent/isaac/asset_root/cloud=/home/dong/isaacsim_assets/Assets/Isaac/4.5' "$@" || vd13_play_exit=$?
if (( vd13_play_exit != 0 )); then exit "$vd13_play_exit"; fi
for vd13_play_arg in "$@"; do
  if [[ "$vd13_play_arg" == --help || "$vd13_play_arg" == -h ]]; then exit 0; fi
done
if [[ ! -s "$VD13_PLAY_RESULT_FILE" ]]; then
  printf '%s\n' 'VD13 Play has no completed result; Kit exit(0) does not imply a pass.' >&2
  exit 1
fi
"$vd13_play_python" -c 'import json, os, sys; result=json.load(open(os.environ["VD13_PLAY_RESULT_FILE"])); sys.exit(result["exit_code"])'
