#!/usr/bin/env bash
# Final posture candidate, local simulation only; no export/hardware.
set -euo pipefail
vd14_play_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vd14_play_cache="$(mktemp -d /dev/shm/vd14_play_XXXXXXXX)"
mkdir -p "$vd14_play_cache/cache" "$vd14_play_cache/tmp" "$vd14_play_cache/assets" "$vd14_play_cache/mpl"
export PYTHONPATH="$vd14_play_root/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$vd14_play_cache/cache" TMPDIR="$vd14_play_cache/tmp"
export MPLCONFIGDIR="$vd14_play_cache/mpl" VD13_ASSET_CACHE="$vd14_play_cache/assets"
export VD10_ASSET_CACHE="$vd14_play_cache/unused_vd10"
export VD14_PLAY_RESULT_FILE="$vd14_play_cache/result.json"
cd "$vd14_play_root"
vd14_play_python="${VD14_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"
vd14_play_exit=0
"$vd14_play_python" -B scripts/diagnostics/play_vd14_checked.py \
  '--kit_args=--/persistent/isaac/asset_root/cloud=/home/dong/isaacsim_assets/Assets/Isaac/4.5' "$@" || vd14_play_exit=$?
if (( vd14_play_exit != 0 )); then exit "$vd14_play_exit"; fi
for vd14_play_arg in "$@"; do
  if [[ "$vd14_play_arg" == --help || "$vd14_play_arg" == -h ]]; then exit 0; fi
done
if [[ ! -s "$VD14_PLAY_RESULT_FILE" ]]; then
  printf '%s\n' 'VD14 Play has no completed result; Kit exit(0) does not imply a pass.' >&2
  exit 1
fi
"$vd14_play_python" -c 'import json, os, sys; result=json.load(open(os.environ["VD14_PLAY_RESULT_FILE"])); sys.exit(result["exit_code"])'
