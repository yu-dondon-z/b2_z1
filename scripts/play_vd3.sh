#!/usr/bin/env bash
# Interactive VD3 playback in the shared training workspace.
set -euo pipefail

vd3_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vd3_python="${VD3_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"
vd3_cache="${VD3_PLAY_CACHE:-/dev/shm/vd3_play_cache_$(id -u)}"
vd3_checkpoint="${VD3_CHECKPOINT:-}"
if [[ -z "$vd3_checkpoint" ]]; then
    vd3_checkpoint="$("$vd3_python" - "$vd3_root" <<'PY'
from pathlib import Path
import sys
roots = (Path(sys.argv[1]),)
runs = []
for root in roots:
    experiment = root / 'logs/rsl_rl/b2_z1_vd3_curve_amp'
    candidates = list(experiment.glob('*_vd3_no_tool_support'))
    candidates += list(experiment.glob('2026-11-04_09-44-59_vd3_zero_gun_flange_stability'))
    for run in candidates:
        models = [p for p in run.glob('model_*.pt') if p.stem[6:].isdigit() and p.is_file()]
        if models:
            runs.append((run.name, max(models, key=lambda p: int(p.stem[6:]))))
if not runs:
    raise SystemExit('No VD3 checkpoint found; set VD3_CHECKPOINT to its absolute path.')
print(max(runs, key=lambda entry: entry[0])[1].resolve())
PY
)"
fi
[[ -f "$vd3_checkpoint" ]] || { echo "Checkpoint not found: $vd3_checkpoint" >&2; exit 1; }
mkdir -p -- "$vd3_cache/cache" "$vd3_cache/tmp" "$vd3_cache/matplotlib" "$vd3_cache/assets"
export PYTHONPATH="$vd3_root/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$vd3_cache/cache" TMPDIR="$vd3_cache/tmp"
export MPLCONFIGDIR="$vd3_cache/matplotlib" VD3_ASSET_CACHE="$vd3_cache/assets"
echo "[VD3 Play] checkpoint: $vd3_checkpoint"
cd -- "$vd3_root"
exec "$vd3_python" scripts/rsl_rl/play.py \
    --task Unitree-b2z1-Velocity-Play --num_envs 1 --device cuda:0 --real-time --follow_robot \
    --checkpoint "$vd3_checkpoint" \
    env.commands.base_velocity.rel_standing_envs=0.0 \
    'env.commands.base_velocity.turn_angle_range=[0.7853981633974483,1.5707963267948966]' \
    'env.viewer.eye=[3.0,3.0,2.0]' 'env.viewer.lookat=[0.0,0.0,0.4]' \
    "$@"
