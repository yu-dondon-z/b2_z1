#!/usr/bin/env bash
# Run the preserved VD3 curve Play code without changing the training checkout.
set -euo pipefail

vd3_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vd3_python="${VD3_PYTHON:-/home/dong/anaconda3/envs/env_isaaclab/bin/python}"
vd3_checkpoint="${VD3_CHECKPOINT:-$vd3_root/logs/rsl_rl/b2_z1_vd3_curve_amp/2026-11-04_09-44-59_vd3_zero_gun_flange_stability/model_18000.pt}"
vd3_prepare_only=0
if [[ "${1:-}" == --prepare-only ]]; then
    vd3_prepare_only=1
    shift
fi

vd3_runtime="$("$vd3_python" - "$vd3_root" <<'PY'
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

root = Path(sys.argv[1]).resolve()
version = "vd3_curve_play_20261105"
registry = root / ".code_versions/versions_20261105.json"
entry = json.loads(registry.read_text())["versions"][version]
commit = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "--verify", "--end-of-options", entry["commit"] + "^{commit}"],
    text=True,
).strip()
runtime_parent = root / ".code_versions/runtime"
runtime_parent.mkdir(parents=True, exist_ok=True)
runtime = runtime_parent / version
with (runtime_parent / (version + ".lock")).open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    marker = runtime / ".code_version_commit"
    if runtime.exists() and (not marker.is_file() or marker.read_text().strip() != commit):
        raise SystemExit(f"Version export differs from registry: {runtime}; move it aside before rebuilding.")
    if not runtime.exists():
        staging = Path(tempfile.mkdtemp(prefix=version + ".", dir=runtime_parent))
        try:
            archive_path = staging / "code.tar"
            with archive_path.open("wb") as archive:
                subprocess.run(
                    ["git", "-C", str(root), "archive", "--format=tar", commit, "source", "scripts"],
                    stdout=archive, check=True,
                )
            with tarfile.open(archive_path) as archive:
                for member in archive.getmembers():
                    target = (staging / member.name).resolve()
                    if not target.is_relative_to(staging):
                        raise RuntimeError(f"Invalid archive path: {member.name}")
                    if member.issym() or member.islnk():
                        raise RuntimeError(f"Unexpected archive link: {member.name}")
                archive.extractall(staging)
            archive_path.unlink()
            for name in ("datasets", "logs", "test_ckpt"):
                (staging / name).symlink_to(root / name, target_is_directory=True)
            (staging / ".code_version_commit").write_text(commit + "\n")
            os.replace(staging, runtime)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
print(runtime)
PY
)"

echo "[VD3 curve Play] source: $vd3_runtime"
echo "[VD3 curve Play] checkpoint: $vd3_checkpoint"
if (( vd3_prepare_only )); then
    exit 0
fi
[[ -f "$vd3_checkpoint" ]] || { echo "Checkpoint not found: $vd3_checkpoint" >&2; exit 1; }
vd3_cache="${VD3_PLAY_CACHE:-/dev/shm/vd3_play_cache_$(id -u)}"
mkdir -p -- "$vd3_cache/cache" "$vd3_cache/tmp" "$vd3_cache/matplotlib" "$vd3_cache/assets"
export PYTHONPATH="$vd3_runtime/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$vd3_cache/cache" TMPDIR="$vd3_cache/tmp"
export MPLCONFIGDIR="$vd3_cache/matplotlib" VD3_ASSET_CACHE="$vd3_cache/assets"
cd -- "$vd3_runtime"
exec "$vd3_python" scripts/rsl_rl/play.py \
    --task Unitree-b2z1-Velocity-Play --num_envs 1 --device cuda:0 --real-time \
    --checkpoint "$vd3_checkpoint" \
    env.commands.base_velocity.rel_standing_envs=0.0 \
    "$@"
