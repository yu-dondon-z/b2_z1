#!/usr/bin/env python3
"""Launch isolated Z1 or Huibo dual-camera Isaac playback (no hardware)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from play_va15_vision import stop_child


TASK = "Unitree-b2z1-AMP-VA16-DualVision-Play"
HUIBO_TASK = "Unitree-b2huibo-AMP-DualVision-Play"


def source_manifest(repo: Path) -> dict:
    """Hash relevant source only; weights are recorded separately."""
    paths = [repo / name for name in (
        "scripts/vision/play_va16_dual_vision.py", "scripts/vision/play_va15_vision.py",
        "scripts/vision/run_dual_camera_bridge.py", "scripts/vision/dual_camera_perception.py",
        "scripts/vision/evaluate_va16_dual_run.py", "scripts/rsl_rl/play_amp.py",
        "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/b2_z1_va16_dual_vision_env_cfg.py",
        "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/b2_z1_rear_vision_env_cfg.py",
        "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/b2_z1_va_env_cfg.py",
        "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/agents/rsl_rl_amp_cfg.py",
        "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/commands/dual_vision_command.py",
        "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/commands/dual_vision_contract.py",
        "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/commands/dual_vision_rewards.py",
        "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/commands/dual_vision_kinematics.py",
        "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/commands/dual_camera_visuals.py",
        "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/commands/vision_trajectory_command.py",
        "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/commands/path_guidance.py",
        "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/commands/ee_vel_command.py",
        "source/b2_z1/b2_z1/assets/b2_z1/b2_z1_no_stick.urdf",
        "source/b2_z1/b2_z1/assets/b2_z1/b2_huibo.urdf",
        "source/b2_z1/b2_z1/assets/b2_huibo.py",
        "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/b2_huibo_dual_vision_env_cfg.py",
    )]
    external_root = Path("/home/dong/RHACrackNet/code")
    for package in ("crack_inference", "crack_trajectory", "models/RHANet_dep"):
        paths.extend(sorted((external_root / package).rglob("*.py")))
    manifest = {}
    for path in paths:
        try:
            label = str(path.relative_to(repo))
        except ValueError:
            label = f"external:{path}"
        try:
            content = path.read_bytes()
            manifest[label] = {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
        except OSError as error:
            manifest[label] = {"missing": True, "reason": str(error)}
    return manifest


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("z1", "huibo"), default="z1")
    parser.add_argument("--checkpoint", type=Path, help="Huibo requires its own trained 73-observation/19-action checkpoint")
    parser.add_argument("--isaac-python", type=Path, default=Path("/home/dong/anaconda3/envs/env_isaaclab/bin/python"))
    parser.add_argument("--vision-python", type=Path, default=Path("/home/dong/anaconda3/envs/crack_vision/bin/python"))
    parser.add_argument("--vision-device", default="cuda:0")
    parser.add_argument("--vision-mode", choices=("rhanet", "classical"), default="rhanet")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--episode-length-s", type=float)
    parser.add_argument("--output-root", type=Path, default=repo / "outputs/va16_dual_vision")
    parser.add_argument("--dry-run", action="store_true", help="Write and print this session's commands without starting processes")
    args = parser.parse_args()
    task = HUIBO_TASK if args.robot == "huibo" else TASK
    if args.checkpoint is None:
        if args.robot == "huibo":
            parser.error("--robot huibo requires --checkpoint for a trained Huibo policy; VA15 is incompatible")
        args.checkpoint = repo / "logs/rsl_rl/b2_z1_amp/2026-09-28_15-07-31_va15_curriculum_from2500/model_5000.pt"
    for path in (args.checkpoint, args.isaac_python, args.vision_python):
        if not path.expanduser().is_file():
            parser.error(f"Missing file: {path}")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.episode_length_s is not None and (not math.isfinite(args.episode_length_s) or args.episode_length_s <= 0):
        parser.error("--episode-length-s must be finite and positive")
    args.output_root.expanduser().mkdir(parents=True, exist_ok=True)
    exchange = Path(tempfile.mkdtemp(prefix=time.strftime("run_%Y%m%d_%H%M%S_"), dir=args.output_root.expanduser().resolve()))
    env = os.environ.copy()
    env.update(B2_Z1_VISION_EXCHANGE=str(exchange), B2_Z1_VISION_ALL_CAMERAS="0",
               B2_Z1_DUAL_VISION_EXCHANGE=str(exchange), PYTHONUNBUFFERED="1", HYDRA_FULL_ERROR="1")
    for inherited in ("B2_Z1_VISION_TRAJECTORY", "B2_Z1_DUAL_VISION_TRAJECTORY"):
        env.pop(inherited, None)
    vision_cmd = [str(args.vision_python.expanduser()), str(repo / "scripts/vision/run_dual_camera_bridge.py"),
                  "--exchange-dir", str(exchange), "--device", args.vision_device,
                  "--mode", args.vision_mode, "--save-overlay"]
    if not args.headless:
        vision_cmd.append("--display")
    play_cmd = [str(args.isaac_python.expanduser()), str(repo / "scripts/rsl_rl/play_amp.py"),
                "--task", task, "--checkpoint", str(args.checkpoint.expanduser().resolve()),
                "--num_envs", "1", "--enable_cameras", "--real-time",
                "--motion-trace-csv", str(exchange / "motion_trace.csv"),
                "--policy-export-dir", str(exchange / "policy_export")]
    if args.headless:
        play_cmd.append("--headless")
    if args.max_steps is not None:
        play_cmd.extend(["--max-steps", str(args.max_steps)])
    if args.episode_length_s is not None:
        play_cmd.append(f"env.episode_length_s={args.episode_length_s}")
    print(f"[VA16] task: {task}\n[VA16] checkpoint: {args.checkpoint}\n[VA16] session and logs: {exchange}", flush=True)
    print("[VA16] working speed command=0.20 m/s; front RGB steering; rear RGB-D TCP measurements.", flush=True)
    print(f"[VA16] vision mode={args.vision_mode}; this entry starts local simulation only.", flush=True)
    checkpoint_hash = hashlib.sha256()
    with args.checkpoint.expanduser().open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            checkpoint_hash.update(block)
    starting_sources = source_manifest(repo)
    session = {"task": task, "robot": args.robot, "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash.hexdigest(),
        "source_sha256_at_launch": starting_sources,
        "source_hash_scope": "listed simulation/control/perception source and URDF; external weights not re-read",
        "created_timestamp": time.time(),
        "phase": ("Huibo checkpoint evaluation; quality must be measured independently" if args.robot == "huibo"
                  else "VA15 outer-loop baseline; dual-camera policy has not been retrained"),
        "play": play_cmd, "vision": vision_cmd, "vision_mode": args.vision_mode,
        "baseline_only": args.robot == "z1", "hardware_started": False}
    (exchange / "session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"play": play_cmd, "vision": vision_cmd}, indent=2), flush=True)
        return
    vision = play = None

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        with (exchange / "vision.log").open("w") as vision_log, (exchange / "play.log").open("w") as play_log:
            vision = subprocess.Popen(vision_cmd, cwd=repo, env=env, stdout=vision_log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            # Warm-up occurs before Isaac begins exporting images, so it cannot
            # turn the first old capture into a seemingly fresh observation.
            startup = time.monotonic()
            while not (exchange / "vision_ready.json").exists():
                if vision.poll() is not None:
                    raise RuntimeError(f"Vision startup exited ({vision.returncode}); see {exchange / 'vision.log'}")
                if time.monotonic() - startup > 120:
                    raise RuntimeError(f"Vision startup timed out; see {exchange / 'vision.log'}")
                time.sleep(.1)
            play = subprocess.Popen(play_cmd, cwd=repo, env=env, stdout=play_log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            status_key = None
            while play.poll() is None:
                if vision.poll() is not None:
                    raise RuntimeError(f"Vision exited ({vision.returncode}); see {exchange / 'vision.log'}")
                try:
                    status = json.loads((exchange / "latest_vision_status.json").read_text())
                    current = (status.get("valid"), status.get("reason"))
                    if current != status_key:
                        print(f"[VA16 vision] {'VALID' if current[0] else 'WAIT'}: {current[1]}", flush=True)
                        status_key = current
                except (OSError, ValueError):
                    pass
                time.sleep(.25)
            if play.returncode:
                raise RuntimeError(f"Simulation exited ({play.returncode}); see {exchange / 'play.log'}")
    except KeyboardInterrupt:
        print("[VA16] Stopping this session.", flush=True)
    finally:
        stop_child(play)
        stop_child(vision)
        ending_sources = source_manifest(repo)
        changed = [path for path in set(starting_sources) | set(ending_sources)
                   if starting_sources.get(path) != ending_sources.get(path)]
        session.update(source_sha256_at_finish=ending_sources, source_files_changed_during_session=sorted(changed),
                       finished_timestamp=time.time())
        (exchange / "session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
        print(f"[VA16] Logs and camera images retained: {exchange}", flush=True)


if __name__ == "__main__":
    main()
