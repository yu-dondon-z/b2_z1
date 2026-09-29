#!/usr/bin/env python3
"""Own one isolated VA15 camera/segmentation playback session."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time


def stop_child(process):
    if process is None or process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 8), (signal.SIGTERM, 4), (signal.SIGKILL, 2)):
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=timeout)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(repo / "logs/rsl_rl/b2_z1_amp/2026-09-28_15-07-31_va15_curriculum_from2500/model_5000.pt"))
    parser.add_argument("--isaac-python", default="/home/dong/anaconda3/envs/env_isaaclab/bin/python")
    parser.add_argument("--vision-python", default="/home/dong/anaconda3/envs/crack_vision/bin/python")
    parser.add_argument("--vision-device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    for path in (args.checkpoint, args.isaac_python, args.vision_python):
        if not Path(path).is_file():
            parser.error(f"Missing file: {path}")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    exchange = Path(tempfile.mkdtemp(prefix="b2_z1_va15_vision_"))
    env = os.environ.copy()
    env.update(B2_Z1_VISION_EXCHANGE=str(exchange), PYTHONUNBUFFERED="1", HYDRA_FULL_ERROR="1")
    # Do not inherit an old session's independently overridden trajectory file.
    env.pop("B2_Z1_VISION_TRAJECTORY", None)
    env["B2_Z1_VISION_ALL_CAMERAS"] = "0"
    vision_cmd = [args.vision_python, str(repo / "scripts/vision/run_rhacracknet_bridge.py"),
                  "--exchange-dir", str(exchange), "--device", args.vision_device,
                  "--save-overlay", "--depth-shadow-filter"]
    if not args.headless:
        vision_cmd.append("--display")
    play_cmd = [args.isaac_python, str(repo / "scripts/rsl_rl/play_amp.py"),
                "--task", "Unitree-b2z1-AMP-VA15-Vision-Play", "--checkpoint", args.checkpoint,
                "--num_envs", "1", "--enable_cameras", "--real-time"]
    if args.headless:
        play_cmd.append("--headless")
    if args.max_steps is not None:
        play_cmd.extend(["--max-steps", str(args.max_steps)])
    print(f"[VA15 vision] checkpoint: {args.checkpoint}\n[VA15 vision] session/logs: {exchange}", flush=True)
    (exchange / "session.json").write_text(json.dumps({"checkpoint": args.checkpoint, "play": play_cmd, "vision": vision_cmd}, indent=2))
    vision = play = None
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        with (exchange / "vision.log").open("w") as vision_log, (exchange / "play.log").open("w") as play_log:
            vision = subprocess.Popen(vision_cmd, cwd=repo, env=env, stdout=vision_log, stderr=subprocess.STDOUT, start_new_session=True)
            play = subprocess.Popen(play_cmd, cwd=repo, env=env, stdout=play_log, stderr=subprocess.STDOUT, start_new_session=True)
            last_status = None
            while play.poll() is None:
                if vision.poll() is not None:
                    raise RuntimeError(f"Vision bridge exited ({vision.returncode}); see {exchange / 'vision.log'}")
                try:
                    status = json.loads((exchange / "latest_vision_status.json").read_text())
                    status_key = (status.get("valid"), status.get("reason"))
                    if status_key != last_status:
                        print(f"[Vision] {'VALID' if status.get('valid') else 'WAIT'}: {status.get('reason', '')}", flush=True)
                        last_status = status_key
                except (OSError, ValueError):
                    pass
                time.sleep(0.25)
            if play.returncode:
                raise RuntimeError(f"Playback exited ({play.returncode}); see {exchange / 'play.log'}")
    except KeyboardInterrupt:
        print("[VA15 vision] Stopping this session.", flush=True)
    finally:
        stop_child(play)
        stop_child(vision)
        print(f"[VA15 vision] Stopped. Logs and last camera images retained: {exchange}", flush=True)


if __name__ == "__main__":
    main()
