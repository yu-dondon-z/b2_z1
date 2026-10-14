"""Own one VB12 training process; stop on nonfinite data or motion starvation."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def guard_reason(series, warmup=200, window=50):
    for tag, samples in series.items():
        if any(not math.isfinite(value) for step, value in samples):
            return f"Nonfinite TensorBoard scalar: {tag}"
    movement = series.get("VB12/movement_given_request", [])
    if len(movement) < max(warmup, window):
        return None
    mean = lambda tag: sum(v for _, v in series.get(tag, [])[-window:]) / window
    required = ["VB12/requested_motion_fraction", "VB12/scheduled_motion_fraction"]
    if any(len(series.get(t, [])) < window for t in required):
        return None
    requested = mean(required[0])
    scheduled = mean(required[1])
    if scheduled > .20 and requested < .05:
        return "Scheduled motion exists but commands disappeared: stage/command starvation"
    if requested > .20 and mean("VB12/movement_given_request") < .15:
        return "Requested motion is present but actual forward movement collapsed"
    full_tags = ["VB12/full_load_scheduled_motion_fraction", "VB12/full_load_requested_motion_fraction",
                 "VB12/full_load_actual_commanded_motion_fraction"]
    if all(len(series.get(t, [])) >= window for t in full_tags):
        full_scheduled, full_requested, full_actual = [mean(t) for t in full_tags]
        if full_scheduled > .20 and full_requested < .05:
            return "Nominal full-load motion commands disappeared; light domains cannot mask this failure"
        if full_requested > .05 and full_actual / full_requested < .15:
            return "Nominal full-load actual movement collapsed; light domains cannot mask this failure"
    contact_tag = "Episode_Termination/gun_contact_budget"
    if len(series.get(contact_tag, [])) >= window and mean(contact_tag) > .85:
        return "Persistent gun-contact budget failures exceed 85% of completed episodes"
    return None


def stop_owned(process):
    if process.poll() is not None:
        return
    for sig, timeout in [(signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 2)]:
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=timeout)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


def read_series(accumulator):
    accumulator.Reload()
    return {tag: [(s.step, s.value) for s in accumulator.Scalars(tag)]
            for tag in accumulator.Tags()["scalars"]}


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--run-name", default="VB12_schedule_fromVB10_12300")
    parser.add_argument("--guard-warmup", type=int, default=200)
    parser.add_argument("--guard-window", type=int, default=50)
    args, hydra = parser.parse_known_args()
    if min(args.num_envs, args.iterations, args.guard_window) <= 0 or args.guard_warmup < 0:
        parser.error("Need positive sizes and nonnegative guard warmup")
    donor = repo / "logs/rsl_rl/b2_z1_amp/2026-10-13_10-35-50_VB10_clearance_fromVB9_10999/model_12300.pt"
    if not donor.is_file():
        parser.error(f"Missing screened donor: {donor}")
    audit = repo / "outputs" / f"vb12_guard_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    audit.mkdir(parents=True, exist_ok=False)
    log_path = audit / "training.log"
    command = [sys.executable, str(repo / "scripts/rsl_rl/train_amp.py"),
               "--task", "Unitree-b2z1-AMP-VB12-Schedule", "--headless",
               "--num_envs", str(args.num_envs), "--max_iterations", str(args.iterations),
               "--resume", "--reset_optimizer", "--load_run", donor.parent.name,
               "--checkpoint", donor.name, "--run_name", args.run_name,
               "agent.amp_cfg.amp_num_preload_transitions=20000", *hydra]
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", HYDRA_FULL_ERROR="1",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    print(f"[VB12 guard] Training log: {log_path}", flush=True)
    print(f"[VB12 guard] New iterations: {args.iterations}; donor: {donor}", flush=True)
    reason = None
    log_dir = None
    accumulator = None
    child = None
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        with log_path.open("w") as log:
            child = subprocess.Popen(command, cwd=repo, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                # Discover only THIS child's log directory, never another
                # concurrently running training's most-recent event file.
                if log_dir is None:
                    match = re.search(r"source code is saved to\s+([^\n]+?)/source.tar.gz", log_path.read_text(errors="replace"))
                    if match:
                        log_dir = Path(match.group(1).strip())
                        accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
                if accumulator is not None:
                    series = read_series(accumulator)
                    reason = guard_reason(series, args.guard_warmup, args.guard_window)
                    if reason:
                        print(f"[VB12 guard] STOP: {reason}", flush=True)
                        stop_owned(child)
                        break
                    samples = series.get("VB12/movement_given_request", [])
                    if samples:
                        latest = samples[-1]
                        print(f"[VB12 guard] iteration={latest[0]} movement_given_request={latest[1]:.3f}", flush=True)
                time.sleep(20)
            # Also inspect the final flushed events. A short run may finish
            # between polling ticks; it must not silently skip monitoring.
            if log_dir is None:
                match = re.search(r"source code is saved to\s+([^\n]+?)/source.tar.gz", log_path.read_text(errors="replace"))
                if match:
                    log_dir = Path(match.group(1).strip())
                    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
            if accumulator is not None and reason is None:
                series = read_series(accumulator)
                reason = guard_reason(series, args.guard_warmup, args.guard_window)
                if not series and child.returncode == 0:
                    reason = "Training exited without any TensorBoard scalars; monitoring was not validated"
                elif "VB12/movement_given_request" not in series and child.returncode == 0:
                    reason = "Missing VB12 movement telemetry; monitoring was not validated"
    except KeyboardInterrupt:
        reason = "Operator interruption"
    except Exception as error:
        reason = f"Monitor/training launch error: {type(error).__name__}: {error}"
    finally:
        if child is not None:
            stop_owned(child)
        (audit / "status.json").write_text(json.dumps(
            {"command": command, "training_dir": str(log_dir) if log_dir else None,
             "stop_reason": reason, "returncode": child.returncode if child else None}, indent=2))
    if reason:
        raise SystemExit(f"Stopped this training only: {reason}. Inspect {audit}")
    if child.returncode:
        raise SystemExit(f"Training failed ({child.returncode}); inspect {log_path}")
    print(f"[VB12 guard] Completed requested training. Not a deployment acceptance: {log_dir}", flush=True)


if __name__ == "__main__":
    main()
