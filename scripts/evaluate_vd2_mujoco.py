#!/usr/bin/env python3
"""Evaluate trusted VD2 checkpoints with the existing VD1 MuJoCo speed diagnostic.

World position is used only to report evaluation results, never as actor input.
This is an open-loop velocity test, not a camera or hardware validation.
"""

import argparse
import json
from pathlib import Path
import sys
import tempfile

import torch

VD1_SCRIPTS = Path("/home/dong/b2_z1/scripts/vision")
sys.path.insert(0, str(VD1_SCRIPTS))
from diagnose_vd1_speed import build_scene, run_case, tool_pose_from_trace


def load_actor(checkpoint: Path):
    # Only load checkpoints produced by our own local training process.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    actor = torch.nn.Sequential(
        torch.nn.Linear(70, 512), torch.nn.ELU(),
        torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 18),
    )
    actor.load_state_dict({name[len("actor."):]: value for name, value in state.items()
                           if name.startswith("actor.")})
    return actor.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--sample-time", type=float, default=4.0)
    parser.add_argument("--export", type=Path, help="Optional TorchScript export path")
    parser.add_argument("--output", type=Path, help="Optional new JSON results file")
    args = parser.parse_args()
    torch.set_num_threads(1)
    actor = load_actor(args.checkpoint)
    if args.export:
        if args.export.exists():
            parser.error("refusing to overwrite existing export")
        torch.jit.script(actor).save(str(args.export))
    pose, sample_time = tool_pose_from_trace(args.trace, args.sample_time)
    cache = Path(tempfile.mkdtemp(prefix="vd2_speed_diagnostic_", dir="/dev/shm"))
    scene = build_scene(0.0, cache, timestep=0.005)
    cases = [run_case(scene, actor, pose, speed, 600) for speed in (0.0, 0.2, 0.4)]
    result = json.dumps({
        "checkpoint": str(args.checkpoint), "trace": str(args.trace),
        "tool_pose_sample_sim_time_s": sample_time, "cases": cases,
        "note": "Open-loop evaluation only; world position is reporting-only.",
    }, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(result + "\n")
    print(result)
    return int(any(case["stop"] != "completed" for case in cases))


if __name__ == "__main__":
    sys.exit(main())
