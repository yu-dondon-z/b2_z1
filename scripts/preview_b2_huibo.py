#!/usr/bin/env python3
"""Inspect the B2 + Huibo asset, cameras and contact tool without a policy.

The simulated robot holds its default pose with gravity disabled. This is
model inspection, not locomotion or evidence of a trained Huibo policy.
"""

import datetime
import os
from pathlib import Path
import sys
import uuid


def main():
    root = Path(__file__).resolve().parents[1]
    arguments = sys.argv[1:]
    output = root / "outputs" / (
        "b2_huibo_preview_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_" + uuid.uuid4().hex[:6]
    )
    command = [sys.executable, str(root / "scripts/vision/check_va16_dual_scene.py"),
               "--task", "Unitree-b2huibo-AMP-DualVision-Play", "--mount-only"]
    if "--output" not in arguments:
        command += ["--output", str(output)]
    if "--headless" not in arguments:
        command += ["--viewport-screenshot", "--hold"]
    os.chdir(root)
    os.execv(sys.executable, command + arguments)


if __name__ == "__main__":
    main()
