"""Shared motion files and explicit dataset ordering for the B2 locomotion prior."""

from pathlib import Path

# Motions are stored per leg, each with (hip, thigh, calf); the arm is excluded.
AMP_LEG_NAMES = ("FL", "FR", "RL", "RR")
AMP_JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in AMP_LEG_NAMES for joint in ("hip", "thigh", "calf")]
AMP_FOOT_NAMES = [f"{leg}_foot" for leg in AMP_LEG_NAMES]
MOTION_FILES = sorted(str(path) for path in (Path(__file__).resolve().parents[3] / "datasets/b2_motion").glob("*.txt"))
