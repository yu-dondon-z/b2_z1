"""Read-only TensorBoard/checkpoint review; writes only the requested report."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    events = EventAccumulator(str(args.run), size_guidance={"scalars": 0})
    events.Reload()
    report = {"run": str(args.run), "scalar_windows": {}, "warnings": [
        "Training scalars are episode/reset aggregates, not deterministic motion-quality metrics.",
        "Episode_Termination tags can be sticky snapshots; do not interpret as event rates.",
        "Reward improvement does not directly quantify millimetres of oscillation or Newton impact peaks.",
        "Source checkpoint is not serialized in final checkpoint metadata; default VD10 model_5998 provenance is an inference unless launch audit corroborates it.",
    ]}
    for tag in events.Tags()["scalars"]:
        samples = events.Scalars(tag)
        vals = np.asarray([x.value for x in samples])
        steps = np.asarray([x.step for x in samples])
        summary = {"count": len(samples), "first_step": int(steps[0]),
                   "last_step": int(steps[-1]), "nonfinite_count": int((~np.isfinite(vals)).sum()),
                   "last_value": float(vals[-1]), "windows": {}}
        if not tag.endswith("/time"):
            for start in range(0, 6000, 500):
                window = vals[(steps >= start) & (steps < start + 500)]
                if len(window):
                    summary["windows"][f"{start}-{start+499}"] = {
                        "mean": float(np.mean(window)), "std": float(np.std(window)),
                        "count": len(window)}
        report["scalar_windows"][tag] = summary
    checkpoint = torch.load(args.run / "model_5999.pt", map_location="cpu", weights_only=False)
    report["checkpoint"] = {
        "iter": checkpoint["iter"],
        "training_experiment_name": checkpoint.get("training_experiment_name"),
        "curriculum": checkpoint.get("vd5_curriculum_state"),
        "optimizer_lrs": [x["lr"] for x in checkpoint["optimizer_state_dict"]["param_groups"]],
        "model_nonfinite_elements": sum(int((~torch.isfinite(x)).sum())
            for x in checkpoint["model_state_dict"].values() if torch.is_tensor(x)),
    }
    report["initial_curriculum"] = json.loads((args.run / "vd5_initial_curriculum.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)
    for tag in ["Train/mean_reward", "Metrics/ee_target/position_error",
                "Metrics/ee_target/gun_axis_error", "Metrics/ee_target/arm_joint_velocity_rms",
                "Episode_Reward/vd11_world_tcp_highpass_200hz",
                "Episode_Reward/vd11_foot_impact_200hz", "Policy/mean_noise_std"]:
        summary = report["scalar_windows"][tag]
        print(tag, [(k, round(v["mean"], 6)) for k, v in summary["windows"].items()])


if __name__ == "__main__":
    main()
