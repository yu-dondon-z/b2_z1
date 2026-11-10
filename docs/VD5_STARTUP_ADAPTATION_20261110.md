# VD5 Startup Adaptation - 2026-11-10

## Status

Completed a short, isolated startup-adaptation run from the VD5 world-seam checkpoint. This is a simulation candidate, not a deployment release and not evidence of real-robot readiness.

## Training

- Source: `logs/rsl_rl/b2_z1_vd5_world_seam_amp/2026-11-10_00-02-22_vd5_world_seam/model_3250.pt` (`iter=3250`).
- New experiment: `logs/rsl_rl/b2_z1_vd5_startup_amp/2026-11-10_09-56-37_vd5_startup_height050_pilot_20261110/`.
- Adaptation completed 250 iterations, ending at `model_3499.pt`; source checkpoint and original experiment were not overwritten.
- Reset mixture: 50% AMP reference initialization and 50% nominal standing initialization.
- Nominal base height: 0.50 m; nominal leg reset velocities: zero.
- Startup transition: 2.0 s blend-in for base velocity and end-effector target.
- Restored VD5 curriculum: curve level 0.10, ground-target blend 0.84048, seam trailing distance 0.70 m.
- Training log: `outputs/vd5_startup_20261110/training_height050_pilot.log`.

## Evaluation

All evaluations used the same startup task, image input disabled, curve level 0.10, ground blend frozen to 0.84048, and 1,600 control steps (32 simulated seconds). The source actor was evaluated in the startup task without adaptation; this provides a short matched baseline.

| Reset distribution | Actor | Environments | Terminations | TCP horizontal error P95 | TCP 3D error P95 | Mean / max tilt |
|---|---|---:|---|---:|---:|---:|
| Nominal standing | Source `model_3250.pt` | 4 | 4 timeouts; no other terminations | 1.33 cm | 1.51 cm | 1.15 / 2.89 deg |
| Nominal standing | Adapted `model_3499.pt` | 4 | 4 timeouts; no other terminations | 1.45 cm | 1.73 cm | 0.96 / 2.50 deg |
| Mixed (nominal/reference) | Source `model_3250.pt` | 8 | 8 timeouts; no other terminations | 1.32 cm | 1.60 cm | 1.18 / 18.27 deg |
| Mixed (nominal/reference) | Adapted `model_3499.pt` | 8 | 8 timeouts; no other terminations | 1.33 cm | 1.73 cm | 0.96 / 14.76 deg |

Reports: `outputs/vd5_startup_20261110/source_nominal_32s.json`, `outputs/vd5_startup_20261110/final_nominal_32s.json`, `outputs/vd5_startup_20261110/source_mixed_32s.json`, and `outputs/vd5_startup_20261110/final_mixed_32s.json`.

The adapted actor did not materially improve TCP error in these short paired probes. Its mean tilt and worst observed tilt were lower, but both actors completed all episodes without termination and the mixed-run max tilt remains notable. Training-window statistics also showed occasional base-contact termination (about 5.5% in one late window), so these finite probes do not establish convergence or robust startup performance across seeds.

## Recommendation

Keep `model_3499.pt` as a separate candidate for further evaluation, but do not treat the 250-iteration run as a demonstrated performance upgrade. Run longer multi-seed startup evaluations and compare against the source checkpoint on identical nominal and mixed reset seeds. Do not replace the original VD5 checkpoint or synchronize this candidate to the robot based only on these probes.
