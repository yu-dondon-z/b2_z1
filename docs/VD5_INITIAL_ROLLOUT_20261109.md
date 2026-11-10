# VD5 initial rollout — 2026-11-09

## Status

Historical failed baseline, superseded by `VD5_WORLD_SEAM_COMMANDS_20261109.md`.
The checkpoint used here was subsequently identified as a previously rejected gun-support
policy. The deployed VD3 mirror model_2000 is the new warm-start; keep this report as evidence.

Do not start long training yet. This is a failed transfer baseline, not a validated VD5 policy.
No robot processes were launched and no onboard files were synchronized.

## Procedure

- Isaac Lab VD5 training task, four environments, 600 policy steps per environment.
- VD3 model_18000 actor, deterministic ELU inference; no optimizer or AMP state loaded.
- Seed 42, observation noise and external pushes disabled; camera assets retained, no images consumed.
- Existing terminations retained; samples captured before automatic resets.
- Headless physics rollout, not rendered-camera or GUI validation.

Report: `outputs/vd5_probe_20261109/reset_fixed.json`.

## Findings

- TCP 3D error: mean 0.1274 m, P95 0.2128 m, max 0.2449 m, including startup.
- Base tilt: P95 9.38 degrees, max 19.51 degrees.
- The original reported counts (base contact 1146, tool support 1154) are invalid: the probe
  counted Isaac Lab's retained last-episode attribution on every later step. The probe now
  masks it with actual resets. These old numbers are neither condition counts nor episode counts.
- Frequent early terminations prevent assessment of sustained walking or path-resampling continuity.
- Ground targets can start around base-frame z=-0.70 m, unlike the old nominal z=-0.30 m.
  This establishes a distribution mismatch, not a proof of kinematic unreachability.

The EE resample hook now refreshes the seam target immediately for resetting environments,
instead of temporarily returning the inherited pose-start command. The probe excludes the first
two episode steps from target-jump statistics; small measured jumps therefore do not prove
that path changes are continuous over a full episode.

## Next Training Decision

The decision at the time was to prefer VD3 actor warm-start over training from scratch, but first establish reachability for the
ground TCP target and a usable initialization/transition curriculum. Also preserve rear seam
history across base path replacement. The current VD5 instead retains one seam for the whole
episode, avoiding that replacement entirely. See the current report for reachability and curriculum
validation; do not use this historical checkpoint as the training default.

## Reproduce

```bash
cd /home/dong/b2_z1_VD2
PYTHONPATH="$PWD/source/b2_z1" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/probe_vd5_commands.py \
  --checkpoint logs/rsl_rl/b2_z1_vd3_curve_amp/2026-11-04_09-44-59_vd3_zero_gun_flange_stability/model_18000.pt \
  --output outputs/vd5_probe_20261109/recheck.json --steps 600 --num_envs 4 --device cuda:0 --headless
```
