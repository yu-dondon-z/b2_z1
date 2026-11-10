# VD2 low-speed candidate, 2026-11-03

## Status

This is a local simulation-only training candidate, not a validated deployment
policy. The original VD1 and the separate B2/Huibo workspace are unchanged. This
worktree is based on upstream `dual_cmd` commit `b64b0faf8b4db3a9c1c2e05235521f61ab4dc47d`.
Its code changes are currently uncommitted on branch `VD2`.

The actor retains the VD1 70-input, 18-output contract. Its base velocity
commands now sample fixed 0.2 m/s walking or a zero-velocity standing command
with probability 0.2 on each 3-8 second command resample. Tracking reward
widths were reduced, and the training was warm-started from the original VD1
checkpoint with a fresh optimizer. The subsequent resume includes optimizer
state and restores the PPO adaptive learning-rate scalar consistently.
The 0.2 m/s command does not guarantee that actual speed is 0.2 m/s.

The original upstream training environment uses ideal geometry to emulate the
front camera. Its rear-arm command interpolation starts from the simulated
true end-effector pose, not an RGBD measurement. True state is also used in
training rewards and critic observations. Therefore this training is **not** a
privilege-free RGB/RGBD sensor-loop validation, and the actor must not be
described as trained end-to-end on the two camera images. The IMU yaw-rate
feedback controller and independent front/rear visual failure handling from the
diagnosis are not implemented here. The terminal 5 cm projected TCP-to-seam
accuracy has not been tested.

## Training and tests

256 Isaac Sim environments, 150 warm-start PPO iterations followed by 500
resumed iterations. Logs, full checkpoints, source snapshots, and parameters
are under `logs/rsl_rl/b2_z1_vd2_curve_amp/2026-11-03_09-57-14/` and
`logs/rsl_rl/b2_z1_vd2_curve_amp/2026-11-03_10-01-54/`. A 32-environment,
one-iteration smoke run preceded these. The full CPU test suite passed: 68
tests, including standing and selective command resampling.

MuJoCo open-loop diagnostic: same initial keyframe and recorded simulated tool
target for each checkpoint; 0, 0.2, and 0.4 m/s commands, zero yaw command,
2 s ramp; true world position is used **only for reporting**. Values below are
world X mean velocity and world Y displacement from 2 to 10 s, not body-frame
velocity or seam error. The source trace was retained in
`logs/vd2_mujoco_diagnostics/vd1_actor_trace.jsonl.gz`.

| Policy / target time | 0 command X m/s | 0.2 command X m/s | 0.2 command Y m / 8 s |
| --- | ---: | ---: | ---: |
| VD1 / 4 s | 0.227 | 0.460 | +0.390 |
| VD2 250 / 4 s | 0.085 | 0.146 | -0.126 |
| VD2 450 / 4 s | -0.019 | 0.171 | -0.207 |
| VD2 450 / 10 s | -0.011 | 0.191 | -0.145 |
| VD2 550 / 4 s | -0.013 | 0.241 | -0.064 |
| VD2 550 / 10 s | 0.069 | 0.240 | -0.116 |
| VD2 600 / 4 s | -0.009 | 0.236 | -0.085 |
| VD2 600 / 10 s | -0.012 | 0.237 | -0.108 |
| VD2 648 / 4 s | -0.003 | -0.005 | +0.033 |

VD2 100 tilted before completion under zero and 0.2 m/s. VD2 149 completed,
but its 0.2 m/s test had X speed 0.314 m/s and Y displacement -0.549 m.
Results are limited to two sampled target poses, one initial robot state and
straight commands. Model 600 is a provisional candidate for further simulation,
not a best-policy claim. Model 648 stopped in this 0.2 m/s test despite the
nonzero command. The JSON results are in `logs/vd2_mujoco_diagnostics/`.

For convenience, `logs/vd2_mujoco_diagnostics/vd2_model_600_candidate_actor.pt`
is a TorchScript actor extracted from model 600. Random-input comparison with
the checkpoint actor had max absolute difference 0.0. It is not a deployment
release.

Reproduce a straight-line diagnostic from this directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/dong/anaconda3/envs/crack_vision/bin/python scripts/evaluate_vd2_mujoco.py \
  --checkpoint logs/rsl_rl/b2_z1_vd2_curve_amp/2026-11-03_10-01-54/model_600.pt \
  --trace logs/vd2_mujoco_diagnostics/vd1_actor_trace.jsonl.gz --sample-time 10
```

## Curriculum checkpoint repair and continuation

New VD2 training checkpoints contain `vd2_curriculum_state` version 1: the
curve offset scale, all six EE target ranges, and the elapsed control steps
since each curriculum check. Resume validates these values against the current
configuration, restores the timers relative to the new environment counter,
and resets the environments so initial commands use the restored difficulty.
This restores curriculum progression, not simulator/RNG/rollout state.
Policy loading for play and warm start continues to leave curricula alone.

Legacy checkpoints, including the original `model_600.pt`, did not save this
state. Resume now rejects these unless both legacy approximation options are
explicitly supplied. TensorBoard reports a curve offset scale of approximately
0.06 at iteration 600. This value is a dimensionless trajectory parameter,
**not curvature in 1/m**. EE `pos_x` had reached its upper limit -0.96, showing
at least five expansions, but cannot reveal whether `pos_y` expanded further.
The conservative five-expansion approximation below initializes `pos_x` to
[-1.08, -0.96], `pos_y` to [-0.07, 0.07], `pos_z` to [-0.34, -0.26], leaves
orientation fixed, and waits a full episode interval before another upgrade
check. It is not an exact recovery of the old training state.

Continue from the original model 600 for **5000 additional PPO iterations**:

```bash
cd /home/dong/b2_z1_VD2
PYTHONPATH=/home/dong/b2_z1_VD2/source/b2_z1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Unitree-b2z1-Velocity --headless --num_envs 256 \
  --resume --load_run 2026-11-03_10-01-54 --checkpoint model_600.pt \
  --vd2_legacy_curve_level 0.06 --vd2_legacy_position_steps 5 \
  --max_iterations 5000 --run_name vd2_curriculum_resume
```

Use the explicit `PYTHONPATH`: the environment's editable installation points
at `/home/dong/b2_z1`, so changing directory alone does not select VD2 code.
The existing runner repeats the saved iteration label on resume; the command
above executes 5000 new updates labeled 600 through 5599 in a new log folder.
It does not overwrite the original run.

For later checkpoints created by this repaired code, select the new run and
checkpoint and **omit both legacy options**; the state is restored from the
checkpoint. Supplying legacy options for a checkpoint with saved state raises
an error to prevent accidental override. Checkpoints remain saved every 50
iterations and at normal completion. An abrupt process kill can lose work
since the most recent saved checkpoint.

The curve curriculum increases the offset scale by 0.01 to its configured
limit 0.10 when both linear and angular velocity tracking rewards exceed 80%
of their configured weights. It checks at most once per episode interval at
episode reset; iteration count alone does not trigger an upgrade. A longer run
does not guarantee that the reward thresholds will be met.

Validation of this repair:

- CPU suite: 78 tests and 111 subtests passed with pytest plugin autoload
  disabled to avoid unrelated ROS plugins.
- Two actual Isaac Sim runs, each 16 environments and one PPO update, passed.
  `2026-11-03_10-29-03_curriculum_smoke_legacy/model_600.pt` saves the explicit
  legacy approximation and 24 elapsed steps. Resuming it without legacy
  arguments creates
  `2026-11-03_10-30-09_curriculum_smoke_exact/model_600.pt`, preserving curve
  scale 0.06 and all six ranges, and advancing both elapsed timers to 48.
  These are checkpoint tests, not recommended new policy candidates.
- Console logs are `/tmp/vd2_curriculum_smoke_legacy_v2_20261103.log` and
  `/tmp/vd2_curriculum_smoke_exact_20261103.log`; full run artifacts are
  retained under the normal experiment directory. An earlier failed launch
  without the VD2 `PYTHONPATH` is also retained as
  `/tmp/vd2_curriculum_smoke_legacy_20261103.log`.
- The original checkpoint and all existing logs were preserved. No long
  training job, board synchronization, or robot control was started.

## Remaining work

1. Replace simulator-truth-derived arm command inputs with a realizable rear
   RGBD measurement/target pipeline, and validate the front RGB processing with
   actual rendered images rather than ideal projected curve geometry.
2. Evaluate measured yaw-rate behavior and implement/test any IMU feedback and
   timestamp handling against delay, dropped frames, and curved seams. Preserve
   honest sensor validity and separate the front/rear dependencies.
3. Test stop/restart, yaw-rate commands, multiple initial states, seeds and tool
   targets, then measure foot clearance and projected TCP-to-seam error against
   ground truth used only for evaluation. Do not claim a 5 cm result without it.
4. Only after simulation validation and explicit authorization consider board
   synchronization or robot motion. Neither was performed here.
