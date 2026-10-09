# VB7 load-robust long continuation

Preserve VB6 playback; branch VB7 adds a separate training task. Start from
VB4 model_5199.pt, not the two-iteration smoke-test checkpoint.

## Domains

- 25% of environments: 0.05 kg light-tool proxy, proportional inertia,
  zero hose force/drag. Gun geometry and COM location remain; NOT literal
  physical removal or proof that the real Z1 can operate without a gun.
- Other environments: gun mass uniform 0.5–4.2 kg at startup, inertia
  proportional to mass. Never change mass abruptly within an episode.
- Hose mass uniform 0–6 kg per episode; an additional 25% zero-hose mixture.
- Hose drag uniform 0–6 N*s/m; zero in unloaded cases. Effective weight,
  filtered acceleration reaction and offset torque remain modeled.
- Free overhead payout, no fixed-truck tether spring/damping. Actual routing,
  taut-hose pull, gun COM changes and hose deformation are not validated.
- Synthetic truth-curve curriculum and existing AMP/gait/arm-smoothing/
  centreline/heading rewards and termination conditions are retained.
- TCP training standoff is 1–2 cm above synthetic ground. Vision inference
  is not run in training. Trained checkpoint must be separately tested with
  the camera bridge and with true gun-removed geometry.

## Long training: 8000 additional iterations

```bash
cd /home/dong/b2_z1
conda activate env_isaaclab
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/rsl_rl/train_amp.py \
  --task Unitree-b2z1-AMP-VB7-Load-Robust \
  --num_envs 512 --headless --max_iterations 8000 \
  --resume --reset_optimizer \
  --load_run 2026-10-08_14-49-38_VB4_payout_fromVA15_5000 \
  --checkpoint model_5199.pt --run_name VB7_load_robust_long \
  agent.amp_cfg.amp_num_preload_transitions=20000
```

This runner adds max_iterations to the restored counter, so the final
checkpoint is approximately model_13198.pt, not model_8000.pt. Learning rate
is 3e-5 and checkpoint interval 100. No long training was automatically started.

```bash
cd /home/dong/b2_z1
conda activate env_isaaclab
tensorboard --logdir logs/rsl_rl/b2_z1_amp --port 6006
```

Monitor finite PPO/AMP losses, TCP tracking, body centreline/yaw, gait symmetry,
clearance, low-height/base-contact terminations and truth-curve curriculum.
`VB7/gun_mass_kg`, `VB7/light_payload_fraction`, `VB7/zero_hose_fraction`
report sampled domains; they are not pass rates. Evaluate saved checkpoints
around +500/+1000 iterations; more iterations alone do not ensure improvement.
Compare fixed 0/1/3/5/6 kg hose loads and nominal/light gun loads separately;
expand to 10 kg only after the lower-load envelope passes. Retain the original
VB4/5 policies and stop a degrading experiment rather than overwriting them.

Available disk at setup was about 4.3 GB. About 80 checkpoints alone occupy
roughly 1 GB; leave additional headroom for logs and future evaluations.

## Startup verification

Run `2026-10-09_10-08-15_VB7_smoke_test`, 128 environments, 2 iterations:
checkpoint saved; all 88 TensorBoard scalar tags finite. Sampled light-domain
fraction 21.1%, zero-hose fraction 40.6%, mean gun mass 2.05 kg. This verifies
configuration wiring only, not long-horizon stability or real payload safety.
