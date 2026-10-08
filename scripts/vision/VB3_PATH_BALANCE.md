# VB3 path-balance experiment

VB3 continues from VB2 iteration 5400. Only the base centreline and heading
rewards are moderately strengthened; actor size, AMP prior, arm rewards, gun,
and 1–3 kg bounded-hose model are unchanged. The 60 N cap remains an unmeasured
simulation assumption, so passing this test would not establish real-world
load tolerance.

```bash
cd /home/dong/b2_z1
B2_Z1_VB1_STAGE=1 /home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train_amp.py \
  --task Unitree-b2z1-AMP-VB3-Path-Balance --num_envs 512 --headless \
  --max_iterations 300 --resume --reset_optimizer \
  --load_run 2026-10-08_10-45-53_VB2_bounded_hose_from5300 \
  --checkpoint model_5400.pt --run_name VB3_path_balance_from5400
```

Screen 3 kg, 60 s in the same synthetic comparison scene as VB2:

```bash
cd /home/dong/b2_z1
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/evaluate_vb1_load_sweep.py \
  --task Unitree-b2z1-AMP-VB3-Path-Balance-Play \
  --checkpoint PATH_TO_CHECKPOINT.pt --masses 3 --max-steps 3000
```

Accept only if TCP p95, lateral p95, reset count, and tilt jointly pass against
the VB1 5300 and VB2 5400 baselines. Repeat without the artificial cap using
`Unitree-b2z1-AMP-VB1-Loaded-Gun-Play`. If cap duty remains near 50%, confirm
the real hose feed/slack before using the result outside simulation.

## 3 kg, 60 s screening result

| Checkpoint/physics | TCP p95 | Lateral p95 | Resets | Peak tether |
|---|---:|---:|---:|---:|
| VB2 5400, capped | 13.5 cm | 8.5 cm | 0 | 60 N applied |
| VB3 5600, capped | 10.5 cm | 9.1 cm | 0 | 60 N applied |
| VB3 5699, capped | 12.9 cm | 8.0 cm | 0 | 60 N applied |
| VB3 5699, uncapped | 12.5 cm | 8.4 cm | 0 | 179 N |

VB3 5699 modestly improves both p95 errors relative to VB2 5400 under the
same capped physics, but remains much less centered than the VA15 comparison
(~3 cm lateral p95 under the uncapped 3 kg model). It is **not** a deployment
candidate. The fixed-anchor hose model reaches 179 N even at 3 kg; a truck
that pays out hose should be modeled and measured explicitly.
