# VB2 bounded-hose experiment

VB2 is a **simulation-only hypothesis**: retain VB1's physical gun and 1–3 kg
effective suspended hose load, but cap the *truck-side tether tension* at 60 N.
The raw uncapped tether tension is still logged as `VB2/hose_raw_tension_peak_n`
and in motion traces. This is not a measured limit or a safe real-robot load.
If the raw tension remains above 60 N for much of a trial, the model relies on
unverified hose feed/slack and the result must not be treated as deployable.

Start from the screened VB1 checkpoint at iteration 5300:

```bash
cd /home/dong/b2_z1
B2_Z1_VB1_STAGE=1 /home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train_amp.py \
  --task Unitree-b2z1-AMP-VB2-Bounded-Hose --num_envs 512 --headless \
  --max_iterations 200 --resume --reset_optimizer \
  --load_run 2026-10-08_10-08-10_VB1_s1_1to3kg \
  --checkpoint model_5300.pt --run_name VB2_bounded_hose_from5300
```

For a fixed-load synthetic comparison (replace the checkpoint path as needed):

```bash
cd /home/dong/b2_z1
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/evaluate_vb1_load_sweep.py \
  --task Unitree-b2z1-AMP-VB2-Bounded-Hose-Play \
  --checkpoint logs/rsl_rl/b2_z1_amp/2026-10-08_10-08-10_VB1_s1_1to3kg/model_5300.pt \
  --masses 3 --max-steps 3000
```

Compare equal durations, mass, task and random seed. Inspect travel, reset
count, TCP and lateral error p95, tilt, applied/raw tether tension and the
fraction of time the cap is active. Do not advance to heavier-load stages
without a clear improvement over VA15 and VB1 on these measures.

## First 60 s, fixed-3 kg screening (2026-10-08)

| Checkpoint | Travel | TCP p95 | Lateral p95 | Resets | Time at 60 N cap |
|---|---:|---:|---:|---:|---:|
| VB1 5300 (same VB2 physics) | 11.66 m | 16.7 cm | 4.3 cm | 0 | 57.4% |
| VB2 5400 | 9.86 m | 13.5 cm | 8.5 cm | 0 | 52.9% |
| VB2 5499 | 10.00 m | 10.9 cm | 10.1 cm | 1 | 51.3% |

The short training improves episode survival, but this screening does **not**
establish a better overall policy: lateral tracking regresses and the cap is
active roughly half the time. Do not replace the VA15 or deployed checkpoint
with either VB2 checkpoint yet. The uncapped 3 kg replay and, ultimately,
measured hose feed/force are necessary before extending the load range.

Uncapped replay of VB2 5400 (same fixed 3 kg and 60 s) produced 8.32 m
travel, 15.0 cm TCP p95, 8.1 cm lateral p95, zero resets, and a 182 N
peak tether tension. Thus the screening improvement depends on the bounded
model and does not transfer cleanly to the original fixed-anchor hose model.
