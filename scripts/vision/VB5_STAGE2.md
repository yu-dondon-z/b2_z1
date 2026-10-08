# VB5 stage-2 payout hypothesis

Continue VB4 iteration 5199 while widening only the sampled effective
unsupported hose mass from 1–3 kg to 1–6 kg. The gun model and assumed free
overhead payout remain unchanged; this is not a validated real-hose model.

VB4 5199 60 s baselines under free payout:

| Hose mass | TCP p95 | Lateral p95 | Resets |
|---|---:|---:|---:|
| 1 kg | 8.7 cm | 2.0 cm | 0 |
| 3 kg | 9.5 cm | 3.1 cm | 0 |
| 5 kg | 10.9 cm | 3.4 cm | 0 |
| 10 kg | 14.8 cm | 2.8 cm | 3 |

At fixed-truck-anchor 3 kg, VB4 5199 had TCP p95 12.6 cm, lateral p95
3.1 cm, zero resets and 92 N peak tether tension. That is a separate physics
assumption; do not substitute it for the free-payout results.

```bash
cd /home/dong/b2_z1
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train_amp.py \
  --task Unitree-b2z1-AMP-VB5-Payout-Stage2 --num_envs 512 --headless \
  --max_iterations 250 --resume --reset_optimizer \
  --load_run 2026-10-08_14-49-38_VB4_payout_fromVA15_5000 \
  --checkpoint model_5199.pt --run_name VB5_payout_stage2_from5199_complete
```

Promote only if 3 and 5 kg retain zero resets and low path/TCP p95, and 10 kg
improves without a collapse. Do not deploy before measuring real hose payout,
force and inertia and repeating more than one 60 s trajectory.

## Completed continuation and first fixed-load sweep

The interrupted initial run saved only model_5200.pt. The completed run is
`2026-10-08_16-32-48_VB5_payout_stage2_from5199_complete`, resumed directly
from VB4 5199 for 250 additional iterations, ending at model_5448.pt.
No non-finite TensorBoard scalars were found. The truth-curve curriculum
remained at level 0; the final timeout fraction was approximately 93%.

| Checkpoint | Mass | TCP p95 | Lateral p95 | Resets in 60 s |
|---|---:|---:|---:|---:|
| VB4 5199 | 3 kg | 9.5 cm | 3.1 cm | 0 |
| VB4 5199 | 5 kg | 10.9 cm | 3.4 cm | 0 |
| VB4 5199 | 10 kg | 14.8 cm | 2.8 cm | 3 |
| VB5 5448 | 3 kg | 12.7 cm | 2.9 cm | 0 |
| VB5 5448 | 5 kg | 13.0 cm | 3.8 cm | 0 |
| VB5 5448 | 10 kg | 16.5 cm | 2.3 cm | 0 |

VB5 5448 improves high-load episode survival but trades away TCP accuracy.
These results do not establish that it is an overall replacement for VB4.
Replay traces now include termination causes and pre-reset height/tilt.
Travel excludes auto-reset teleportation. All of the table's trials use the
free-payout assumption and seed 42.

The middle checkpoint model_5350.pt did not dominate the final checkpoint:
3/5 kg TCP p95 were 12.2/12.1 cm with lateral p95 4.6/4.5 cm. At 10 kg it
reset once due to `low_base_height`, with pre-reset height 0.386 m.

For model_5448.pt, the repeated seed-123 3/10 kg trials had zero resets;
TCP p95 was 14.2/16.1 cm. Fixed right-turn radius-10 m trials at 3/10 kg
also had zero resets, with TCP p95 13.4/16.5 cm and lateral p95 5.0/4.6 cm.
These finite trials support further simulation testing, not real welding
precision or arbitrary 1–10 kg hose routing.

Selection: preserve VB4 model_5199.pt as the low-load precision reference;
use VB5 model_5448.pt as the high-load survival candidate. Do not label VB5
as a universal improvement. Its 3 kg lateral TCP component p95 rose from
VB4's 2.7 cm to 9.6 cm even though body-centreline p95 stayed near 3 cm.
The next precision adaptation should therefore examine the arm's tracking
response as well as the base rewards.

VS Code launch entries:

- `VB5：软管负载曲线播放（5448，5kg）`
- `VB5：视觉裂缝＋焊枪负载（5448，3kg）`
- `VB4：低负载精度对比（5199，3kg）`
- `VB4：视觉低负载精度对比（5199，3kg）`

The synthetic and visual sessions allow 15,000 policy steps (about 5 min)
with a 600 s episode horizon. Ctrl+C stops the selected playback session;
the visual launcher also stops its owned segmentation subprocess.

## Full RGB-D segmentation playback, 3 kg, 60 s

Both policies completed without resets using the same payout scene and visual
pipeline. These are single trials, not a statistical certification.

| Policy | Travel | TCP target error p95 | Base distance to actual crack p95 | TCP planar distance to crack p95 |
|---|---:|---:|---:|---:|
| VB4 5199 | 10.16 m | 14.48 cm | 14.34 cm | 2.91 cm |
| VB5 5448 | 11.26 m | 15.98 cm | 13.31 cm | 9.39 cm |

Actual TCP height above the road p05–p95 was 0.71–0.92 cm for VB4 and
0.88–3.39 cm for VB5. Target-pose error and distance to the physical crack
are separate metrics; neither should be substituted for the other.

Important: file-mode `path_lateral_error_b` is the lateral coordinate of a
forward lookahead point, NOT the base's shortest distance to the crack.
The geometry metrics above are independently computed against the scene's
actual mesh polyline with `evaluate_vision_ground_trace.py`. Synthetic-mode
lateral errors in the earlier tables have a different definition.

Evidence is in `outputs/vb4_5199_vision_3kg` and
`outputs/vb5_5448_vision_3kg`. Preserve VB4 for low-load visual precision and
VB5 for high-load stability testing. Similar base offsets in both visual
trials warrant investigating file-mode path guidance before further training.
