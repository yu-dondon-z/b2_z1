# VB6 local visual path guidance

Date: 2026-10-09. Branch: VB6. Policy weights are unchanged.

The legacy file controller added the global visible-path tangent angle to
the forward lookahead point's lateral angle. The latter already includes
heading on an oblique line, so this is not independent cross-track feedback.
It also used a distant secant instead of the tangent near the base.

VB6 estimates a local tangent and signed normal cross-track displacement:

1. Require a fresh valid current camera trajectory, as before.
2. Transform cached observed world points into the current base frame.
3. Fit y=a*x+b over cached points within x=[-0.5,0.5] m when at least four
   points span 0.3 m. Cache insertion order is not used as path ordering.
4. If local coverage is unavailable, use the closest nonzero segment of the
   current ordered camera path, extrapolating its line near the origin.
5. Set tangent heading atan(a), cross-track b/sqrt(1+a*a), and desired yaw
   correction = tangent heading + atan2(cross_track_gain*cross_track, lookahead).
6. Keep existing velocity filters, start handoff, stale-vision stop,
   turn limits, TCP target generation, policy and load physics unchanged.

Synthetic training commands are unchanged. All file-mode playback commands
on this branch use VB6 guidance. Old behavior is preserved on branch VB5.

## Independent actual-ground validation

Same VB4 model_5199.pt, 3 kg effective hose load, free overhead payout,
physical gun, RGB-D/RHACrackNet and 60 s scene with 0.4 m sine-cubed crack:

| Guidance | Travel | Base-to-crack p95 | TCP planar-to-crack p95 | TCP target XYZ p95 | Resets |
|---|---:|---:|---:|---:|---:|
| Legacy VB5 branch | 10.16 m | 14.34 cm | 2.91 cm | 14.48 cm | 0 |
| Local camera segment only | 9.88 m | 11.36 cm | 2.54 cm | 15.79 cm | 0 |
| VB6 local cached fit | 9.96 m | 5.05 cm | 2.16 cm | 15.74 cm | 0 |

VB6 maximum base distance was 5.40 cm; maximum tilt was 3.71 degrees.
Actual TCP height above the road p05–p95 was 0.69–0.90 cm. Planar crack
tracking improved, but target-pose accuracy did not. Target height/workspace
and longitudinal timing remain separate issues; do not claim welding-ready
precision from the planar metric. These are single asynchronous-vision
trials, not repeat-seed or heavy-load certification.

Evidence:
- outputs/vb6_guidance_vb4_3kg: local-segment intermediate experiment.
- outputs/vb6_cached_guidance_vb4_3kg: final CSV, overlay and process logs.
- scripts/vision/evaluate_vision_ground_trace.py: independent mesh distances.
- scripts/vision/test_path_guidance.py: four regression tests passed.

## Playback

VS Code F5: `VB6：视觉局部中心线＋TCP离地1cm（VB4-5199，3kg）`.

```bash
cd /home/dong/b2_z1
B2_Z1_HOSE_MASS_KG=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/play_va15_vision.py \
  --payout \
  --checkpoint logs/rsl_rl/b2_z1_amp/2026-10-08_14-49-38_VB4_payout_fromVA15_5000/model_5199.pt \
  --episode-length-s 600 --max-steps 15000
```

Ctrl+C stops both owned playback and segmentation subprocesses. No real
robot control or GitHub upload was performed.

## Subsequent requested TCP-height adjustment

The original active-tracking actual gun TCP height averaged 7.45 mm;
p05–p95 was 6.88–8.52 mm. The tracked point is the downward extended
gun TCP (`dummy_target`), not the arm flange. The old target averaged
100.96 mm above the road, explaining most of the XYZ target error.

Current payout visual playback uses `tool_height_m=0.01`. It additionally
smooths world height, leaving the original body-relative planar filtering
unchanged, to avoid converting base bobbing into target-height oscillation.
For this known flat simulation only, `file_road_surface_z_w_m=0.0` anchors
the height to the actual road surface. Otherwise projected trench-depth
noise and bias shifts the requested 1 cm standoff. Generic commands keep
this override at None, and training/synthetic playback remain unchanged.
An unknown/uneven real road needs a validated surface-height estimate;
do not copy the fixed world z=0 assumption into real deployment.

Intermediate 60 s tests at 3 kg had no resets. Merely reducing tool_height
while retaining body-frame filtering produced mean target z=2.02 cm and
occasional below-road targets. World-height filtering removed the below-road
excursions in the steady window but the projected-surface bias remained
(mean target z=2.18 cm). These intermediate variants are not the final config.

Final known-road configuration: 30 s camera/segmentation replay, 3 kg,
VB4 5199, zero resets, maximum tilt 3.72 degrees. In the settled 13–30 s
window, mean target height was 10.001 mm (p05–p95: 10.000–10.004 mm),
actual TCP mean height 7.032 mm (p05–p95: 6.852–7.195 mm), and absolute
height error mean/p95 2.969/3.149 mm. XYZ target error mean/p95 was
2.16/4.04 cm. The whole 30 s base-to-crack p95 was 4.55 cm.
The final variant has not yet repeated the preceding 60 s/multi-load tests;
do not treat shorter-window improvements as a directly matched benchmark.
Evidence: `outputs/vb6_tcp_1cm_known_road_vb4_3kg`.
