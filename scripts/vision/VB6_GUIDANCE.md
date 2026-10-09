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

VS Code F5: `VB6：视觉局部中心线引导（VB4-5199，3kg）`.

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
