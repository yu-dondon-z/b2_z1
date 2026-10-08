# VB4: overhead-payout hypothesis

The fixed-anchor VB1–VB3 model generated up to 179 N tension at only 3 kg in
the 60 s sweep, while VB2's 60 N cap was active roughly half the time. VB4
instead assumes the truck feeds hose freely over the evaluated distance. It
keeps the physical gun and the effective suspended hose weight, inertia and
drag; the fixed-anchor spring/damping pull is disabled.

This is **not a measured real-hose model**. Real payout rate, friction,
unsupported length and worst-case pull must be measured before hardware use.

The unchanged VA15 model_5000.pt under the VB4 3 kg play task reached TCP p95
11.2 cm, base lateral p95 3.5 cm, zero resets, 3.4° max tilt in 60 s. This is
the baseline that any new VB4 checkpoint must at least preserve.

```bash
cd /home/dong/b2_z1
B2_Z1_VB1_STAGE=1 /home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train_amp.py \
  --task Unitree-b2z1-AMP-VB4-Payout --num_envs 512 --headless \
  --max_iterations 200 --resume --reset_optimizer \
  --load_run 2026-09-28_15-07-31_va15_curriculum_from2500 \
  --checkpoint model_5000.pt --run_name VB4_payout_fromVA15_5000
```

For equal-duration fixed-load evaluation:

```bash
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/evaluate_vb1_load_sweep.py \
  --task Unitree-b2z1-AMP-VB4-Payout-Play \
  --checkpoint PATH_TO_CHECKPOINT.pt --masses 1 3 5 --max-steps 3000
```

Do not promote based on training reward alone: require path, TCP, tilt and
reset metrics across multiple masses. Then test real-hose load separately.
