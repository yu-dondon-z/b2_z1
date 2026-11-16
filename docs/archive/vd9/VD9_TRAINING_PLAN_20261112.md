> 历史归档：内容和命令对应原记录日期，不是当前VD13操作指令。当前状态见[文档入口](../../README.md)。

# VD9 Training Plan: Physics-Rate TCP and Landing Feedback

VD9 is an isolated warm-start from the deployed-reference VD5 Startup `model_4498`. It does not continue VD8 and is not a deployment candidate until it passes the same MuJoCo gait and TCP checks. The warm start retains the actor, AMP discriminator, observation contract, and saved VD5 startup/world curriculum; it resets PPO optimizer state and the learning-iteration counter because the reward objective changes.

## Why this change

Existing evidence does not establish a single root cause. It does establish that the 50 Hz reward path can miss 200 Hz foot descent and TCP vibration, while the actual VD8 contact-force reward averaged roughly `-0.001` per actor step. The physics-rate implementation samples only true simulator state after each 5 ms scene update. It neither fabricates contact timing nor adds privileged inputs to the policy.

VD9 adds exactly two training-only terms:

| Reward | Measurement and shaping | Initial weight |
| --- | --- | ---: |
| TCP high-pass | Causal 3 Hz high-pass of measured base-frame TCP tracking error, sampled at 200 Hz; 1 mm deadzone and 5 mm scale | -0.1 |
| Foot descent | Per-foot maximum airborne downward speed within the final 12 cm above plane ground; penalize only speed above 0.35 m/s, with 0.50 m/s scale | -2 |

Both terms are inactive for the first 2 s of an episode to avoid shaping the startup handoff. Contact force is used only to exclude feet already in contact. A 10-iteration, 32-environment wiring run first used provisional weights `-5/-10`; its episode returns were about `-12.1/-0.34`, versus `+0.38` TCP-position and `+0.47` velocity tracking, so the TCP penalty was disproportionate. The retained weights `-0.1/-2` scale those measured terms to roughly `-0.24/-0.07` at that same early snapshot. This is a magnitude calibration, not an efficacy result; the actual episode lengths were short and the policy had not adapted. There is no roll pulse, touchdown-force surrogate, action filter, gain change, or actor input change. Terrain, actuator setup, commands, AMP, randomization, and all baseline rewards otherwise stay inherited from VD5 Startup.

## Training and convergence

Run `bash scripts/train_vd9.sh`. It saves every 100 iterations, retains every checkpoint, and has an 1800-iteration ceiling. Iteration count is not a convergence claim. Read the rolling 50-iteration scalar summaries with `scripts/diagnostics/analyze_vd9_training.py`; inspect tracking, both new terms, episode length, velocity error, terminations, and the late-stage trend. Continue or resume in the same experiment only while metrics are healthy and still moving toward a stable regime. If trends are still moving at iteration 1800, report “not yet converged” and stop this recipe rather than extending it blindly. If the changed rewards cause sustained episode-length collapse, loss of speed tracking, or a termination increase, stop and diagnose before spending more compute.

The first sim2sim screen compares VD5 `4498` and VD9 checkpoints at the same 0.70 m and 0.80 m distances, speed command, physical setup, and 5–30 s measurement window. Use early/middle/late snapshots to detect non-monotonicity; never select from training reward alone. A checkpoint advances only if TCP XYZ high-pass RMS improves at both distances without slower commanded travel or any of FL/FR/RL/RR touchdown-velocity / force regressions. World path projection error remains a separate task metric, not a substitute for these checks. The existing VD5 actor remains the deployment baseline throughout.

Training artifacts and checkpoints stay under the VD9 experiment directory. No board sync, deployment export, controller/PD adjustment, or robot motion is part of this plan.

This 1800-iteration stage and its early/middle/late MuJoCo screening are complete. The policy was non-monotonic: `model_1500` is a promising offline checkpoint, while the final `model_1799` regressed at 0.70 m. Neither is a released candidate; see [the training and gait review](/home/dong/rl_sar_b2_z1/docs/reports/vd9/VD9_TRAINING_REVIEW_20261112.md).
