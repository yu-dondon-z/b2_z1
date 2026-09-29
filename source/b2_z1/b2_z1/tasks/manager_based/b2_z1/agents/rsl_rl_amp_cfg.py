# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from b2_z1.rsl_rl import (
    AMP_RslRlOnPolicyRunnerCfg,
    AMP_RslRlPpoActorCriticCfg,
    AMP_RslRlPpoAlgorithmCfg,
    AMP_AlgorithmCfg,
)

import glob
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", ".."))
MOTION_FILES = glob.glob(os.path.join(_PROJECT_ROOT, "datasets", "b2_motion", "*"))




@configclass
class AMP_PPORunnerCfg(AMP_RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    # Keep the local training budget while using the latest amp_loco code.
    max_iterations = 8000
    save_interval = 100
    experiment_name = "b2_z1_amp"  # same as task name
    # Match the raw policy-action clamp used by the MuJoCo deployment.
    clip_actions = 5.0
    empirical_normalization = False
    policy = AMP_RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    
    amp_cfg = AMP_AlgorithmCfg(
        amp_motion_files=MOTION_FILES,
        amp_reward_coef=0.2,
        
        min_normalized_std=[0.01, 0.01, 0.01] * 6
    )


@configclass
class AMPDeploymentFineTuneRunnerCfg(AMP_PPORunnerCfg):
    """Conservative optimizer settings for deployment-oriented fine-tuning."""

    # ``runner.learn`` interprets this as additional iterations after loading
    # a checkpoint. The first V2 pass therefore finishes near model_13998.
    # First run a conservative 2000-iteration adaptation and inspect the
    # intermediate checkpoints instead of blindly training to 16000 again.
    max_iterations = 2000
    save_interval = 100
    experiment_name = "b2_z1_amp"
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.0,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    amp_cfg = AMP_AlgorithmCfg(
        amp_motion_files=MOTION_FILES,
        amp_reward_coef=0.2,
        # The legacy checkpoint has no discriminator state. Initially favor
        # the known task reward while the new discriminator is learning.
        amp_task_reward_lerp=0.95,
        amp_num_preload_transitions=20000,
        min_normalized_std=[0.01, 0.01, 0.01] * 6,
        # Per-action caps are model_11998's learned std values plus a 10%
        # margin.  This preserves its joint-specific exploration profile while
        # preventing the global noise growth observed in model_15997.
        max_policy_std=[
            0.096071, 0.098269, 0.101664, 0.103195, 0.314309, 0.320201,
            0.230908, 0.228696, 0.222399, 0.221731, 0.232750, 0.233184,
            0.220575, 0.234827, 0.310242, 0.289739, 0.500038, 0.097695,
        ],
    )


@configclass
class AMPRearDeploymentFineTuneRunnerCfg(AMP_PPORunnerCfg):
    """Conservative continuation settings for rear-vision ``model_7999``."""

    max_iterations = 2000
    save_interval = 100
    experiment_name = "b2_z1_amp"
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.0,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    amp_cfg = AMP_AlgorithmCfg(
        amp_motion_files=MOTION_FILES,
        amp_reward_coef=0.2,
        amp_task_reward_lerp=0.90,
        amp_num_preload_transitions=20000,
        min_normalized_std=[0.01, 0.01, 0.01] * 6,
        # model_7999's learned standard deviations plus roughly 5%. This does
        # not abruptly alter the loaded exploration profile, but prevents the
        # late-training noise growth that previously led to NaNs.
        max_policy_std=[
            0.500, 0.504, 0.504, 0.503, 0.628, 0.648,
            0.583, 0.583, 0.518, 0.587, 0.579, 0.607,
            0.601, 0.504, 0.599, 0.526, 0.712, 0.427,
        ],
    )


@configclass
class AMPRearVisionV2FineTuneRunnerCfg(AMP_PPORunnerCfg):
    """Low-drift continuation settings for rear-vision ``model_9400``."""

    max_iterations = 1000
    save_interval = 100
    experiment_name = "b2_z1_amp"
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.5,
    )
    amp_cfg = AMP_AlgorithmCfg(
        amp_motion_files=MOTION_FILES,
        amp_reward_coef=0.2,
        # This value is the TASK fraction, not the AMP fraction. Using 0.80
        # restores four times as much discriminator influence as the failed
        # 0.95 run and protects the recorded trot during path adaptation.
        amp_task_reward_lerp=0.80,
        amp_num_preload_transitions=20000,
        min_normalized_std=[0.01, 0.01, 0.01] * 6,
        # model_9400 standard deviations plus about five percent. The cap
        # prevents late-stage noise growth without clipping the loaded policy
        # on the first update.
        max_policy_std=[
            0.247132, 0.248000, 0.248041, 0.237991, 0.338560, 0.345751,
            0.298613, 0.295068, 0.201421, 0.298491, 0.295944, 0.306050,
            0.305866, 0.196675, 0.242351, 0.212919, 0.323570, 0.168675,
        ],
    )


@configclass
class AMPRearVisionV3GaitFineTuneRunnerCfg(AMP_PPORunnerCfg):
    """Very low-drift gait continuation settings for ``model_9600``."""

    max_iterations = 600
    save_interval = 100
    # Keep this name so Isaac Lab can resolve the V2.2 source run on resume.
    experiment_name = "b2_z1_amp"
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.03,
        entropy_coef=0.0,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.003,
        max_grad_norm=0.4,
    )
    amp_cfg = AMP_AlgorithmCfg(
        amp_motion_files=MOTION_FILES,
        amp_reward_coef=0.2,
        amp_task_reward_lerp=0.80,
        amp_num_preload_transitions=20000,
        min_normalized_std=[0.01, 0.01, 0.01] * 6,
        # model_9600 learned std plus about 3%.  The tight cap preserves its
        # exploration profile while preventing the late noise growth/NaNs seen
        # in older long continuations.
        max_policy_std=[
            0.239283, 0.240109, 0.238404, 0.230884, 0.327471, 0.333064,
            0.287306, 0.284478, 0.193761, 0.285627, 0.284256, 0.295570,
            0.295242, 0.189852, 0.234105, 0.204933, 0.308848, 0.164316,
        ],
    )


@configclass
class AMPRearCurveGaitHeadingV5FineTuneRunnerCfg(
    AMPRearVisionV3GaitFineTuneRunnerCfg
):
    """Conservative continuation for mixed straight and curved paths."""

    # Curves introduce a new state distribution, so allow a longer run while
    # retaining V3's low learning rate, tight KL limit and bounded policy std.
    max_iterations = 1200
    save_interval = 100
    experiment_name = "b2_z1_amp"


@configclass
class AMPRealGaitClearanceV1FineTuneRunnerCfg(
    AMPRearCurveGaitHeadingV5FineTuneRunnerCfg
):
    """Low-drift continuation for explicit foot-clearance adaptation."""

    max_iterations = 1200
    save_interval = 100
    experiment_name = "b2_z1_amp"


@configclass
class AMPRealGaitClearanceV2FineTuneRunnerCfg(
    AMPRealGaitClearanceV1FineTuneRunnerCfg
):
    """Short, low-drift continuation after rejecting the aggressive V1 run."""

    max_iterations = 800
    save_interval = 100
    experiment_name = "b2_z1_amp"


@configclass
class AMPRealGaitClearanceV3FineTuneRunnerCfg(
    AMPRearCurveGaitHeadingV5FineTuneRunnerCfg
):
    """Minimal continuation that retains the V5 optimizer state."""

    max_iterations = 600
    save_interval = 100
    experiment_name = "b2_z1_amp"


@configclass
class AMPRealGaitClearanceV4HeadingRecoveryRunnerCfg(
    AMPRealGaitClearanceV3FineTuneRunnerCfg
):
    """Brief yaw-tracking recovery from the selected V3 checkpoint."""

    max_iterations = 300
    save_interval = 50
    experiment_name = "b2_z1_amp"


@configclass
class AMPRealGaitClearanceV5StraightRecoveryRunnerCfg(
    AMPRealGaitClearanceV3FineTuneRunnerCfg
):
    """Very-low-KL straight-heading recovery from selected V3 model_10700."""

    max_iterations = 200
    save_interval = 25
    experiment_name = "b2_z1_amp"
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.02,
        entropy_coef=0.0,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=3.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.2,
    )


@configclass
class AMPProprioOmniV4FineTuneRunnerCfg(AMP_PPORunnerCfg):
    """Conservative omnidirectional continuation from V3 ``model_10000``."""

    max_iterations = 2000
    save_interval = 100
    experiment_name = "b2_z1_amp"
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.5,
    )
    amp_cfg = AMP_AlgorithmCfg(
        amp_motion_files=MOTION_FILES,
        amp_reward_coef=0.2,
        amp_task_reward_lerp=0.80,
        amp_num_preload_transitions=20000,
        min_normalized_std=[0.01, 0.01, 0.01] * 6,
        # model_10000 learned standard deviations plus five percent. This is
        # enough exploration for new directions without reopening the late-run
        # noise growth that previously caused NaNs.
        max_policy_std=[
            0.251247, 0.252114, 0.250324, 0.242428, 0.343845, 0.349717,
            0.301671, 0.298702, 0.203449, 0.299908, 0.298469, 0.310349,
            0.310004, 0.199345, 0.245810, 0.215180, 0.324290, 0.172532,
        ],
    )


@configclass
class AMPProprioOmniV5FoundationRunnerCfg(
    AMPRealGaitClearanceV3FineTuneRunnerCfg
):
    """Low-drift omni adaptation from clearance V3 ``model_10700``."""

    max_iterations = 600
    save_interval = 50
    experiment_name = "b2_z1_amp"


@configclass
class AMPProprioForwardV6FoundationRunnerCfg(
    AMPRealGaitClearanceV3FineTuneRunnerCfg
):
    """Short direct-forward adaptation before expanding the command space."""

    max_iterations = 400
    save_interval = 25
    experiment_name = "b2_z1_amp"


@configclass
class AMPProprioForwardV7HeadingHoldRunnerCfg(
    AMPProprioForwardV6FoundationRunnerCfg
):
    """Short adaptation to bounded IMU heading corrections."""

    max_iterations = 200
    save_interval = 25


@configclass
class AMPProprioForwardBackwardV8ClearanceRunnerCfg(
    AMPProprioForwardV7HeadingHoldRunnerCfg
):
    """Conservative bidirectional continuation with mild clearance shaping."""

    max_iterations = 400
    save_interval = 25


@configclass
class AMPProprioForwardBackwardV9GentleRunnerCfg(
    AMPProprioForwardBackwardV8ClearanceRunnerCfg
):
    """Short low-drift continuation with only 15% low-speed reverse samples."""

    max_iterations = 300
    save_interval = 25


@configclass
class AMPProprioForwardBackwardV10BalancedRunnerCfg(
    AMPProprioForwardBackwardV9GentleRunnerCfg
):
    """Brief checkpointed adaptation for measurable lift and reverse motion."""

    max_iterations = 250
    save_interval = 25


@configclass
class AMPProprioBackwardV11BootstrapRunnerCfg(
    AMPProprioForwardBackwardV10BalancedRunnerCfg
):
    """Short backward-only acquisition stage with frequent checkpoints."""

    max_iterations = 250
    save_interval = 25


@configclass
class AMPProprioBackwardV12AcquisitionRunnerCfg(
    AMPProprioBackwardV11BootstrapRunnerCfg
):
    """Moderately faster optimizer for escaping the stationary policy basin."""

    max_iterations = 200
    save_interval = 25
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.5,
    )


@configclass
class AMPProprioBackwardV13ExpertStartRunnerCfg(
    AMPProprioBackwardV12AcquisitionRunnerCfg
):
    """Velocity-matched expert-reset bootstrap; discriminator keeps all priors."""

    max_iterations = 200
    save_interval = 25


@configclass
class AMPProprioForwardV14ClearanceRunnerCfg(
    AMPProprioForwardV7HeadingHoldRunnerCfg
):
    """Low-KL V7 continuation for a measured 2.5 cm swing clearance."""

    max_iterations = 200
    save_interval = 25
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.03,
        entropy_coef=0.0,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=5.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.3,
    )


@configclass
class AMPProprioForwardV15ClearanceRunnerCfg(
    AMPProprioForwardV14ClearanceRunnerCfg
):
    """Second lift stage after the conservative V14 response audit."""

    max_iterations = 100
    save_interval = 25
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.04,
        entropy_coef=0.0,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.003,
        max_grad_norm=0.3,
    )


@configclass
class AMPProprioForwardV16RearClearanceRunnerCfg(
    AMPProprioForwardV15ClearanceRunnerCfg
):
    """Short rear-clearance balancing stage with frequent checkpoints."""

    max_iterations = 75
    save_interval = 15
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.04,
        entropy_coef=0.0,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=1.2e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.003,
        max_grad_norm=0.3,
    )


@configclass
class AMPProprioForwardV17ClearanceRecoveryRunnerCfg(
    AMPProprioForwardV16RearClearanceRunnerCfg
):
    """Very-low-drift recovery after the maximum-clearance V16 checkpoint."""

    max_iterations = 50
    save_interval = 10
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.03,
        entropy_coef=0.0,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=5.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.25,
    )


@configclass
class AMPProprioForwardV18ContactBalanceRunnerCfg(
    AMPProprioForwardV15ClearanceRunnerCfg
):
    """Low-KL contact-balance continuation from V15 ``model_11350``."""

    max_iterations = 100
    save_interval = 10
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.03,
        entropy_coef=0.0,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=5.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.25,
    )


@configclass
class AMPProprioForwardV19ContactCycleRunnerCfg(
    AMPProprioForwardV15ClearanceRunnerCfg
):
    """Low-KL contact-cycle continuation from V15 ``model_11350``."""

    max_iterations = 100
    save_interval = 10
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.03,
        entropy_coef=0.0,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=5.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.25,
    )


@configclass
class AMPProprioOmniV20VisionPreserveRunnerCfg(
    AMPProprioForwardV19ContactCycleRunnerCfg
):
    """Low-drift cardinal-motion acquisition from V19 ``model_11449``."""

    # Frequent checkpoints let the direction/vision regression gate select an
    # early model before the forward gait or TCP response begins to drift.
    max_iterations = 250
    save_interval = 10
    experiment_name = "b2_z1_amp"
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.04,
        entropy_coef=0.0,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=1.5e-5,
        # A fixed rate prevents the low initial KL from silently raising the
        # optimizer by several multiples and eroding the V19 visual behavior.
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.004,
        max_grad_norm=0.30,
    )


@configclass
class AMPVA1TruthCurveRunnerCfg(AMP_PPORunnerCfg):
    """Stable from-scratch 8000-iteration runner for the VA1 curriculum."""

    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 50
    experiment_name = "b2_z1_amp"
    clip_actions = 5.0
    empirical_normalization = False

    # Starting below std=1 avoids violent early arm commands, while the
    # explicit upper bounds prevent the late noise growth that caused NaNs in
    # older long runs.  The leg cap remains high enough to discover a trot.
    policy = AMP_RslRlPpoActorCriticCfg(
        init_noise_std=0.65,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.20,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        # A fixed conservative rate cannot silently climb toward 1e-2 when
        # the initial KL is small, which was a source of late-run instability.
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.50,
    )
    amp_cfg = AMP_AlgorithmCfg(
        amp_motion_files=MOTION_FILES,
        amp_reward_coef=0.20,
        amp_task_reward_lerp=0.80,
        amp_num_preload_transitions=20000,
        min_normalized_std=[0.01, 0.01, 0.01] * 6,
        max_policy_std=[0.85] * 12 + [0.55] * 6,
    )


@configclass
class AMPVA2TruthCurveRunnerCfg(AMPVA1TruthCurveRunnerCfg):
    """From-scratch runner balancing explicit progress with the AMP gait prior."""

    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.20,
        entropy_coef=0.010,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.5e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.50,
    )
    amp_cfg = AMP_AlgorithmCfg(
        amp_motion_files=MOTION_FILES,
        amp_reward_coef=0.20,
        # Give the motion discriminator enough influence to turn progress into
        # a trot instead of an arbitrary low-energy shuffle.
        amp_task_reward_lerp=0.65,
        amp_num_preload_transitions=20000,
        min_normalized_std=[0.01, 0.01, 0.01] * 6,
        max_policy_std=[0.85] * 12 + [0.55] * 6,
    )


@configclass
class AMPVA3TruthCurveRunnerCfg(AMPVA1TruthCurveRunnerCfg):
    """Conservative from-scratch runner for the anti-kneeling VA3 task."""

    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.20,
        entropy_coef=0.007,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.50,
    )
    amp_cfg = AMP_AlgorithmCfg(
        amp_motion_files=MOTION_FILES,
        amp_reward_coef=0.20,
        # More style influence than VA2, while task rewards still retain a
        # majority share for curve, speed and TCP tracking.
        amp_task_reward_lerp=0.55,
        amp_num_preload_transitions=20000,
        min_normalized_std=[0.01, 0.01, 0.01] * 6,
        max_policy_std=[0.75] * 12 + [0.45] * 6,
    )


@configclass
class AMPVA4SmoothToolRunnerCfg(AMPVA3TruthCurveRunnerCfg):
    """Conservative VA3 continuation after correcting the TCP objective."""

    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.20,
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.50,
    )


@configclass
class AMPVA5ElbowWristRunnerCfg(AMPVA4SmoothToolRunnerCfg):
    """Small, separate continuation to suppress joint3/joint4 cycling."""

    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.20,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=8.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.50,
    )


@configclass
class AMPVA6NullspaceRunnerCfg(AMPVA5ElbowWristRunnerCfg):
    """Use VA5's conservative update rate for a different nullspace objective."""


@configclass
class AMPVA7WholeBodyRunnerCfg(AMPVA3TruthCurveRunnerCfg):
    """Independent 8000-iteration run, with the VA3 stable from-scratch PPO."""

    max_iterations = 8000
    save_interval = 100


@configclass
class AMPVA8PathRunnerCfg(AMPVA7WholeBodyRunnerCfg):
    """Low-drift fine-tuning from VA7/7999 for body path alignment."""

    max_iterations = 1000
    save_interval = 100
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.10,
        entropy_coef=0.002,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=0.40,
    )


@configclass
class AMPVA9PreserveArmRunnerCfg(AMPVA8PathRunnerCfg):
    """Very conservative B2-head adaptation with the VA7 arm actor fixed."""

    max_iterations = 800
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.001,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.35,
    )


@configclass
class AMPVA10SignedYawRunnerCfg(AMPVA9PreserveArmRunnerCfg):
    """Low-rate full-body PPO with VA7 arm-action distillation at train time."""

    max_iterations = 1000


@configclass
class AMPVA11SafetyRunnerCfg(AMPVA10SignedYawRunnerCfg):
    """Conservative continuation with safety-gated curve progression."""

    max_iterations = 300


@configclass
class AMPVA12GaitPreserveRunnerCfg(AMPVA11SafetyRunnerCfg):
    """Same PPO step size; only the prior and safety terms change."""


@configclass
class AMPVA13LowNoiseRunnerCfg(AMPVA12GaitPreserveRunnerCfg):
    """Keep the conservative PPO schedule for the lower-noise comparison."""


@configclass
class AMPVA14FromScratchRunnerCfg(AMPVA7WholeBodyRunnerCfg):
    """Fresh PPO/AMP student; the independent Z1 teacher is training-only."""

    resume = False
    max_iterations = 8000
    save_interval = 100

    def __post_init__(self):
        self.policy.init_noise_std = 0.50
        self.algorithm.learning_rate = 2.0e-4
        self.algorithm.entropy_coef = 0.005
        # Uniform bounds avoid assuming that Z1 occupies the final six rows.
        self.amp_cfg.max_policy_std = [0.65] * 18


@configclass
class AMPVA15CurriculumRunnerCfg(AMPVA14FromScratchRunnerCfg):
    """Conservative continuation from the screened VA14 checkpoint."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 1.0e-4
        self.max_iterations = 5500
