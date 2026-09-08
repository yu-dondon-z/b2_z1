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
