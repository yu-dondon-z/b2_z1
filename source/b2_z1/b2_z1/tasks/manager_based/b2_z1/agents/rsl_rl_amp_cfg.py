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
    # a checkpoint. model_11998 therefore finishes near model_15998.
    max_iterations = 4000
    save_interval = 100
    experiment_name = "b2_z1_amp"
    algorithm = AMP_RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
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
        amp_task_reward_lerp=0.9,
        amp_num_preload_transitions=20000,
        min_normalized_std=[0.01, 0.01, 0.01] * 6,
    )
