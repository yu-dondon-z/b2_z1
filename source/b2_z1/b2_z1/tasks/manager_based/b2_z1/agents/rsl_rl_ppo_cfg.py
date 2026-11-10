# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg,
)
from b2_z1.amp_config import MOTION_FILES


@configclass
class AMPAlgorithmCfg:
    amp_motion_files: list[str] = MOTION_FILES
    amp_reward_coef: float = 0.2
    amp_discr_hidden_dims: list[int] = [1024, 512]
    amp_task_reward_lerp: float = 0.8
    amp_num_preload_transitions: int = 200_000
    min_normalized_std: list[float] = [0.01] * 18


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 250
    experiment_name = "b2_z1_vd3_curve_amp"
    keep_last_checkpoints: int = 5
    empirical_normalization = False
    amp_cfg: AMPAlgorithmCfg = AMPAlgorithmCfg()
    policy = RslRlPpoActorCriticCfg(
        class_name="AMP_ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMP_PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class VD3LegSymmetryCfg(RslRlSymmetryCfg):
    use_data_augmentation = False
    use_mirror_loss = True
    data_augmentation_func = (
        "b2_z1.tasks.manager_based.b2_z1.mdp.vd3_symmetry:vd3_left_right"
    )
    mirror_loss_coeff = 0.2
    # joint1 is index 8, so the twelve legs are not the first twelve actions.
    mirror_action_indices: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]


@configclass
class VD3LegSymmetryRunnerCfg(PPORunnerCfg):
    """Conservative fine-tuning profile; the original VD3 profile stays available."""

    max_iterations = 3000
    algorithm = PPORunnerCfg().algorithm.replace(
        learning_rate=5.0e-5, schedule="fixed", symmetry_cfg=VD3LegSymmetryCfg(),
    )


@configclass
class VD4PPORunnerCfg(PPORunnerCfg):
    """Separate experiment namespace for calibrated-front-camera training."""

    experiment_name = "b2_z1_vd4_front_calibrated_amp"


@configclass
class VD5PPORunnerCfg(VD3LegSymmetryRunnerCfg):
    """Seam-command fine-tuning retaining the deployed policy's leg symmetry."""

    experiment_name = "b2_z1_vd5_world_seam_amp"
    max_iterations = 20000


@configclass
class VD5StartupPPORunnerCfg(VD5PPORunnerCfg):
    """Separate logs and checkpoints for nominal-stand startup adaptation."""

    experiment_name = "b2_z1_vd5_startup_amp"
