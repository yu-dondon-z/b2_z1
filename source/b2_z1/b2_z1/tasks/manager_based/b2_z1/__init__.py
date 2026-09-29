# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Unitree-b2z1-Velocity",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    entry_point="b2_z1.envs:B2Z1ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_env_cfg:B2Z1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-Velocity-Play",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    entry_point="b2_z1.envs:B2Z1ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_env_cfg:B2Z1PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-Gun-Legacy-Play",
    entry_point="b2_z1.envs:B2Z1ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_gun_legacy_env_cfg:B2Z1GunLegacyPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


## AMP 

gym.register(
    id="Unitree-b2z1-AMP",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_amp_env_cfg:B2Z1AMPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMP_PPORunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Deployment-Finetune",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_amp_env_cfg:B2Z1AMPDeploymentFineTuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPDeploymentFineTuneRunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Stability-Finetune",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_amp_env_cfg:B2Z1AMPStabilityFineTuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPDeploymentFineTuneRunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_amp_env_cfg:B2Z1AMPEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMP_PPORunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Filtered-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_amp_env_cfg:B2Z1AMPFilteredEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMP_PPORunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Gun-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_amp_env_cfg:B2Z1AMPGunEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMP_PPORunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Rear-Synthetic",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_rear_vision_env_cfg:B2Z1RearSyntheticAMPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMP_PPORunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Rear-Deployment-Finetune",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:B2Z1RearDeploymentFineTuneAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPRearDeploymentFineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Rear-Deployment-Finetune-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RearDeploymentFineTuneAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPRearDeploymentFineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Rear-Vision-V2-Finetune",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RearVisionV2FineTuneAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRearVisionV2FineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Rear-Vision-V2-Finetune-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RearVisionV2FineTuneAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRearVisionV2FineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Rear-Vision-V2-Finetune-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RearVisionV2FineTuneAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRearVisionV2FineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Rear-Vision-V3-Gait-Finetune",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RearVisionV3GaitFineTuneAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRearVisionV3GaitFineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Rear-Vision-V3-Gait-Finetune-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RearVisionV3GaitFineTuneAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRearVisionV3GaitFineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Rear-Vision-V3-Gait-Finetune-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RearVisionV3GaitFineTuneAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRearVisionV3GaitFineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Rear-Curve-Gait-Heading-V5-Finetune",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RearCurveGaitHeadingV5FineTuneAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRearCurveGaitHeadingV5FineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Rear-Curve-Gait-Heading-V5-Finetune-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RearCurveGaitHeadingV5FineTuneAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRearCurveGaitHeadingV5FineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Rear-Curve-Gait-Heading-V5-Finetune-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RearCurveGaitHeadingV5FineTuneAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRearCurveGaitHeadingV5FineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Real-Gait-Clearance-V1-Finetune",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RealGaitClearanceV1FineTuneAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRealGaitClearanceV1FineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Real-Gait-Clearance-V1-Finetune-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RealGaitClearanceV1FineTuneAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRealGaitClearanceV1FineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Real-Gait-Clearance-V2-Finetune",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RealGaitClearanceV2FineTuneAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRealGaitClearanceV2FineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Real-Gait-Clearance-V2-Finetune-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RealGaitClearanceV2FineTuneAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRealGaitClearanceV2FineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Real-Gait-Clearance-V3-Finetune",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RealGaitClearanceV3FineTuneAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRealGaitClearanceV3FineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Real-Gait-Clearance-V3-Finetune-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RealGaitClearanceV3FineTuneAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRealGaitClearanceV3FineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Real-Gait-Clearance-V4-Heading-Recovery",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RealGaitClearanceV4HeadingRecoveryAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRealGaitClearanceV4HeadingRecoveryRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Real-Gait-Clearance-V4-Heading-Recovery-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RealGaitClearanceV4HeadingRecoveryAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRealGaitClearanceV4HeadingRecoveryRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Real-Gait-Clearance-V5-Straight-Recovery",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RealGaitClearanceV5StraightRecoveryAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRealGaitClearanceV5StraightRecoveryRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Real-Gait-Clearance-V5-Straight-Recovery-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_rear_vision_env_cfg:"
            "B2Z1RealGaitClearanceV5StraightRecoveryAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPRealGaitClearanceV5StraightRecoveryRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Omni-V4-Finetune",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioOmniV4FineTuneAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioOmniV4FineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Omni-V4-Finetune-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioOmniV4FineTuneAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioOmniV4FineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Omni-V4-Finetune-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioOmniV4FineTuneAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioOmniV4FineTuneRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Omni-V5-Foundation",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioOmniV5FoundationAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioOmniV5FoundationRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Omni-V5-Foundation-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioOmniV5FoundationAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioOmniV5FoundationRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Omni-V5-Foundation-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioOmniV5FoundationAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioOmniV5FoundationRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-V6-Foundation",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardV6FoundationAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardV6FoundationRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-V6-Foundation-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardV6FoundationAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardV6FoundationRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-V6-Foundation-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardV6FoundationAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardV6FoundationRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-V7-Heading-Hold",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardV7HeadingHoldAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardV7HeadingHoldRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-V7-Heading-Hold-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardV7HeadingHoldAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardV7HeadingHoldRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-V7-Heading-Hold-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardV7HeadingHoldAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardV7HeadingHoldRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-Backward-V8-Clearance",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardBackwardV8ClearanceAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardBackwardV8ClearanceRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-Backward-V8-Clearance-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardBackwardV8ClearanceAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardBackwardV8ClearanceRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-Backward-V8-Clearance-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardBackwardV8ClearanceAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardBackwardV8ClearanceRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-Backward-V9-Gentle",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardBackwardV9GentleAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardBackwardV9GentleRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-Backward-V9-Gentle-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardBackwardV9GentleAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardBackwardV9GentleRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-Backward-V9-Gentle-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardBackwardV9GentleAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardBackwardV9GentleRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-Backward-V10-Balanced",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardBackwardV10BalancedAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardBackwardV10BalancedRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-Backward-V10-Balanced-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardBackwardV10BalancedAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardBackwardV10BalancedRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-Backward-V10-Balanced-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardBackwardV10BalancedAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardBackwardV10BalancedRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Backward-V11-Bootstrap",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioBackwardV11BootstrapAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioBackwardV11BootstrapRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Backward-V11-Bootstrap-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioBackwardV11BootstrapAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioBackwardV11BootstrapRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Backward-V11-Bootstrap-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioBackwardV11BootstrapAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioBackwardV11BootstrapRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Backward-V12-Acquisition",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioBackwardV12AcquisitionAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioBackwardV12AcquisitionRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Backward-V12-Acquisition-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioBackwardV12AcquisitionAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioBackwardV12AcquisitionRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Backward-V13-Expert-Start",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioBackwardV13ExpertStartAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioBackwardV13ExpertStartRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Backward-V13-Expert-Start-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioBackwardV13ExpertStartAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioBackwardV13ExpertStartRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-V14-Clearance",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardV14ClearanceAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardV14ClearanceRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-V14-Clearance-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardV14ClearanceAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardV14ClearanceRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-V15-Clearance",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardV15ClearanceAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardV15ClearanceRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-V16-Rear-Clearance",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardV16RearClearanceAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardV16RearClearanceRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-V17-Clearance-Recovery",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardV17ClearanceRecoveryAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardV17ClearanceRecoveryRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-V18-Contact-Balance",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardV18ContactBalanceAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardV18ContactBalanceRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Forward-V19-Contact-Cycle",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioForwardV19ContactCycleAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioForwardV19ContactCycleRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Omni-V20-Vision-Preserve",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioOmniV20VisionPreserveAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioOmniV20VisionPreserveRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Omni-V20-Vision-Preserve-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioOmniV20VisionPreserveAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioOmniV20VisionPreserveRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Omni-V20-Vision-Preserve-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioOmniV20VisionPreserveAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioOmniV20VisionPreserveRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Omni-V21-Matched-Acquisition",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioOmniV21MatchedAcquisitionAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioOmniV20VisionPreserveRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Omni-V21-Matched-Handoff",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioOmniV21MatchedHandoffAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioOmniV20VisionPreserveRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Omni-V21-Vision-Preserve",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioOmniV21VisionPreserveAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioOmniV20VisionPreserveRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Proprio-Omni-V21-Vision-Preserve-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_proprio_omni_env_cfg:"
            "B2Z1ProprioOmniV21VisionPreserveAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:"
            "AMPProprioOmniV20VisionPreserveRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Rear-Vision-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_rear_vision_env_cfg:B2Z1RearVisionEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMP_PPORunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-Rear-Synthetic-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_rear_vision_env_cfg:B2Z1RearSyntheticAMPEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMP_PPORunnerCfg",
    },
)


# VA-series: from-scratch AMP with performance-gated world-frame truth curves.
gym.register(
    id="Unitree-b2z1-AMP-VA1-Truth-Curve",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_va_env_cfg:B2Z1VA1TruthCurveAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA1TruthCurveRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA1-Truth-Curve-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_va_env_cfg:B2Z1VA1TruthCurveAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA1TruthCurveRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA1-Truth-Curve-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_va_env_cfg:B2Z1VA1TruthCurveAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA1TruthCurveRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA2-Truth-Curve",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_va_env_cfg:B2Z1VA2TruthCurveAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA2TruthCurveRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA2-Truth-Curve-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_va_env_cfg:B2Z1VA2TruthCurveAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA2TruthCurveRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA2-Truth-Curve-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_va_env_cfg:B2Z1VA2TruthCurveAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA2TruthCurveRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA3-Truth-Curve",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_va_env_cfg:B2Z1VA3TruthCurveAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA3TruthCurveRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA3-Truth-Curve-Visual",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_va_env_cfg:B2Z1VA3TruthCurveAMPEnvCfg_Visual"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA3TruthCurveRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA3-Truth-Curve-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_va_env_cfg:B2Z1VA3TruthCurveAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA3TruthCurveRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA4-Smooth-Tool",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_va_env_cfg:B2Z1VA4SmoothToolAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA4SmoothToolRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA4-Smooth-Tool-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_va_env_cfg:B2Z1VA4SmoothToolAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA4SmoothToolRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA5-Elbow-Wrist",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_va_env_cfg:B2Z1VA5ElbowWristAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA5ElbowWristRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA5-Elbow-Wrist-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_va_env_cfg:B2Z1VA5ElbowWristAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA5ElbowWristRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA6-Nullspace",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_va_env_cfg:B2Z1VA6NullspaceAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA6NullspaceRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA6-Nullspace-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_va_env_cfg:B2Z1VA6NullspaceAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA6NullspaceRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA7-Whole-Body",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_va_env_cfg:B2Z1VA7WholeBodyAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA7WholeBodyRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA7-Whole-Body-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.b2_z1_va_env_cfg:B2Z1VA7WholeBodyAMPEnvCfg_Play"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA7WholeBodyRunnerCfg"
        ),
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA8-Path",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_va_env_cfg:B2Z1VA8PathAMPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA8PathRunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA8-Path-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_va_env_cfg:B2Z1VA8PathAMPEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA8PathRunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA9-Preserve-Arm",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_va_env_cfg:B2Z1VA9PreserveArmAMPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA9PreserveArmRunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA9-Preserve-Arm-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_va_env_cfg:B2Z1VA9PreserveArmAMPEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA9PreserveArmRunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA10-Signed-Yaw",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_va_env_cfg:B2Z1VA10SignedYawAMPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA10SignedYawRunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA10-Signed-Yaw-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_va_env_cfg:B2Z1VA10SignedYawAMPEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA10SignedYawRunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA11-Safety",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_va_env_cfg:B2Z1VA11SafetyAMPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA11SafetyRunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA11-Safety-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_va_env_cfg:B2Z1VA11SafetyAMPEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA11SafetyRunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA12-Gait-Preserve",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_va_env_cfg:B2Z1VA12GaitPreserveAMPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA12GaitPreserveRunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA12-Gait-Preserve-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_va_env_cfg:B2Z1VA12GaitPreserveAMPEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA12GaitPreserveRunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA13-Low-Noise",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_va_env_cfg:B2Z1VA13LowNoiseAMPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA13LowNoiseRunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-AMP-VA13-Low-Noise-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_va_env_cfg:B2Z1VA13LowNoiseAMPEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA13LowNoiseRunnerCfg",
    },
)

gym.register(
    id="Unitree-b2z1-AMP-VA14-From-Scratch",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_va_env_cfg:B2Z1VA14FromScratchAMPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA14FromScratchRunnerCfg",
    },
)

gym.register(
    id="Unitree-b2z1-AMP-VA14-From-Scratch-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_va_env_cfg:B2Z1VA14FromScratchAMPEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA14FromScratchRunnerCfg",
    },
)

gym.register(
    id="Unitree-b2z1-AMP-VA15-Curriculum",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_va_env_cfg:B2Z1VA15CurriculumAMPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA15CurriculumRunnerCfg",
    },
)

gym.register(
    id="Unitree-b2z1-AMP-VA15-Curriculum-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_va_env_cfg:B2Z1VA15CurriculumAMPEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA15CurriculumRunnerCfg",
    },
)

gym.register(
    id="Unitree-b2z1-AMP-VA15-Vision-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_va15_vision_env_cfg:B2Z1VA15VisionAMPEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:AMPVA15CurriculumRunnerCfg",
    },
)
