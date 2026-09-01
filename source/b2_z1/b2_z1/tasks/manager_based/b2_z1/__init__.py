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
