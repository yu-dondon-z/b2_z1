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
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_env_cfg:B2Z1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "rsl_rl_symmetry_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VD3LegSymmetryRunnerCfg",
    },
)
gym.register(
    id="Unitree-b2z1-Velocity-Play",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_env_cfg:B2Z1PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "rsl_rl_symmetry_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VD3LegSymmetryRunnerCfg",
    },
)

gym.register(
    id="Unitree-b2z1-VD4-Velocity",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_vd4_env_cfg:B2Z1VD4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VD4PPORunnerCfg",
    },
)

gym.register(
    id="Unitree-b2z1-VD4-Velocity-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_vd4_env_cfg:B2Z1VD4PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VD4PPORunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-VD5-Velocity",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_vd5_env_cfg:B2Z1VD5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VD5PPORunnerCfg",
    },
)


gym.register(
    id="Unitree-b2z1-VD5-Velocity-Play",
    entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2_z1_vd5_env_cfg:B2Z1VD5PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VD5PPORunnerCfg",
    },
)


for _suffix, _config in (("", "B2Z1VD5StartupEnvCfg"), ("-Play", "B2Z1VD5StartupPlayEnvCfg")):
    gym.register(
        id=f"Unitree-b2z1-VD5-Startup-Velocity{_suffix}",
        entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.b2_z1_vd5_startup_env_cfg:{_config}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VD5StartupPPORunnerCfg",
        },
    )
