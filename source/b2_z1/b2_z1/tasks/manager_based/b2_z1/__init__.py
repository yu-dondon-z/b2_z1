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

for _suffix, _config in (("", "B2Z1VD7EnvCfg"), ("-Play", "B2Z1VD7PlayEnvCfg")):
    gym.register(
        id=f"Unitree-b2z1-VD7-Velocity{_suffix}",
        entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.b2_z1_vd7_env_cfg:{_config}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VD7PPORunnerCfg",
        },
    )


for _suffix, _config in (("", "B2Z1VD8EnvCfg"), ("-Play", "B2Z1VD8PlayEnvCfg")):
    gym.register(
        id=f"Unitree-b2z1-VD8-Velocity{_suffix}",
        entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.b2_z1_vd8_env_cfg:{_config}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VD8PPORunnerCfg",
        },
    )

for _suffix, _config in (("", "B2Z1VD9EnvCfg"), ("-Play", "B2Z1VD9PlayEnvCfg")):
    gym.register(
        id=f"Unitree-b2z1-VD9-Velocity{_suffix}",
        entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.b2_z1_vd9_env_cfg:{_config}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VD9PPORunnerCfg",
        },
    )


for _suffix, _config in (("", "B2Z1VD10EnvCfg"), ("-Play", "B2Z1VD10PlayEnvCfg")):
    gym.register(
        id=f"Unitree-b2z1-VD10-Velocity{_suffix}",
        entry_point="b2_z1.envs:B2Z1VD10ManagerAMPBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.b2_z1_vd10_env_cfg:{_config}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VD10PPORunnerCfg",
        },
    )

for _suffix, _config in (("", "B2Z1VD11EnvCfg"), ("-Play", "B2Z1VD11PlayEnvCfg")):
    gym.register(
        id=f"Unitree-b2z1-VD11-Velocity{_suffix}",
        entry_point="b2_z1.envs:B2Z1VD10ManagerAMPBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.b2_z1_vd11_env_cfg:{_config}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VD11PPORunnerCfg",
        },
    )


for _suffix, _config in (("", "B2Z1VD12EnvCfg"), ("-Play", "B2Z1VD12PlayEnvCfg")):
    gym.register(
        id=f"Unitree-b2z1-VD12-Velocity{_suffix}",
        entry_point="b2_z1.envs:B2Z1VD10ManagerAMPBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.b2_z1_vd12_env_cfg:{_config}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VD12PPORunnerCfg",
        },
    )

for _suffix, _config in (("", "B2Z1VD13EnvCfg"), ("-Play", "B2Z1VD13PlayEnvCfg")):
    gym.register(
        id=f"Unitree-b2z1-VD13-Velocity{_suffix}",
        entry_point="b2_z1.envs:B2Z1VD10ManagerAMPBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.b2_z1_vd13_env_cfg:{_config}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VD13PPORunnerCfg",
        },
    )


for _suffix, _config in (("", "B2Z1VD14EnvCfg"), ("-Play", "B2Z1VD14PlayEnvCfg")):
    gym.register(
        id=f"Unitree-b2z1-VD14-Velocity{_suffix}",
        entry_point="b2_z1.envs:B2Z1VD10ManagerAMPBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.b2_z1_vd14_env_cfg:{_config}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VD14PPORunnerCfg",
        },
    )


for _suffix, _config in (("", "B2Z1VD6EnvCfg"), ("-Play", "B2Z1VD6PlayEnvCfg")):
    gym.register(
        id=f"Unitree-b2z1-VD6-Velocity{_suffix}",
        entry_point="b2_z1.envs:B2Z1ManagerAMPBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.b2_z1_vd6_env_cfg:{_config}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VD6PPORunnerCfg",
        },
    )
