"""Playback configuration for the original B2-Z1 welding-gun PPO policy."""

from isaaclab.assets import ArticulationCfg
from isaaclab.envs.mdp.commands import UniformPoseCommandCfg, UniformVelocityCommandCfg
from isaaclab.utils import configclass

from b2_z1.assets.unitree import B2_Z1_GUN_LEGACY_CFG

from .b2_z1_env_cfg import B2Z1EnvCfg, B2Z1SceneCfg


@configclass
class B2Z1GunLegacySceneCfg(B2Z1SceneCfg):
    """Scene using the URDF whose end effector includes ``link_gun``."""

    robot: ArticulationCfg = B2_Z1_GUN_LEGACY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class B2Z1GunLegacyCommandsCfg:
    """Command generators and ranges used by the June 2026 gun checkpoint."""

    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.25,
        debug_vis=True,
        heading_command=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.5),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(0.0, 0.0),
        ),
    )

    ee_target = UniformPoseCommandCfg(
        asset_name="robot",
        body_name="dummy_target",
        resampling_time_range=(2.0, 4.0),
        debug_vis=True,
        ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(0.6, 0.75),
            pos_y=(-0.4, 0.4),
            pos_z=(-0.4, -0.1),
            roll=(0.0, 0.0),
            pitch=(0.7854, 1.5708),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class B2Z1GunLegacyPlayEnvCfg(B2Z1EnvCfg):
    """Exact observation/model interface expected by the 70/94/18 policy."""

    scene: B2Z1GunLegacySceneCfg = B2Z1GunLegacySceneCfg(num_envs=1, env_spacing=4.0)
    commands: B2Z1GunLegacyCommandsCfg = B2Z1GunLegacyCommandsCfg()
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1

        # The original checkpoint critic has 94 inputs.  This 7-D privileged
        # observation was introduced later and is not part of that checkpoint.
        self.observations.critic.ee_link0_rel_pose = None
