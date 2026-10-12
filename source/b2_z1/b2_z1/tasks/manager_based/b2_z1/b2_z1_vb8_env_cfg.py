"""VB8: retain AMP/load/path tasks, target torso heave and loaded-gun shake."""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from .b2_z1_vb7_env_cfg import B2Z1VB7LoadRobustAMPEnvCfg
from .b2_z1_vb5_env_cfg import (B2Z1VB5PayoutStage2AMPEnvCfg_Play,
                               B2Z1VB5PayoutVisionAMPEnvCfg_Play)
from .b2_z1_va_env_cfg import VA12GaitPreserveRewardsCfg
from .mdp.loaded_tool_stability import (base_world_heave_cost,
                                       gun_world_tilt_rate_cost,
                                       gun_vertical_axis_cost,
                                       arm_limit_buffer_cost)


@configclass
class VB8QuietLoadedToolRewardsCfg(VA12GaitPreserveRewardsCfg):
    base_world_heave = RewTerm(
        func=base_world_heave_cost, weight=-0.35,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "speed_scale": 0.10})
    gun_world_tilt_rate = RewTerm(
        func=gun_world_tilt_rate_cost, weight=-0.20,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="link_gun"),
                "speed_scale": 0.50})
    gun_vertical_axis = RewTerm(
        func=gun_vertical_axis_cost, weight=-0.30,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="link_gun")})
    arm_limit_buffer = RewTerm(
        func=arm_limit_buffer_cost, weight=-0.40,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[f"joint{i}" for i in range(1, 7)]),
                "margin_rad": 0.10})


@configclass
class B2Z1VB8QuietLoadAMPEnvCfg(B2Z1VB7LoadRobustAMPEnvCfg):
    rewards: VB8QuietLoadedToolRewardsCfg = VB8QuietLoadedToolRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_target.tool_orientation_gravity_aligned = True
        # Legacy cost subtracts torso acceleration: it can penalize exactly
        # the compensation that keeps the gun still in the world. Retain
        # world TCP/acceleration/nullspace objectives, remove this conflict.
        self.rewards.ee_stability.weight = 0.0
        # Still one actor for all 18 joints; never freeze useful arm motion.
        self.rewards.tcp_cross_path_velocity.weight = -0.40
        self.rewards.tcp_cross_path_velocity.params["vertical_scale"] = 0.10
        self.rewards.arm_action_rate_l2.weight = -0.16
        self.rewards.base_angular_velocity.weight = -1.25
        # Height/gait/clearance/heading/AMP/terminations and VB7 load domains
        # stay intact. Do not make success gates easier to hide failures.


@configclass
class B2Z1VB8QuietLoadAMPEnvCfg_Play(B2Z1VB5PayoutStage2AMPEnvCfg_Play):
    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_target.tool_height_m = 0.01
        self.commands.ee_target.tool_height_range_m = (0.01, 0.01)
        self.commands.ee_target.tool_orientation_gravity_aligned = True


@configclass
class B2Z1VB8QuietLoadVisionAMPEnvCfg_Play(B2Z1VB5PayoutVisionAMPEnvCfg_Play):
    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_target.tool_orientation_gravity_aligned = True
