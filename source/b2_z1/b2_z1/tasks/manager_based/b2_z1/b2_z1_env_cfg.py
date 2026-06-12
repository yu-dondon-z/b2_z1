# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# Pre-defined configs
##

from b2_z1.assets.unitree import B2_Z1_GUN_CFG
# from b2_z1.assets.unitree import B2_Z1_CFG



##
# Scene definition
##


@configclass
class B2Z1SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # robot
    # robot: ArticulationCfg = B2_Z1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot: ArticulationCfg = B2_Z1_GUN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, 
                                      track_air_time=True, debug_vis=False)
    
    

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
        target_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "ee_target"})
        
        
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
    
    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100), noise=Unoise(n_min=-0.1, n_max=0.1))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        
        base_external_force = ObsTerm(
            func=mdp.base_external_force,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
            scale=0.1,
            clip=(-100, 100),
        )
        
        
        # feet_contact = ObsTerm(
        #     func=mdp.contact,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")
        #     },
        #     clip=(0.0, 1.0),
        #     scale=0.1,
        # )
        ee_link0_rel_pose = ObsTerm(
            func=mdp.end_effector_relative_pose,
        )
        
        
    
    # privileged observations
    critic: CriticCfg = CriticCfg()
        


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
    

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    
    base_velocity = mdp.command_cfg.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.33,
        debug_vis=True,
        heading_command=False, 
        
        ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.0, 0.0),
            lin_vel_y=(-0.0, 0.0),
            ang_vel_z=(-0.0, 0.0),
            heading=(-0.0, 0.0),
        ),
        limit_ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 0.6),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-0.6, 0.6),
            heading=(-0.0, 0.0),
        ),
    )
    
    
    ee_target = mdp.command_cfg.UniformPoseWBCCommandCfg(
        asset_name="robot",
        body_name="dummy_target",
        resampling_time_range=(8.0, 10.0),
        debug_vis=True,
        ranges=mdp.command_cfg.UniformPoseWBCCommandCfg.Ranges(
            # pos_x=(0.4, 0.45),
            # pos_y=(-0.05, 0.05),
            # pos_z=(0.3, 0.3),  # World frame
            
            pos_x=(0.4, 0.5),
            pos_y=(-0.15, 0.15),
            pos_z=(0.2, 0.4),  # World frame
            
            
            roll=(-0.0, 0.0),
            pitch=(3.14/2, 3.14/2),  # depends on end-effector axis
            yaw=(-0.0, -0.0),
        ),
        limit_ranges=mdp.command_cfg.UniformPoseWBCCommandCfg.Ranges(
            # pos_x=(0.55, 0.85),
            # pos_y=(-0.35, 0.35),
            # pos_z=(0.1, 1.0),  # World frame
            
            pos_x=(0.35, 0.7),
            pos_y=(-0.6, 0.6),
            # pos_z=(0.0, 0.3),  # World frame
            pos_z=(0.0, 0.4),  # World frame
            
            
            # roll=(-3.14 / 3, 3.14 / 3),
            # pitch=(-3.14 / 4, 3.14 / 4),  # depends on end-effector axis
            # yaw=(-3.14 / 6, 3.14 / 6),
            
            roll=(-0.0, 0.0),
            pitch=(3.14/2, 3.14/2),  # depends on end-effector axis
            yaw=(-0.0, -0.0),   
        ),    
    )
    



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=1.5, 
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=0.75, 
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    
    
    
    end_effector_position_tracking_exp = RewTerm(
        func=mdp.position_command_error_exp,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="dummy_target"),
            "command_name": "ee_target",
            "std": math.sqrt(0.1),
        },
    )
    
    
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="dummy_target"),
            "std": 0.1,
            "command_name": "ee_target",
        },
    )
    
    
    
    ee_stability = RewTerm(
        func=mdp.link_stability,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="dummy_target"),
            "link_name": "link06",
        },
    )

    
    
    

    # -- base
    
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    # base_height_exp = RewTerm(
    #     func=mdp.base_height_tracking, 
    #     weight=1.0,
    #     params={ 
    #         "target_height": 0.3,  
    #         "std": 0.02
    #     }
    # )
    
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "target_height": 0.48,
        },
    )
    
    
    # -- joints
    
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    
    # hip_torques_max = RewTerm(
    #     func=mdp.joint_torques_max, params={"joint_names": ".*_hip_joint"}, weight=-8.0e-5
    # )  # - 0.0002
    # thigh_torques_max = RewTerm(
    #     func=mdp.joint_torques_max, params={"joint_names": ".*_thigh_joint"}, weight=-8.0e-5
    # )  # - 0.0002
    # calf_torques_max = RewTerm(
    #     func=mdp.joint_torques_max, params={"joint_names": ".*_calf_joint"}, weight=-2.0e-5
    # )  # - 0.0002
    
    
    
    hip_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])})
    
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf_joint", ".*_thigh_joint"])})
    
    
    
    # arm_deviation = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", joint_names=[ "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    #         ),
    #     },
    # )
    
    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
        },
    )
    

    
    # -- feet
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=0.25,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.5,
    #     },
    # )
    
    feet_long_air = RewTerm(
        func=mdp.feet_long_air_penalty,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "max_air_time": 1.0,
        },
    )
    
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    
    feet_height = RewTerm(
        func=mdp.swing_leg_height_reward,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_height": 0.15,
        },
    )
    
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    
    # feet_clearance = RewTerm(
    #     func=mdp.foot_clearance_reward,
    #     weight=0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    #         "target_height": 0.2,
    #         "std": 0.1,
    #         "tanh_mult": 1.0,
    #     },
    # )
    
    
    
    feet_gait = RewTerm(
        # func=mdp.GaitRewardV2,
        func=mdp.GaitReward,
        weight=0.0,
        params={
            "std": math.sqrt(0.25),
            "max_err": 0.2,
            "velocity_threshold": 0.2,
            "synced_feet_pair_names": (("", ""), ("", "")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            # "front_sensor_cfg": SceneEntityCfg("front_scanner"),
        },
    )
    
    
    
    
    
    
    
    
    # -- Misc
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    
    joint_power = RewTerm(
        func=mdp.power_penalty,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    action_smoothness = RewTerm(func=mdp.smoothness, weight=-0.01)
    
    
    # undesired_contacts = RewTerm(
    #     func=mdp.uncontacts,
    #     weight=-1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names="link_gun")},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        # params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
        
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*_calf"]), "threshold": 1.0},
        
        
    )
    # ".*_thigh"
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)  # type: ignore
    ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels)  # type: ignore
    pos_cmd_levels = CurrTerm(func=mdp.pos_cmd_levels)  # type: ignore
    

##
# Environment configuration
##


@configclass
class B2Z1EnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: B2Z1SceneCfg = B2Z1SceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # # viewer settings
        # self.viewer.eye = (8.0, 0.0, 5.0)

        # simulation settings 
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        
        # PhysX settings
        self.sim.physx.solver_type = 1  # TGS solver
        self.sim.physx.max_position_iteration_count = 4 
        self.sim.physx.max_velocity_iteration_count = 0 
        self.sim.physx.bounce_threshold_velocity = 0.5


        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        
        
        
        ######### Reward Settings #########
          
        # -- task rewards
        self.rewards.track_lin_vel_xy.weight = 4.0
        self.rewards.track_ang_vel_z.weight = 3.5 #2.5
        
        self.rewards.end_effector_position_tracking_exp.weight = 4.5 
        self.rewards.end_effector_position_tracking_fine_grained.weight = 2.75
        
        self.rewards.ee_stability.weight = -0.1
        
        
        # -- base rewards
        
        self.rewards.base_height_l2.params["target_height"] = 0.48
        self.rewards.base_height_l2.weight = -15.0  # -30.0
        self.rewards.flat_orientation_l2.weight = -3.0 # -5.0
        
        self.rewards.base_linear_velocity.weight = -2.5
        self.rewards.base_angular_velocity.weight = -0.15 #-0.1
        
        
        # -- joint rewards
        
        self.rewards.hip_deviation.weight = -0.2
        self.rewards.joint_deviation.weight = -0.01# -0.005
        # self.rewards.arm_deviation.weight = -0.1
        
        
        
        # -- feet rewards
        self.rewards.feet_long_air.params["max_air_time"] = 0.6
        self.rewards.feet_long_air.weight = -1.5
        
        self.rewards.air_time_variance.weight = -1.0
        self.rewards.feet_height.params["target_height"] = 0.15
        self.rewards.feet_height.weight = 0.15
        
        self.rewards.feet_slide.weight = -1.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.feet_gait.weight = 1.0
        
        
        
        # -- misc rewards
        self.rewards.joint_power.weight = -2.5e-5
        self.rewards.dof_torques_l2.weight = -1.25e-5
        
    
        
        ######### Command Settings #########
        
        self.commands.base_velocity.curriculum_enabled = True
        self.commands.ee_target.curriculum_enabled = True
        
        
        
                 
@configclass
class B2Z1PlayEnvCfg(B2Z1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.4)
        
        
        self.commands.ee_target.ranges.pos_x = (0.4, 0.5)
        self.commands.ee_target.ranges.pos_y = (-0.4, 0.4)
        self.commands.ee_target.ranges.pos_z = (0.0, 0.05)
        
        # self.commands.ee_target.ranges = self.commands.ee_target.limit_ranges
        
        
        
        