# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
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

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# Pre-defined configs
##

from b2_z1.assets.unitree import B2_Z1_GUN_CFG, B2_Z1_NO_STICK_CFG
# from b2_z1.assets.unitree import B2_Z1_CFG

B2_JOINT_NAMES = [".*_hip_joint", ".*_calf_joint", ".*_thigh_joint"]
Z1_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

GROUND_MATERIAL_CFG = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    static_friction=1.0,
    dynamic_friction=1.0,
)

##
# Scene definition
##


@configclass
class B2Z1SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        # A local static cuboid avoids the remote USD used by GroundPlaneCfg.
        # Without a TerrainImporter, InteractiveScene keeps its unique grid
        # origins for every parallel environment.
        spawn=sim_utils.CuboidCfg(
            size=(200.0, 200.0, 0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.12, 0.12),
                roughness=0.9,
            ),
            physics_material=GROUND_MATERIAL_CFG,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
        collision_group=-1,
    )

    # robot
    # robot: ArticulationCfg = B2_Z1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot: ArticulationCfg = B2_Z1_NO_STICK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
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

    JointPositionAction = mdp.FilteredJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        alpha=0.20,
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
        
        ee_link0_rel_pose = ObsTerm(
            func=mdp.end_effector_relative_pose,
        )
        
        
    
    # privileged observations
    critic: CriticCfg = CriticCfg()
        

    
    @configclass
    class AMPObsCfg(ObsGroup):
        # amp dataset observations
        # joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos
        
        joint_pos = ObsTerm(
            func=mdp.joint_pos_reindex,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=B2_JOINT_NAMES)},
        )
        foot_pos = ObsTerm(
            func=mdp.foot_pos,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
        )
        
        root_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
        )
        root_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
        )
        
        joint_vel = ObsTerm(
            func=mdp.joint_vel_reindex,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=B2_JOINT_NAMES)},
        )
        
        base_z_pos = ObsTerm(
            func=mdp.base_pos_z,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
        )
        
    
    # AMP observations
    amp_obs: AMPObsCfg = AMPObsCfg()
    

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

    # Sim2Sim robustness: the Z1 payload and the contact model move the whole
    # robot CoM slightly differently in MuJoCo. Train across that offset
    # instead of relying on one exact PhysX mass distribution.
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {
                "x": (-0.04, 0.04),
                "y": (-0.04, 0.04),
                "z": (-0.02, 0.02),
            },
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.80, 1.20),
            "damping_distribution_params": (0.60, 1.40),
            "operation": "scale",
        },
    )

    joint_dynamics = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.08),
            "armature_distribution_params": (0.0, 0.12),
            "operation": "abs",
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
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "roll": (-0.10, 0.10),
                "pitch": (-0.10, 0.10),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (-0.10, 0.10),
                "roll": (-0.30, 0.30),
                "pitch": (-0.30, 0.30),
                "yaw": (-0.15, 0.15),
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
        interval_range_s=(3.0, 6.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.7, 0.7),
                "roll": (-0.45, 0.45),
                "pitch": (-0.35, 0.35),
                "yaw": (-0.25, 0.25),
            }
        },
    )
    

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    
    # base_velocity = mdp.command_cfg.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.33,
    #     debug_vis=True,
    #     heading_command=False,
    #     ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-0.0, 0.0),
    #         lin_vel_y=(-0.0, 0.0),
    #         ang_vel_z=(-0.0, 0.0),
    #         heading=(-0.0, 0.0),
    #     ),
    #     limit_ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.0, 1.0),
    #         lin_vel_y=(-1.0, 1.0),
    #         ang_vel_z=(-1.0, 1.0),
    #         heading=(-0.0, 0.0),
    #     ),
    # )
    
    base_velocity = mdp.ee_vel_command.EEDerivedVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        # rel_standing_envs=0.33,
        rel_standing_envs=0.0,
        # Training does not need the remote arrow USD marker.
        debug_vis=False,
        heading_command=False,
        arrival_threshold=0.7,
        ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.0, 0.0),
            lin_vel_y=(-0.0, 0.0),
            ang_vel_z=(-0.0, 0.0),
            heading=(-0.0, 0.0),
        ),
        limit_ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-0.0, 0.0),
        ),
        
        # limit_ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
        #     lin_vel_x=(-0.6, 0.6),
        #     lin_vel_y=(-0.6, 0.6),
        #     ang_vel_z=(-0.5, 0.5),
        #     heading=(-0.0, 0.0),
        # ),
    )
    
    
    ee_target = mdp.command_cfg.UniformPoseWBCCommandCfg(
        asset_name="robot",
        body_name="dummy_target",
        resampling_time_range=(8.0, 10.0),
        debug_vis=False,
        ranges=mdp.command_cfg.UniformPoseWBCCommandCfg.Ranges(     
            # pos_x=(0.4, 0.5),
            # pos_y=(-0.15, 0.15),
            # pos_z=(0.15, 0.15),  # World frame
            
            
            # roll=(-0.0, 0.0),
            # pitch=(3.14/2, 3.14/2),  # depends on end-effector axis
            # yaw=(-0.0, -0.0),
            
            # Start with a one-metre-long, centred path. Curriculum later
            # extends only the forward endpoint.
            pos_x=(2.0, 2.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.10, 0.10),  # World frame
            
            
            roll=(-0.0, 0.0),
            pitch=(3.14/2, 3.14/2),  # depends on end-effector axis
            yaw=(-0.0, -0.0),
            
        ),
        limit_ranges=mdp.command_cfg.UniformPoseWBCCommandCfg.Ranges(
            # pos_x=(0.35, 0.55),
            # pos_y=(-0.5, 0.5),
            # pos_z=(0.0, 0.2),  # World frame
            
            # roll=(-0.0, 0.0),
            # pitch=(3.14/2, 3.14/2),  # depends on end-effector axis
            # yaw=(-0.0, -0.0),   
            
            
            pos_x=(2.0, 6.0),
            # pos_x=(-4.0, 6.0),
            # Keep the tool trajectory on B2's longitudinal centerline.
            pos_y=(0.0, 0.0),
            pos_z=(0.10, 0.10),  # World frame
            
            
            roll=(-0.0, 0.0),
            pitch=(3.14/2, 3.14/2),  # depends on end-effector axis
            yaw=(-0.0, -0.0),
            
        ),    
        trajectory_waypoint_count=3,
        # Zero offset makes all intermediate waypoints lie on the straight
        # centreline from the fixed start to the forward target.
        trajectory_waypoint_offset_range_m=(0.0, 0.0),
        # link00 is centred on base_link. Starting one metre ahead leaves
        # clearance for B2 and the Z1 arm before the tracked line begins.
        trajectory_start_pos=(1.0, 0.0, 0.10),
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
    
    

    
    ########## legged_manip
    
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
    
    
    # end_effector_orientation_tracking = RewTerm(
    #     func=mdp.orientation_command_error,
    #     weight=-1.5,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="dummy_target"),
    #         "command_name": "ee_target",
    #     },
    # )
    
    
    
    
    
    
    ee_stability = RewTerm(
        func=mdp.link_stability,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="dummy_target"),
            # "link_name": "link06",
            "link_name": "link_gun",
        },
    )

    
    
    

    # -- base
    
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # Directly penalize accumulated roll/pitch. Angular-velocity penalties
    # alone allow a slow lean that looks acceptable in PhysX but eventually
    # tips over in MuJoCo.
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "target_height": 0.45,
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
    
    # joint_deviation = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.02,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf_joint", ".*_thigh_joint"])})
    
    
    
    arm_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=Z1_JOINT_NAMES
            ),
        },
    )
    
    # joint_mirror = RewTerm(
    #     func=mdp.joint_mirror,
    #     weight=-0.15,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
    #     },
    # )
    

    
    # -- feet
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    
    # feet_long_air = RewTerm(
    #     func=mdp.feet_long_air_penalty,
    #     weight=-0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "max_air_time": 1.0,
    #     },
    # )
    
    # air_time_variance = RewTerm(
    #     func=mdp.air_time_variance_penalty,
    #     weight=-1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    # )
    
    # feet_height = RewTerm(
    #     func=mdp.swing_leg_height_reward,
    #     weight=1.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    #         "target_height": 0.15,
    #     },
    # )
    
    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    #     },
    # )
    
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
    
    
    
    # feet_gait = RewTerm(
    #     func=mdp.GaitReward,
    #     weight=0.0,
    #     params={
    #         "std": math.sqrt(0.25),
    #         "max_err": 0.2,
    #         "velocity_threshold": 0.2,
    #         "synced_feet_pair_names": (("", ""), ("", "")),
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "sensor_cfg": SceneEntityCfg("contact_forces"),
    #     },
    # )
    
    
    
    # stand_still_contact = RewTerm(
    #     func=mdp.stand_still_contact_reward,
    #     weight=1.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "velocity_threshold": 0.1,
    #         "contact_force_threshold": 5.0,
    #         "target_contacts": 4,
    #     },
    # )

    # stand_still_air = RewTerm(
    #     func=mdp.stand_still_feet_air_penalty,
    #     weight=-3.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "velocity_threshold": 0.1,
    #         "max_air_time": 0.15,
    #     },
    # )
    
    stand_still_load_balance = RewTerm(
        func=mdp.stand_still_load_balance_penalty,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "velocity_threshold": 0.1,
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
     
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    
    
    undesired_contacts = RewTerm(
        func=mdp.uncontacts,
        weight=-1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names="link_gun")},
    )


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
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)  # type: ignore
    ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels)  # type: ignore
    # pos_cmd_levels = CurrTerm(func=mdp.pos_cmd_levels)  # type: ignore
    
    
    vel_cmd_to_pos_levels = CurrTerm(
        func=mdp.vel_cmd_to_ee_pos_levels,
        params={"reward_threshold": 0.65}
    )  # type: ignore    

##
# Environment configuration
##


@configclass
class B2Z1AMPEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: B2Z1SceneCfg = B2Z1SceneCfg(num_envs=1024, env_spacing=4.0)
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
        self.sim.physics_material = GROUND_MATERIAL_CFG
        
        # PhysX settings
        self.sim.physx.solver_type = 1  # TGS solver
        self.sim.physx.max_position_iteration_count = 4 
        self.sim.physx.max_velocity_iteration_count = 0 
        self.sim.physx.bounce_threshold_velocity = 0.5


        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        
        
        ######### AMP Settings ###########
        self.reference_state_initialization = True
        # Keep AMP motion starts, but also teach recovery from ordinary
        # standing and perturbed states required by deployment.
        self.reference_state_initialization_prob = 0.5
        
        if self.reference_state_initialization:
            from .agents.rsl_rl_amp_cfg import MOTION_FILES
            self.amp_motion_files = MOTION_FILES
        else:
            self.amp_motion_files = None
            
        
        ######### Reward Settings #########
          
        # -- task rewards

        # Stage two: retain locomotion tracking while explicitly teaching the
        # arm endpoint to follow the centred forward trajectory.
        self.rewards.track_lin_vel_xy.weight = 2.5
        self.rewards.track_ang_vel_z.weight = 1.5
        self.rewards.end_effector_position_tracking_exp.weight = 4.0
        self.rewards.end_effector_position_tracking_fine_grained.weight = 2.5
        
        
        
        self.rewards.ee_stability.weight = -0.1 #-0.2 
        
        
        
        # -- base rewards
        self.rewards.base_height_l2.params["target_height"] = 0.45
        self.rewards.base_height_l2.weight = -20.0  # -30.0
        
        self.rewards.base_linear_velocity.weight = -3.0 # -2.5
        self.rewards.base_angular_velocity.weight = -0.5 #-0.25
        self.rewards.flat_orientation_l2.weight = -3.0
        
        
        
        # -- joint rewards
        self.rewards.hip_deviation.weight = -0.5
        # self.rewards.joint_deviation.weight = -0.005 # -0.01
        # A strong -3.0 penalty prevents the arm from leaving its nominal
        # posture and competes directly with endpoint tracking.
        self.rewards.arm_deviation.weight = -0.5
        # self.rewards.arm_deviation = None
        
        # self.rewards.joint_mirror.weight = -0.3 # -0.25
        
        
        
        # -- feet rewards
        # self.rewards.feet_long_air.params["max_air_time"] = 0.6
        # self.rewards.feet_long_air.weight = -1.5
        
        self.rewards.feet_air_time.weight = 0.75 # 1.0 
        # self.rewards.air_time_variance.weight = -1.5 #-1.0
        # self.rewards.feet_height.params["target_height"] = 0.15
        # self.rewards.feet_height.weight = 0.15 # 0.1
        
        # self.rewards.feet_slide.weight = -1.0 # -1.0
        # self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        # self.rewards.feet_gait.weight = 1.5 #1.0
        
        # self.rewards.stand_still_contact.weight = 1.75 # 1.5
        # self.rewards.stand_still_air.weight = -4.5 #-4.0
        self.rewards.stand_still_load_balance.weight = -1.5 # -1.0
        
        
        
        # -- misc rewards
        self.rewards.joint_power.weight = -2.5e-5
        self.rewards.dof_torques_l2.weight = -1.25e-5
        
        self.rewards.undesired_contacts.weight = -1.0
        
        
        self.commands.base_velocity.curriculum_enabled = True
        # self.commands.base_velocity.curriculum_enabled = False
        self.commands.ee_target.curriculum_enabled = True
        # self.commands.ee_target.curriculum_enabled = False
        
        
        
                 
@configclass
class B2Z1AMPEnvCfg_Play(B2Z1AMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # Keep each playback episode running long enough to inspect sustained
        # locomotion and end-effector tracking before the automatic reset.
        self.episode_length_s = 60.0

        # The centerline checkpoints in
        # 2026-07-28_09-31-44_ee_centerline_stage2 were trained with direct
        # joint-position actions.  The EMA action filter below was added later
        # for MuJoCo sim-to-sim work and changes both the applied action and the
        # previous-action observation, so it must not be used to play those
        # historical checkpoints.
        self.actions.JointPositionAction = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.25,
            use_default_offset=True,
            clip={".*": (-100.0, 100.0)},
        )

        # Use nominal dynamics for deterministic policy inspection.  These
        # randomizations were introduced after the centerline run and are
        # useful for new sim-to-sim training, not for evaluating the saved
        # PhysX checkpoint under its original action/dynamics assumptions.
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_dynamics = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.reference_state_initialization_prob = 1.0
        
        
        # self.commands.base_velocity.limit_ranges.lin_vel_x = (-0.6, 0.6)
        # self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.6, 0.6)
        # self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.5, 0.5)
        
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        
        
        
        # self.commands.ee_target.limit_ranges.pos_x = (1.0, 1.45)
        # self.commands.ee_target.limit_ranges.pos_y = (-0.0, 0.0)
        # self.commands.ee_target.limit_ranges.pos_z = (0.0, 0.02)
        
        self.commands.ee_target.limit_ranges.pos_x = (2.0, 6.0)
        self.commands.ee_target.limit_ranges.pos_y = (0.0, 0.0)
        self.commands.ee_target.limit_ranges.pos_z = (0.10, 0.10)
        
        self.commands.ee_target.ranges = self.commands.ee_target.limit_ranges
        # Show the desired straight centerline and the actual end-effector
        # trace during playback. Training keeps this disabled for performance.
        self.commands.ee_target.debug_vis = True
        
        
        
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class B2Z1AMPGunEnvCfg_Play(B2Z1AMPEnvCfg_Play):
    """Trajectory-tracking AMP playback with the cylindrical welding gun."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = B2_Z1_GUN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
