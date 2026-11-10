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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, TiledCameraCfg, patterns

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .mdp.commands.curve_camera import make_curve_camera_cfg, make_rear_camera_cfg

##
# Pre-defined configs
##

from b2_z1.amp_config import AMP_FOOT_NAMES, AMP_JOINT_NAMES, MOTION_FILES
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
    
    # 20 physics samples at 200 Hz cover the 0.1 s tool-support window.
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=20,
                                      track_air_time=True, debug_vis=False)

    # Populated from commands.base_velocity calibration: USD prim in training, tiled RGB sensor in Play.
    base_camera: AssetBaseCfg | TiledCameraCfg | None = None
    # Rear EE camera uses commands.ee_target calibration and the same rendering modes.
    rear_camera: AssetBaseCfg | TiledCameraCfg | None = None
    
    

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
        velocity_commands = ObsTerm(func=mdp.generated_commands, 
                                    clip=(-100, 100), params={"command_name": "base_velocity"})
        
        target_commands = ObsTerm(func=mdp.generated_commands, 
                                  clip=(-100, 100), params={"command_name": "ee_target"})
        
        
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
    class AMPObsCfg(ObsGroup):
        """43 motion features in amp_loco dataset order; no arm observations or noise."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=AMP_JOINT_NAMES, preserve_order=True)},
        )
        foot_pos = ObsTerm(
            func=mdp.amp_foot_pos,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=AMP_FOOT_NAMES, preserve_order=True)},
        )
        root_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        root_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=AMP_JOINT_NAMES, preserve_order=True)},
        )
        base_z_pos = ObsTerm(func=mdp.amp_base_height, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

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
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]),
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # Keep leg reference-state initialization, without injecting an unrelated
    # random initial angular velocity into the six arm joints.
    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[f"joint{i}" for i in range(1, 7)]),
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
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
    
    base_velocity = mdp.command_cfg.CurveVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        forward_speed=0.2,
        rel_standing_envs=0.2,
        lookahead_distance=0.6,
        path_type="turn",
        trajectory_num_points=192,
        turn_count_range=(2, 3),
        max_curvature=1.0,
        trajectory_curve_offset_scale=0.0,
        limit_trajectory_curve_offset_scale=0.1,
        debug_vis=False,
        ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.2),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.8, 0.8),
        ),
        limit_ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.2),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.8, 0.8),
        ),
    )
    
    # yaw_velocy = mdp.command_cfg.UniformVelocityCommandCfg()
    
    
    ee_target = mdp.command_cfg.UniformPoseWBCCommandCfg(
        asset_name="robot",
        body_name="dummy_target",
        base_name="base_link",
        resampling_time_range=(8.0, 10.0),
        debug_vis=True,
        # All XYZ/RPY values are in base_link. The rear-mounted arm works at negative X.
        ranges=mdp.command_cfg.UniformPoseWBCCommandCfg.Ranges(
            pos_x=(-1.03, -1.01),
            pos_y=(-0.02, 0.02),
            pos_z=(-0.31, -0.29),
            roll=(0.0, 0.0),
            pitch=(math.pi / 2 - 0.45, math.pi / 2 - 0.45),
            yaw=(math.pi, math.pi),
        ),
        limit_ranges=mdp.command_cfg.UniformPoseWBCCommandCfg.Ranges(
            pos_x=(-1.08, -0.96),
            pos_y=(-0.15, 0.15),
            pos_z=(-0.34, -0.26),
            roll=(0.0, 0.0),
            pitch=(math.pi / 2 - 0.45, math.pi / 2 - 0.45),
            yaw=(math.pi, math.pi),
        ),
    )



@configclass
class RewardsCfg:
    """Task tracking and physical regularization; gait style comes from AMP."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=6.0,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=2.5,
        params={"command_name": "base_velocity", "std": 0.3},
    )
    
    
    end_effector_position_tracking_exp = RewTerm(
        func=mdp.position_command_error_exp, weight=5.0,
        params={
            "command_name": "ee_target", "std": 0.07,
            "asset_cfg": SceneEntityCfg("robot", body_names="link06"),
            "base_name": "base_link",
        },
    )
    end_effector_orientation_error = RewTerm(
        func=mdp.tcp_orientation_command_error, weight=-0.4,
        params={"command_name": "ee_target", "asset_cfg": SceneEntityCfg("robot", body_names="link06")},
    )
    
    
    # A long tool can pivot around a steady TCP while its flange swings. Track
    # link06 against the flange pose/twist implied by the desired TCP pose.
    flange_position_residual = RewTerm(
        func=mdp.flange_position_residual, weight=-0.25,
        params={"command_name": "ee_target", "scale": 0.05, "deadzone": 0.005, "max_penalty": 4.0},
    )
    flange_linear_velocity_residual = RewTerm(
        func=mdp.flange_linear_velocity_residual, weight=-0.3,
        params={"command_name": "ee_target", "scale": 0.12, "deadzone": 0.01, "max_penalty": 4.0},
    )
    flange_angular_velocity_residual = RewTerm(
        func=mdp.flange_angular_velocity_residual, weight=-0.1,
        params={"command_name": "ee_target", "scale": 0.5, "deadzone": 0.05, "max_penalty": 4.0},
    )
    arm_joint_velocity = RewTerm(
        func=mdp.arm_joint_velocity_l2, weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[f"joint{i}" for i in range(1, 7)])},
    )
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-3.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    # AMP provides gait features but has no static base attitude observation.
    # A small inclination deadzone permits normal gait motion while penalizing leaning.
    base_tilt = RewTerm(
        func=mdp.base_tilt_penalty, weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "deadzone": 0.1, "scale": 0.2},
    )
    
    arm_deviation = RewTerm(
        func=mdp.joint_deviation_l1, 
        # weight=-2.5,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[f"joint{i}" for i in range(1, 7)])},
    )
    
    
    
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.25e-5)
    joint_power = RewTerm(
        func=mdp.power_penalty, weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    action_smoothness = RewTerm(func=mdp.smoothness, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    undesired_contacts = RewTerm(
        func=mdp.uncontacts, weight=-1.0,
        # The massless gun collision now belongs to its physical parent link06.
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names="link06")},
    )
    # A zero-payload tool still transmits ground reaction forces through link06.
    # Permit light working contact, and penalize upward load without saturation.
    tool_support_force = RewTerm(
        func=mdp.tool_support_force, weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="link06"),
            "deadzone": 5.0, "force_scale": 100.0, "duration_s": 0.1,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["base_link", ".*_hip", ".*_thigh", ".*_calf", "link0[0-5]"]
            ),
            "threshold": 1.0,
        },
    )
    tool_support = DoneTerm(
        func=mdp.tool_support_termination,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="link06"),
            "sustained_force": 20.0, "peak_force": 150.0, "duration_s": 0.1,
        },
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.5})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    
    # Fixed-speed commands; increase curve bending instead of velocity ranges.
    lin_vel_cmd_levels = None
    ang_vel_cmd_levels = None

    curve_curvature_cmd_levels = CurrTerm(
        func=mdp.curve_curvature_cmd_levels,
        params={"reward_threshold": 0.8, "offset_scale_step": 0.01},
    )
    pos_cmd_levels = CurrTerm(
        func=mdp.pos_cmd_levels,
        params={"position_step": 0.01, "orientation_step": 0.0, "reward_threshold": 0.8},
    )
    

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

    reference_state_initialization: bool = True
    reference_state_initialization_prob: float = 1.0
    amp_motion_files: list[str] = MOTION_FILES

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 30.0
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
        self.scene.base_camera = make_curve_camera_cfg(self.commands.base_velocity, render=False)
        self.scene.rear_camera = make_rear_camera_cfg(self.commands.ee_target, render=False)
        
        self.commands.base_velocity.curriculum_enabled = True
        self.commands.ee_target.curriculum_enabled = True
        # self.commands.ee_target.curriculum_enabled = False
        
        
        
                 
@configclass
class B2Z1PlayEnvCfg(B2Z1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.reference_state_initialization = False
        
        
        
        self.commands.base_velocity.curriculum_enabled = False
        maximum_turns = self.commands.base_velocity.turn_count_range[1]
        self.commands.base_velocity.turn_count_range = (maximum_turns, maximum_turns)
        maximum_angle = self.commands.base_velocity.turn_angle_range[1]
        self.commands.base_velocity.turn_angle_range = (maximum_angle, maximum_angle)
        # Hard cap; nominal turn_curvature is 1.0 1/m (radius 1 m).
        self.commands.base_velocity.max_curvature = 1.5
        
        self.commands.base_velocity.debug_vis = True
        
        
        self.scene.base_camera = make_curve_camera_cfg(self.commands.base_velocity, render=True)
        self.scene.rear_camera = make_rear_camera_cfg(self.commands.ee_target, render=True)
        
        self.commands.ee_target.curriculum_enabled = False
        self.commands.ee_target.debug_vis = True

        # Preserve the training contact, tool-support, and inclination guards.
