"""Vision-independent omnidirectional continuation task for B2+Z1."""

from __future__ import annotations

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .b2_z1_rear_vision_env_cfg import (
    B2Z1RearVisionV3GaitFineTuneAMPEnvCfg,
    RearVisionV3GaitFineTuneRewardsCfg,
)


@configclass
class ProprioOmniV4RewardsCfg(RearVisionV3GaitFineTuneRewardsCfg):
    """Locomotion rewards for straight, lateral and turning commands."""

    straight_lateral_velocity = RewTerm(
        func=mdp.straight_lateral_velocity_l2,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    straight_yaw_rate = RewTerm(
        func=mdp.straight_yaw_rate_l2,
        weight=-1.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    straight_heading_drift = RewTerm(
        func=mdp.straight_heading_drift_l2,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class B2Z1ProprioOmniV4FineTuneAMPEnvCfg(B2Z1RearVisionV3GaitFineTuneAMPEnvCfg):
    """Continue ``model_10000`` without vision and learn full SE(2) motion.

    The actor observation remains 70-D.  The rear TCP target is fixed in the
    base frame, while independent velocity primitives cover forward/backward,
    left/right, yaw-only and diagonal motion.  This makes the checkpoint and
    downstream deployment interface compatible with the existing V3 policy.
    """

    rewards: ProprioOmniV4RewardsCfg = ProprioOmniV4RewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity = mdp.OmniVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(8.0, 12.0),
            rel_standing_envs=0.15,
            debug_vis=False,
            heading_command=False,
            curriculum_enabled=True,
            longitudinal_fraction=0.35,
            lateral_fraction=0.25,
            yaw_fraction=0.15,
            mixed_fraction=0.25,
            minimum_linear_speed=0.10,
            minimum_yaw_rate=0.10,
            heading_hold_stiffness=1.25,
            ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.40, 0.40),
                lin_vel_y=(-0.20, 0.20),
                ang_vel_z=(-0.25, 0.25),
                heading=None,
            ),
            limit_ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.60, 0.60),
                lin_vel_y=(-0.45, 0.45),
                ang_vel_z=(-0.50, 0.50),
                heading=None,
            ),
        )

        # Keep the arm at the familiar rear-facing welding pose.  This task
        # deliberately contains no camera/path dependency, so lateral motion
        # is learned without a trajectory controller fighting the command.
        self.commands.ee_target.synthetic_world_fixed_path = False
        self.commands.ee_target.synthetic_lateral_amplitude_range_m = (0.0, 0.0)
        self.commands.ee_target.synthetic_wavelength_range_m = (20.0, 20.0)
        self.commands.ee_target.synthetic_speed_range_mps = (0.30, 0.30)
        self.commands.ee_target.rear_work_x_range_m = (-0.775, -0.775)
        self.commands.ee_target.synthetic_ground_z_b_m = -0.44
        self.commands.ee_target.tool_height_m = 0.09
        self.commands.ee_target.tool_height_range_m = (0.09, 0.09)
        self.commands.ee_target.resampling_time_range = (1.0e6, 1.0e6)
        self.commands.ee_target.curriculum_enabled = False

        # Remove vision/path-specific privileged objectives. AMP still sees
        # every B2 motion file, including left/right and turning examples.
        self.rewards.crack_centerline_tracking = None
        self.rewards.crack_heading_tracking = None
        self.rewards.stance_feet_crack_clearance = None
        self.curriculum.vel_cmd_to_pos_levels = None

        self.rewards.track_lin_vel_xy.weight = 4.0
        self.rewards.track_lin_vel_xy.params["std"] = math.sqrt(0.20)
        self.rewards.track_ang_vel_z.weight = 2.5
        self.rewards.track_ang_vel_z.params["std"] = math.sqrt(0.20)


@configclass
class B2Z1ProprioOmniV4FineTuneAMPEnvCfg_Visual(B2Z1ProprioOmniV4FineTuneAMPEnvCfg):
    """Small-batch viewer for inspecting command arrows during training."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.debug_vis = True
        self.commands.ee_target.debug_vis = True


@configclass
class B2Z1ProprioOmniV4FineTuneAMPEnvCfg_Play(B2Z1ProprioOmniV4FineTuneAMPEnvCfg):
    """Long, deterministic-condition playback for V4 checkpoint selection."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 3600.0
        self.reference_state_initialization_prob = 0.0
        self.commands.base_velocity.debug_vis = True

        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_dynamics = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None
