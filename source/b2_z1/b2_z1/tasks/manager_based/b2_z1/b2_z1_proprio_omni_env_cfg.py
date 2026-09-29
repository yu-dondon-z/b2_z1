"""Vision-independent omnidirectional continuation task for B2+Z1."""

from __future__ import annotations

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .b2_z1_rear_vision_env_cfg import (
    B2Z1RealGaitClearanceV3FineTuneAMPEnvCfg,
    B2Z1RearVisionV3GaitFineTuneAMPEnvCfg,
    RealGaitClearanceV3RewardsCfg,
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


@configclass
class ProprioOmniV5FoundationRewardsCfg(RealGaitClearanceV3RewardsCfg):
    """AMP locomotion rewards for staged omnidirectional adaptation.

    These penalties are deliberately no stronger than the last stable path
    task.  The failed straight-recovery experiment showed that increasing
    them abruptly trades heading error for contact and orientation failures.
    """

    straight_lateral_velocity = RewTerm(
        func=mdp.straight_lateral_velocity_l2,
        weight=-1.50,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    straight_yaw_rate = RewTerm(
        func=mdp.straight_yaw_rate_l2,
        weight=-1.00,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    straight_heading_drift = RewTerm(
        func=mdp.straight_heading_drift_l2,
        weight=-1.00,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class B2Z1ProprioOmniV5FoundationAMPEnvCfg(
    B2Z1RealGaitClearanceV3FineTuneAMPEnvCfg
):
    """Stage-one body-motion adaptation from clearance V3 ``model_10700``.

    The policy still receives the same 70-D observation and all AMP motion
    files.  Most moving environments train forward/backward locomotion, while
    smaller lateral, yaw and mixed subsets introduce the eventual full SE(2)
    interface without immediately destroying the selected trot.
    """

    rewards: ProprioOmniV5FoundationRewardsCfg = ProprioOmniV5FoundationRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity = mdp.OmniVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 14.0),
            rel_standing_envs=0.15,
            debug_vis=False,
            heading_command=False,
            curriculum_enabled=False,
            longitudinal_fraction=0.60,
            lateral_fraction=0.15,
            yaw_fraction=0.10,
            mixed_fraction=0.15,
            minimum_linear_speed=0.10,
            minimum_yaw_rate=0.08,
            # Keep pure translation commands exactly [vx, vy, 0].  Heading
            # retention must therefore come from the policy, not a hidden
            # outer-loop yaw correction unavailable to direct teleoperation.
            heading_hold_stiffness=0.0,
            ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.35, 0.35),
                lin_vel_y=(-0.15, 0.15),
                ang_vel_z=(-0.18, 0.18),
                heading=None,
            ),
            limit_ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.50, 0.50),
                lin_vel_y=(-0.30, 0.30),
                ang_vel_z=(-0.35, 0.35),
                heading=None,
            ),
        )

        # Hold the rear welding pose in the body frame.  Vision later maps a
        # path to this same velocity command interface, but is intentionally
        # absent while the locomotion foundation is adapted.
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

        # Remove path-only objectives.  AMP remains active and uses all
        # forward/backward/lateral/turn demonstrations in datasets/b2_motion.
        self.rewards.crack_centerline_tracking = None
        self.rewards.crack_heading_tracking = None
        self.rewards.stance_feet_crack_clearance = None
        self.curriculum.vel_cmd_to_pos_levels = None

        self.rewards.track_lin_vel_xy.weight = 3.50
        self.rewards.track_lin_vel_xy.params["std"] = math.sqrt(0.20)
        self.rewards.track_ang_vel_z.weight = 2.00
        self.rewards.track_ang_vel_z.params["std"] = math.sqrt(0.20)

        # Stage one isolates command adaptation.  Extra actuator delay is
        # introduced only after this nominal distribution passes validation.
        for actuator_name in ("M107-24-2", "2"):
            actuator_cfg = self.scene.robot.actuators[actuator_name]
            actuator_cfg.min_delay = 0
            actuator_cfg.max_delay = 0


@configclass
class B2Z1ProprioOmniV5FoundationAMPEnvCfg_Visual(
    B2Z1ProprioOmniV5FoundationAMPEnvCfg
):
    """Small-batch viewer for the stage-one body-motion policy."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.debug_vis = True
        self.commands.ee_target.debug_vis = True


@configclass
class B2Z1ProprioOmniV5FoundationAMPEnvCfg_Play(
    B2Z1ProprioOmniV5FoundationAMPEnvCfg
):
    """Long deterministic playback for body-command checkpoint selection."""

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


@configclass
class B2Z1ProprioForwardV6FoundationAMPEnvCfg(
    B2Z1ProprioOmniV5FoundationAMPEnvCfg
):
    """First curriculum stage: direct forward commands with zero commanded yaw."""

    def __post_init__(self):
        super().__post_init__()

        # Do not ask one continuation to learn every new direction at once.
        # This stage matches the immediate hardware requirement exactly:
        # positive vx, zero vy and zero yaw, plus stationary handoff samples.
        command = self.commands.base_velocity
        command.longitudinal_fraction = 1.0
        command.lateral_fraction = 0.0
        command.yaw_fraction = 0.0
        command.mixed_fraction = 0.0
        command.rel_standing_envs = 0.15
        command.minimum_linear_speed = 0.10
        command.minimum_yaw_rate = 0.0
        command.heading_hold_stiffness = 0.0
        command.ranges.lin_vel_x = (0.10, 0.35)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.limit_ranges.lin_vel_x = (0.0, 0.50)
        command.limit_ranges.lin_vel_y = (0.0, 0.0)
        command.limit_ranges.ang_vel_z = (0.0, 0.0)


@configclass
class B2Z1ProprioForwardV6FoundationAMPEnvCfg_Visual(
    B2Z1ProprioForwardV6FoundationAMPEnvCfg
):
    """Small-batch viewer for direct forward-command adaptation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.debug_vis = True
        self.commands.ee_target.debug_vis = True


@configclass
class B2Z1ProprioForwardV6FoundationAMPEnvCfg_Play(
    B2Z1ProprioForwardV6FoundationAMPEnvCfg
):
    """Long deterministic playback for the selected forward checkpoint."""

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
            "x": (0.0, 0.0), "y": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class B2Z1ProprioForwardV7HeadingHoldAMPEnvCfg(
    B2Z1ProprioForwardV6FoundationAMPEnvCfg
):
    """Forward locomotion with deployment-matched IMU heading hold."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        # The operator command remains vx-only.  Internally, world-yaw error
        # from the IMU is converted to a small policy yaw-rate command.  The
        # clamp avoids a sharp steering transient after policy takeover.
        command.heading_hold_stiffness = 1.0
        command.ranges.ang_vel_z = (-0.15, 0.15)
        command.limit_ranges.ang_vel_z = (-0.18, 0.18)


@configclass
class B2Z1ProprioForwardV7HeadingHoldAMPEnvCfg_Visual(
    B2Z1ProprioForwardV7HeadingHoldAMPEnvCfg
):
    """Small-batch viewer for IMU heading-hold adaptation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.debug_vis = True
        self.commands.ee_target.debug_vis = True


@configclass
class B2Z1ProprioForwardV7HeadingHoldAMPEnvCfg_Play(
    B2Z1ProprioForwardV7HeadingHoldAMPEnvCfg
):
    """Long deterministic playback for the heading-hold checkpoint."""

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
            "x": (0.0, 0.0), "y": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class ProprioForwardBackwardV8RewardsCfg(ProprioOmniV5FoundationRewardsCfg):
    """Direction-aware rewards with separately observable front/rear lift."""

    forward_velocity_error = RewTerm(
        func=mdp.command_mode_lin_vel_xy_l2,
        weight=-0.50,
        params={
            "command_name": "base_velocity",
            "mode_attribute": "is_forward_env",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    backward_velocity_error = RewTerm(
        func=mdp.command_mode_lin_vel_xy_l2,
        weight=-0.75,
        params={
            "command_name": "base_velocity",
            "mode_attribute": "is_backward_env",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    front_feet_height = RewTerm(
        func=mdp.swing_leg_height_reward,
        weight=0.275,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="F[LR]_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names="F[LR]_foot"),
            "target_height": 0.11,
            "velocity_threshold": 0.08,
            "std": 0.045,
            "include_body_velocity": False,
        },
    )
    rear_feet_height = RewTerm(
        func=mdp.swing_leg_height_reward,
        weight=0.275,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R[LR]_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names="R[LR]_foot"),
            "target_height": 0.11,
            "velocity_threshold": 0.08,
            "std": 0.045,
            "include_body_velocity": False,
        },
    )


@configclass
class B2Z1ProprioForwardBackwardV8ClearanceAMPEnvCfg(
    B2Z1ProprioForwardV7HeadingHoldAMPEnvCfg
):
    """Add low-speed reverse samples while gently improving swing clearance."""

    rewards: ProprioForwardBackwardV8RewardsCfg = ProprioForwardBackwardV8RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity

        # Roughly 70% forward and 30% reverse after the near-zero exclusion.
        # Reverse is initially capped at 0.15 m/s so it cannot dominate the
        # already selected forward gait in this first bidirectional stage.
        command.ranges.lin_vel_x = (-0.15, 0.35)
        command.limit_ranges.lin_vel_x = (-0.25, 0.50)

        # Replace the aggregate 10 cm height reward with separate front/rear
        # 11 cm terms.  Their weights sum to only 0.55, a small increase over
        # V7's 0.45, and the sole-clearance floor rises by just 5 mm.
        self.rewards.feet_height = None
        self.rewards.swing_foot_clearance_deficit.weight = -0.10
        self.rewards.swing_foot_clearance_deficit.params["minimum_clearance"] = 0.045


@configclass
class B2Z1ProprioForwardBackwardV8ClearanceAMPEnvCfg_Visual(
    B2Z1ProprioForwardBackwardV8ClearanceAMPEnvCfg
):
    """Small-batch viewer for bidirectional clearance adaptation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.debug_vis = True
        self.commands.ee_target.debug_vis = True


@configclass
class B2Z1ProprioForwardBackwardV8ClearanceAMPEnvCfg_Play(
    B2Z1ProprioForwardBackwardV8ClearanceAMPEnvCfg
):
    """Long deterministic-condition playback for V8 checkpoint selection."""

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
            "x": (0.0, 0.0), "y": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class B2Z1ProprioForwardBackwardV9GentleAMPEnvCfg(
    B2Z1ProprioForwardBackwardV8ClearanceAMPEnvCfg
):
    """Gentle reverse curriculum after rejecting V8's 30% reverse mix."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        command.longitudinal_forward_fraction = 0.85
        command.ranges.lin_vel_x = (-0.12, 0.35)
        command.limit_ranges.lin_vel_x = (-0.20, 0.50)

        self.rewards.backward_velocity_error.weight = -0.50
        self.rewards.front_feet_height.weight = 0.25
        self.rewards.rear_feet_height.weight = 0.25
        self.rewards.swing_foot_clearance_deficit.params["minimum_clearance"] = 0.0425


@configclass
class B2Z1ProprioForwardBackwardV9GentleAMPEnvCfg_Visual(
    B2Z1ProprioForwardBackwardV9GentleAMPEnvCfg
):
    """Small-batch viewer for the gentle reverse curriculum."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.debug_vis = True
        self.commands.ee_target.debug_vis = True


@configclass
class B2Z1ProprioForwardBackwardV9GentleAMPEnvCfg_Play(
    B2Z1ProprioForwardBackwardV9GentleAMPEnvCfg
):
    """Long deterministic-condition playback for V9 checkpoint selection."""

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
            "x": (0.0, 0.0), "y": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class B2Z1ProprioForwardBackwardV10BalancedAMPEnvCfg(
    B2Z1ProprioForwardBackwardV9GentleAMPEnvCfg
):
    """Learn observable reverse motion and a measured 3.5 cm swing clearance."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity

        # V9's 15% reverse subset was ignored by the inherited forward policy.
        # Twenty percent preserves the mature forward gait, while the
        # direction-specific loss is scaled to remain visible in a full-batch
        # mean.  Reverse speed stays deliberately below 0.20 m/s in this stage.
        command.longitudinal_forward_fraction = 0.80
        command.ranges.lin_vel_x = (-0.20, 0.35)
        command.limit_ranges.lin_vel_x = (-0.25, 0.50)
        self.rewards.backward_velocity_error.weight = -3.0

        # Express the target as sole-to-ground clearance instead of foot-link
        # world height.  Front and rear terms are kept separate for monitoring.
        clearance_params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="F[LR]_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names="F[LR]_foot"),
            "target_clearance": 0.035,
            "foot_radius": 0.032,
            "minimum_air_time": 0.04,
            "velocity_threshold": 0.08,
            "std": 0.020,
            "command_name": "base_velocity",
        }
        self.rewards.front_feet_height.func = mdp.swing_foot_clearance_target_exp
        self.rewards.front_feet_height.params = clearance_params
        self.rewards.front_feet_height.weight = 0.60

        rear_clearance_params = clearance_params.copy()
        rear_clearance_params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names="R[LR]_foot"
        )
        rear_clearance_params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names="R[LR]_foot"
        )
        self.rewards.rear_feet_height.func = mdp.swing_foot_clearance_target_exp
        self.rewards.rear_feet_height.params = rear_clearance_params
        self.rewards.rear_feet_height.weight = 0.60
        self.rewards.swing_foot_clearance_deficit.weight = -0.30
        self.rewards.swing_foot_clearance_deficit.params["minimum_clearance"] = 0.025


@configclass
class B2Z1ProprioForwardBackwardV10BalancedAMPEnvCfg_Visual(
    B2Z1ProprioForwardBackwardV10BalancedAMPEnvCfg
):
    """Small-batch viewer for V10 forward/reverse clearance training."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.debug_vis = True
        self.commands.ee_target.debug_vis = True


@configclass
class B2Z1ProprioForwardBackwardV10BalancedAMPEnvCfg_Play(
    B2Z1ProprioForwardBackwardV10BalancedAMPEnvCfg
):
    """Long deterministic-condition playback for the selected V10 policy."""

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
            "x": (0.0, 0.0), "y": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class B2Z1ProprioBackwardV11BootstrapAMPEnvCfg(
    B2Z1ProprioForwardBackwardV10BalancedAMPEnvCfg
):
    """Backward-only bootstrap before returning to bidirectional commands."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        command.longitudinal_forward_fraction = 0.0
        command.rel_standing_envs = 0.10
        command.ranges.lin_vel_x = (-0.25, -0.12)
        command.limit_ranges.lin_vel_x = (-0.30, 0.0)

        # With the previous sqrt(0.20) kernel, standing under a -0.15 m/s
        # command still earned nearly all velocity reward.  A 0.10 m/s kernel
        # makes that shortcut unprofitable during this short bootstrap.
        self.rewards.track_lin_vel_xy.weight = 4.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.10
        self.rewards.forward_velocity_error.weight = 0.0
        self.rewards.backward_velocity_error.weight = -5.0


@configclass
class B2Z1ProprioBackwardV11BootstrapAMPEnvCfg_Visual(
    B2Z1ProprioBackwardV11BootstrapAMPEnvCfg
):
    """Small-batch viewer for backward gait acquisition."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.debug_vis = True
        self.commands.ee_target.debug_vis = True


@configclass
class B2Z1ProprioBackwardV11BootstrapAMPEnvCfg_Play(
    B2Z1ProprioBackwardV11BootstrapAMPEnvCfg
):
    """Long deterministic playback for a V11 backward checkpoint."""

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
            "x": (0.0, 0.0), "y": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class B2Z1ProprioBackwardV12AcquisitionAMPEnvCfg(
    B2Z1ProprioBackwardV11BootstrapAMPEnvCfg
):
    """Break the stationary local optimum and acquire a clear reverse gait."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        command.rel_standing_envs = 0.05
        command.ranges.lin_vel_x = (-0.35, -0.20)
        command.limit_ranges.lin_vel_x = (-0.40, 0.0)

        # During acquisition the static TCP rewards otherwise dominate the
        # velocity objective by more than an order of magnitude.  Preserve a
        # small arm-pose incentive, but make actual reverse motion decisive.
        self.rewards.track_lin_vel_xy.weight = 8.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.10
        self.rewards.backward_velocity_error.weight = -10.0
        self.rewards.end_effector_position_tracking_exp.weight = 0.50
        self.rewards.end_effector_position_tracking_fine_grained.weight = 0.25
        self.rewards.front_feet_height.weight = 0.80
        self.rewards.rear_feet_height.weight = 0.80


@configclass
class B2Z1ProprioBackwardV12AcquisitionAMPEnvCfg_Play(
    B2Z1ProprioBackwardV12AcquisitionAMPEnvCfg
):
    """Deterministic-condition playback for V12 checkpoint screening."""

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
            "x": (0.0, 0.0), "y": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class B2Z1ProprioBackwardV13ExpertStartAMPEnvCfg(
    B2Z1ProprioBackwardV12AcquisitionAMPEnvCfg
):
    """Acquire reverse locomotion from velocity-matched expert reset states."""

    def __post_init__(self):
        super().__post_init__()
        # The recorded slow backward prior is centred near -0.406 m/s.  Reset
        # exclusively from it so the actor learns to maintain an existing
        # reverse gait instead of having to discover gait initiation from a
        # forward-only local optimum.  The discriminator runner still loads
        # all motion files; only environment reset initialization is filtered.
        backward_slow = [
            path
            for path in self.amp_motion_files
            if path.endswith("/b2_backward.txt")
        ]
        if len(backward_slow) != 1:
            raise RuntimeError(
                f"Expected exactly one b2_backward.txt reset prior, got {backward_slow}"
            )
        self.amp_motion_files = backward_slow
        self.reference_state_initialization = True
        self.reference_state_initialization_prob = 1.0

        command = self.commands.base_velocity
        command.rel_standing_envs = 0.0
        command.ranges.lin_vel_x = (-0.45, -0.35)
        command.limit_ranges.lin_vel_x = (-0.50, -0.30)


@configclass
class B2Z1ProprioBackwardV13ExpertStartAMPEnvCfg_Play(
    B2Z1ProprioBackwardV13ExpertStartAMPEnvCfg
):
    """Default-state playback for screening V13 expert-start checkpoints."""

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
            "x": (0.0, 0.0), "y": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class ProprioForwardV14ClearanceRewardsCfg(ProprioOmniV5FoundationRewardsCfg):
    """Physical front/rear sole-clearance rewards for the selected V7 gait."""

    front_swing_clearance = RewTerm(
        func=mdp.swing_foot_clearance_target_exp,
        weight=0.65,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="F[LR]_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names="F[LR]_foot"),
            "target_clearance": 0.025,
            "foot_radius": 0.032,
            "minimum_air_time": 0.04,
            "velocity_threshold": 0.08,
            "std": 0.0125,
            "command_name": "base_velocity",
        },
    )
    rear_swing_clearance = RewTerm(
        func=mdp.swing_foot_clearance_target_exp,
        weight=0.65,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R[LR]_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names="R[LR]_foot"),
            "target_clearance": 0.025,
            "foot_radius": 0.032,
            "minimum_air_time": 0.04,
            "velocity_threshold": 0.08,
            "std": 0.0125,
            "command_name": "base_velocity",
        },
    )


@configclass
class B2Z1ProprioForwardV14ClearanceAMPEnvCfg(
    B2Z1ProprioForwardV7HeadingHoldAMPEnvCfg
):
    """Raise V7 swing clearance without changing its command distribution."""

    rewards: ProprioForwardV14ClearanceRewardsCfg = (
        ProprioForwardV14ClearanceRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()

        # The legacy term targets a foot-link world-z value, which is not a
        # measurable sole-to-ground clearance.  Replace it while preserving
        # V7's forward-only commands, heading hold and stability objectives.
        self.rewards.feet_height = None
        self.rewards.swing_foot_clearance_deficit.weight = -0.20
        self.rewards.swing_foot_clearance_deficit.params["minimum_clearance"] = 0.015


@configclass
class B2Z1ProprioForwardV14ClearanceAMPEnvCfg_Play(
    B2Z1ProprioForwardV14ClearanceAMPEnvCfg
):
    """Deterministic playback for selecting a V14 clearance checkpoint."""

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
            "x": (0.0, 0.0), "y": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class B2Z1ProprioForwardV15ClearanceAMPEnvCfg(
    B2Z1ProprioForwardV14ClearanceAMPEnvCfg
):
    """Stronger second-stage lift shaping with velocity-tracking protection."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.front_swing_clearance.weight = 1.25
        self.rewards.rear_swing_clearance.weight = 1.25
        self.rewards.swing_foot_clearance_deficit.weight = -0.50
        self.rewards.swing_foot_clearance_deficit.params["minimum_clearance"] = 0.020

        # Lift must not be gained by slowing down or hopping in place.
        self.rewards.track_lin_vel_xy.weight = 4.50
        self.rewards.track_ang_vel_z.weight = 2.25


@configclass
class ProprioForwardV16RearClearanceRewardsCfg(
    ProprioForwardV14ClearanceRewardsCfg
):
    """Add an independently visible rear-foot clearance floor."""

    rear_swing_clearance_deficit = RewTerm(
        func=mdp.swing_foot_clearance_deficit,
        weight=-0.75,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R[LR]_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names="R[LR]_foot"),
            "minimum_clearance": 0.020,
            "foot_radius": 0.032,
            "minimum_air_time": 0.04,
            "velocity_threshold": 0.08,
            "command_name": "base_velocity",
        },
    )


@configclass
class B2Z1ProprioForwardV16RearClearanceAMPEnvCfg(
    B2Z1ProprioForwardV15ClearanceAMPEnvCfg
):
    """Raise the lagging rear feet while retaining V15 front-foot motion."""

    rewards: ProprioForwardV16RearClearanceRewardsCfg = (
        ProprioForwardV16RearClearanceRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()

        # Front feet already improved in V15.  Bias adaptation toward the rear
        # pair without removing the front target entirely.
        self.rewards.front_swing_clearance.weight = 0.60
        self.rewards.rear_swing_clearance.weight = 2.00
        self.rewards.swing_foot_clearance_deficit.weight = -0.15
        self.rewards.swing_foot_clearance_deficit.params["minimum_clearance"] = 0.015

        self.rewards.track_lin_vel_xy.weight = 5.00
        self.rewards.track_ang_vel_z.weight = 2.50
        self.rewards.trot_diagonal_sync.weight = 0.75


@configclass
class B2Z1ProprioForwardV17ClearanceRecoveryAMPEnvCfg(
    B2Z1ProprioForwardV16RearClearanceAMPEnvCfg
):
    """Recover speed and straightness while retaining V16 clearance gains."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.front_swing_clearance.weight = 0.80
        self.rewards.rear_swing_clearance.weight = 1.25
        self.rewards.swing_foot_clearance_deficit.weight = -0.10
        self.rewards.rear_swing_clearance_deficit.weight = -0.40

        self.rewards.track_lin_vel_xy.weight = 6.00
        self.rewards.track_ang_vel_z.weight = 2.75
        self.rewards.straight_lateral_velocity.weight = -2.00
        self.rewards.straight_yaw_rate.weight = -1.25
        self.rewards.straight_heading_drift.weight = -1.25
        self.rewards.trot_diagonal_sync.weight = 0.70


@configclass
class ProprioForwardV18ContactBalanceRewardsCfg(
    ProprioForwardV14ClearanceRewardsCfg
):
    """V15 rewards plus a bounded guard against a persistently planted foot."""

    prolonged_foot_contact = RewTerm(
        func=mdp.prolonged_foot_contact_penalty,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_foot"
            ),
            "max_contact_time": 0.40,
            "maximum_excess_time": 0.40,
            "velocity_threshold": 0.10,
            "command_name": "base_velocity",
        },
    )


@configclass
class B2Z1ProprioForwardV18ContactBalanceAMPEnvCfg(
    B2Z1ProprioForwardV15ClearanceAMPEnvCfg
):
    """Single-variable V15 continuation for balanced contact transitions."""

    rewards: ProprioForwardV18ContactBalanceRewardsCfg = (
        ProprioForwardV18ContactBalanceRewardsCfg()
    )

    def __post_init__(self):
        # Retain V15's command distribution, clearance shaping, velocity
        # tracking and AMP prior.  Only the reward term declared above is new.
        super().__post_init__()


@configclass
class ProprioForwardV19ContactCycleRewardsCfg(
    ProprioForwardV18ContactBalanceRewardsCfg
):
    """V15 shaping with earlier long-contact and direct trot-state feedback."""

    diagonal_contact_state_mismatch = RewTerm(
        func=mdp.DiagonalContactStateMismatchPenalty,
        weight=-0.35,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "synced_feet_pair_names": (
                ("FL_foot", "RR_foot"),
                ("FR_foot", "RL_foot"),
            ),
            "velocity_threshold": 0.10,
            "command_name": "base_velocity",
        },
    )


@configclass
class B2Z1ProprioForwardV19ContactCycleAMPEnvCfg(
    B2Z1ProprioForwardV15ClearanceAMPEnvCfg
):
    """Contact-cycle continuation from the original V15 ``model_11350``."""

    rewards: ProprioForwardV19ContactCycleRewardsCfg = (
        ProprioForwardV19ContactCycleRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.prolonged_foot_contact.weight = -0.60
        self.rewards.prolonged_foot_contact.params["max_contact_time"] = 0.32
        self.rewards.prolonged_foot_contact.params["maximum_excess_time"] = 0.25


@configclass
class ProprioOmniV20VisionPreserveRewardsCfg(
    ProprioForwardV19ContactCycleRewardsCfg
):
    """Direction-visible losses that overcome the low-speed stationary basin."""

    forward_velocity_error = RewTerm(
        func=mdp.command_mode_lin_vel_xy_l2,
        weight=-0.50,
        params={
            "command_name": "base_velocity",
            "mode_attribute": "is_forward_env",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    backward_velocity_error = RewTerm(
        func=mdp.command_mode_lin_vel_xy_l2,
        weight=-6.0,
        params={
            "command_name": "base_velocity",
            "mode_attribute": "is_backward_env",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    lateral_velocity_error = RewTerm(
        func=mdp.command_mode_lin_vel_xy_l2,
        weight=-6.0,
        params={
            "command_name": "base_velocity",
            "mode_attribute": "is_lateral_env",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    yaw_velocity_error = RewTerm(
        func=mdp.command_mode_ang_vel_z_l2,
        weight=-4.0,
        params={
            "command_name": "base_velocity",
            "mode_attribute": "is_yaw_env",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class B2Z1ProprioOmniV20VisionPreserveAMPEnvCfg(
    B2Z1ProprioForwardV19ContactCycleAMPEnvCfg
):
    """Conservative cardinal-motion continuation from V19 ``model_11449``.

    V19 remains the immutable forward/vision baseline.  This first omni stage
    keeps most moving environments longitudinal (and predominantly forward),
    while introducing only low-speed reverse, lateral, yaw and mixed commands.
    The observation/action contract stays unchanged, so the resulting actor is
    still compatible with the existing visual trajectory controller.
    """

    rewards: ProprioOmniV20VisionPreserveRewardsCfg = (
        ProprioOmniV20VisionPreserveRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity = mdp.OmniVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 14.0),
            rel_standing_envs=0.15,
            debug_vis=False,
            heading_command=False,
            curriculum_enabled=False,
            # Before the standing override this yields 41.25% forward, 13.75%
            # reverse, 20% lateral, 15% yaw and 10% mixed commands.  Forward
            # remains the largest single mode and anchors the visual gait.
            longitudinal_fraction=0.55,
            longitudinal_forward_fraction=0.75,
            lateral_fraction=0.20,
            yaw_fraction=0.15,
            mixed_fraction=0.10,
            minimum_linear_speed=0.08,
            minimum_yaw_rate=0.06,
            # Pure translations retain their sampled world heading.  The
            # correction is still exposed through the ordinary yaw command.
            heading_hold_stiffness=1.0,
            ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.18, 0.35),
                lin_vel_y=(-0.12, 0.12),
                ang_vel_z=(-0.15, 0.15),
                heading=None,
            ),
            # Keep limits identical in this stage.  The inherited curriculum
            # manager copies ``limit_ranges`` into ``ranges`` when command
            # curriculum is disabled, so wider limits here would silently turn
            # this conservative stage into a full-range omni run.
            limit_ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.18, 0.35),
                lin_vel_y=(-0.12, 0.12),
                ang_vel_z=(-0.15, 0.15),
                heading=None,
            ),
        )

        # Preserve the visual/TCP branch instead of presenting the actor with
        # one fixed arm target for the entire omni continuation.  This legacy
        # base-frame path moves the rear TCP target smoothly without coupling
        # the independently sampled body velocity to a synthetic crack.
        self.commands.ee_target.synthetic_world_fixed_path = False
        self.commands.ee_target.synthetic_lateral_amplitude_range_m = (-0.08, 0.08)
        self.commands.ee_target.synthetic_wavelength_range_m = (10.0, 16.0)
        self.commands.ee_target.synthetic_speed_range_mps = (0.15, 0.30)
        self.commands.ee_target.resampling_time_range = (14.0, 20.0)

        # Retain V19's contact-cycle, clearance, velocity and AMP objectives.
        # No vision/path reward is removed or reweighted in this continuation.


@configclass
class B2Z1ProprioOmniV20VisionPreserveAMPEnvCfg_Visual(
    B2Z1ProprioOmniV20VisionPreserveAMPEnvCfg
):
    """Small-batch viewer for checking V20 body and TCP command coverage."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.debug_vis = True
        self.commands.ee_target.debug_vis = True


@configclass
class B2Z1ProprioOmniV20VisionPreserveAMPEnvCfg_Play(
    B2Z1ProprioOmniV20VisionPreserveAMPEnvCfg
):
    """Deterministic long-horizon environment for V20 checkpoint gates."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 3600.0
        self.reference_state_initialization_prob = 0.0
        self.commands.base_velocity.debug_vis = True
        self.commands.ee_target.debug_vis = True

        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_dynamics = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class B2Z1ProprioOmniV21MatchedAcquisitionAMPEnvCfg(
    B2Z1ProprioOmniV20VisionPreserveAMPEnvCfg
):
    """Acquire cardinal skills from direction-matched AMP reset states."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        # Keep one primitive for the complete 60 s episode.  Reference poses
        # are matched at reset; changing direction mid-episode would break
        # that correspondence during the acquisition stage.
        command.resampling_time_range = (60.0, 60.0)
        command.rel_standing_envs = 0.0
        command.longitudinal_fraction = 0.40
        command.longitudinal_forward_fraction = 0.50
        command.lateral_fraction = 0.35
        command.yaw_fraction = 0.25
        command.mixed_fraction = 0.0
        command.minimum_linear_speed = 0.35
        command.minimum_yaw_rate = 0.18
        command.ranges.lin_vel_x = (-0.45, 0.45)
        command.ranges.lin_vel_y = (-0.45, 0.45)
        command.ranges.ang_vel_z = (-0.25, 0.25)
        command.limit_ranges = command.ranges

        self.reference_state_initialization = True
        self.reference_state_initialization_prob = 1.0
        self.command_conditioned_reference_initialization = True

        # At acquisition speed, standing still must be decisively worse than
        # maintaining the matched expert gait.  TCP objectives remain active.
        self.rewards.track_lin_vel_xy.weight = 8.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.12
        self.rewards.track_ang_vel_z.weight = 5.0
        self.rewards.track_ang_vel_z.params["std"] = 0.10
        self.rewards.forward_velocity_error.weight = -4.0
        self.rewards.backward_velocity_error.weight = -8.0
        self.rewards.lateral_velocity_error.weight = -8.0
        self.rewards.yaw_velocity_error.weight = -6.0


@configclass
class B2Z1ProprioOmniV21MatchedHandoffAMPEnvCfg(
    B2Z1ProprioOmniV21MatchedAcquisitionAMPEnvCfg
):
    """Introduce ordinary standing resets after cardinal gait acquisition."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        command.resampling_time_range = (18.0, 24.0)
        command.rel_standing_envs = 0.10
        command.longitudinal_fraction = 0.45
        command.lateral_fraction = 0.30
        command.yaw_fraction = 0.20
        command.mixed_fraction = 0.05
        command.minimum_linear_speed = 0.15
        command.minimum_yaw_rate = 0.10
        command.ranges.lin_vel_x = (-0.35, 0.35)
        command.ranges.lin_vel_y = (-0.30, 0.30)
        command.ranges.ang_vel_z = (-0.20, 0.20)
        command.limit_ranges = command.ranges
        self.reference_state_initialization_prob = 0.50


@configclass
class B2Z1ProprioOmniV21VisionPreserveAMPEnvCfg(
    B2Z1ProprioOmniV21MatchedHandoffAMPEnvCfg
):
    """Final low-speed omni distribution with mostly default-state starts."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        command.resampling_time_range = (10.0, 14.0)
        command.rel_standing_envs = 0.15
        command.longitudinal_fraction = 0.45
        command.lateral_fraction = 0.25
        command.yaw_fraction = 0.15
        command.mixed_fraction = 0.15
        command.minimum_linear_speed = 0.08
        command.minimum_yaw_rate = 0.06
        command.ranges.lin_vel_x = (-0.25, 0.40)
        command.ranges.lin_vel_y = (-0.20, 0.20)
        command.ranges.ang_vel_z = (-0.20, 0.20)
        command.limit_ranges = command.ranges
        self.reference_state_initialization_prob = 0.10


@configclass
class B2Z1ProprioOmniV21VisionPreserveAMPEnvCfg_Play(
    B2Z1ProprioOmniV21VisionPreserveAMPEnvCfg
):
    """Deterministic default-state gate for the selected V21 checkpoint."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 3600.0
        self.reference_state_initialization_prob = 0.0
        self.commands.base_velocity.debug_vis = True
        self.commands.ee_target.debug_vis = True
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_dynamics = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None
