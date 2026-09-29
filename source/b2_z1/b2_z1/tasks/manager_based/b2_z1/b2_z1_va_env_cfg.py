"""VA-series truth-curve curriculum tasks for B2+Z1 AMP training."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .b2_z1_amp_env_cfg import CurriculumCfg, TerminationsCfg
from .b2_z1_proprio_omni_env_cfg import ProprioForwardV19ContactCycleRewardsCfg
from .b2_z1_rear_vision_env_cfg import B2Z1RealGaitClearanceV3FineTuneAMPEnvCfg


@configclass
class VA1TruthCurveCurriculumCfg(CurriculumCfg):
    """Advance curve difficulty only after locomotion and TCP gates pass."""

    # These legacy curricula expand generic velocity/pose boxes.  VA1 instead
    # changes the actual world-frame crack geometry, matching the deployment
    # command path and the intent of the ee_drived curriculum.
    lin_vel_cmd_levels = None
    ang_vel_cmd_levels = None
    vel_cmd_to_pos_levels = None

    truth_curve_level = CurrTerm(
        func=mdp.truth_curve_levels,
        params={
            "command_name": "ee_target",
            "reward_term_names": (
                "track_lin_vel_xy",
                "track_ang_vel_z",
                "end_effector_position_tracking_exp",
            ),
            "reward_thresholds": (0.60, 0.35, 0.50),
            "straight_path_fractions": (0.80, 0.65, 0.50, 0.40),
            "arc_radius_ranges_m": (
                (16.0, 24.0),
                (12.0, 20.0),
                (9.0, 16.0),
                (6.0, 12.0),
            ),
            "speed_ranges_mps": (
                (0.15, 0.26),
                (0.16, 0.30),
                (0.18, 0.34),
                (0.20, 0.38),
            ),
            "max_yaw_rates_rps": (0.10, 0.13, 0.16, 0.18),
            "cross_track_gains": (0.35, 0.40, 0.45, 0.50),
            "path_heading_gains": (0.75, 0.80, 0.85, 0.90),
            "minimum_evaluation_episodes": 1024,
            "successful_evaluations_required": 2,
        },
    )


@configclass
class VA1TruthCurveRewardsCfg(ProprioForwardV19ContactCycleRewardsCfg):
    """V19 gait-cycle rewards combined with active path/TCP objectives."""


@configclass
class VA2TruthCurveCurriculumCfg(CurriculumCfg):
    """VA2 curve curriculum gated by measured forward progress."""

    lin_vel_cmd_levels = None
    ang_vel_cmd_levels = None
    vel_cmd_to_pos_levels = None

    truth_curve_level = CurrTerm(
        func=mdp.truth_curve_levels,
        params={
            "command_name": "ee_target",
            "reward_term_names": (
                "track_lin_vel_xy",
                "track_ang_vel_z",
                "end_effector_position_tracking_exp",
                "along_path_speed_ratio",
            ),
            # The last gate is the decisive VA2 change: the actual body speed
            # along the path must average at least 70% of the requested speed.
            "reward_thresholds": (0.65, 0.35, 0.45, 0.70),
            "straight_path_fractions": (0.90, 0.70, 0.55, 0.40),
            "arc_radius_ranges_m": (
                (20.0, 30.0),
                (14.0, 22.0),
                (9.0, 16.0),
                (6.0, 12.0),
            ),
            "speed_ranges_mps": (
                (0.18, 0.26),
                (0.18, 0.30),
                (0.20, 0.34),
                (0.20, 0.38),
            ),
            "max_yaw_rates_rps": (0.08, 0.12, 0.16, 0.18),
            "cross_track_gains": (0.30, 0.40, 0.45, 0.50),
            "path_heading_gains": (0.70, 0.80, 0.85, 0.90),
            "minimum_evaluation_episodes": 1024,
            "successful_evaluations_required": 2,
        },
    )


@configclass
class VA2TruthCurveRewardsCfg(VA1TruthCurveRewardsCfg):
    """VA1 gait shaping plus non-degenerate path-progress objectives."""

    commanded_velocity_error = RewTerm(
        func=mdp.commanded_lin_vel_xy_l2,
        weight=-8.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    along_path_speed_ratio = RewTerm(
        func=mdp.commanded_along_path_speed_ratio,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "velocity_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class B2Z1VA1TruthCurveAMPEnvCfg(B2Z1RealGaitClearanceV3FineTuneAMPEnvCfg):
    """From-scratch AMP task with a performance-gated truth-curve curriculum.

    The policy interface remains the deployment-compatible 70 observations and
    18 actions.  Truth-curve geometry is privileged command/reward information;
    it is not added to the actor observation.  At deployment the same command
    slots can therefore be populated by the validated visual trajectory.
    """

    rewards: VA1TruthCurveRewardsCfg = VA1TruthCurveRewardsCfg()
    curriculum: VA1TruthCurveCurriculumCfg = VA1TruthCurveCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # Stage 0: mostly straight paths and very gentle left/right turns.
        # The curriculum expands these values only after consecutive batches
        # pass velocity, yaw and TCP tracking gates.
        command = self.commands.ee_target
        command.synthetic_world_fixed_path = True
        command.synthetic_path_family = "straight_arc"
        command.synthetic_straight_path_fraction = 0.80
        command.synthetic_arc_radius_range_m = (16.0, 24.0)
        command.synthetic_arc_direction = "both"
        command.synthetic_speed_range_mps = (0.15, 0.26)
        command.resampling_time_range = (14.0, 20.0)
        command.path_lookahead_m = 1.00
        command.cross_track_gain = 0.35
        command.path_heading_gain = 0.75
        command.max_yaw_rate_rps = 0.10
        command.minimum_forward_tangent = 0.35
        command.command_transition_time_s = 4.0
        command.velocity_filter_time_constant_s = 0.40
        command.tool_height_m = 0.09
        command.tool_height_range_m = (0.09, 0.09)

        # EEDerivedVelocityCommand consumes the exact curve tangent exposed by
        # ee_target.  Standing samples explicitly retain stable takeover and
        # zero-command behavior instead of forcing continuous motion.
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.resampling_time_range = (14.0, 20.0)
        self.commands.base_velocity.curriculum_enabled = False

        # Retain the V19 contact-cycle guards that addressed long single-foot
        # support, while keeping the physical 2.5 cm sole-clearance objective.
        self.rewards.prolonged_foot_contact.weight = -0.60
        self.rewards.prolonged_foot_contact.params["max_contact_time"] = 0.32
        self.rewards.prolonged_foot_contact.params["maximum_excess_time"] = 0.25
        self.rewards.diagonal_contact_state_mismatch.weight = -0.35
        self.rewards.front_swing_clearance.weight = 1.25
        self.rewards.rear_swing_clearance.weight = 1.25
        self.rewards.swing_foot_clearance_deficit.weight = -0.30
        self.rewards.swing_foot_clearance_deficit.params["minimum_clearance"] = 0.020
        self.rewards.feet_height = None

        # Curve/TCP tracking stays visible but does not overwhelm AMP gait
        # style.  Straight-line drift terms are inherited from the V5/V19
        # reward chain and remain active for the 80% stage-zero straight set.
        self.rewards.crack_centerline_tracking.weight = 0.75
        self.rewards.crack_heading_tracking.weight = 0.35
        self.rewards.stance_feet_crack_clearance.weight = -0.20
        self.rewards.track_lin_vel_xy.weight = 4.50
        self.rewards.track_ang_vel_z.weight = 2.25
        self.rewards.trot_diagonal_sync.weight = 0.70
        self.rewards.air_time_variance.weight = -0.45
        self.rewards.action_rate_l2.weight = -0.025
        self.rewards.action_smoothness.weight = -0.025

        # Keep modest actuator-delay randomization for sim-to-real robustness.
        # Each delay step is 5 ms.  The 0--10 ms range covers the B2 command
        # path without making actuator lag large enough to hide gait errors.
        for actuator_name in ("M107-24-2", "2"):
            actuator_cfg = self.scene.robot.actuators[actuator_name]
            actuator_cfg.min_delay = 0
            actuator_cfg.max_delay = 2


@configclass
class B2Z1VA1TruthCurveAMPEnvCfg_Visual(B2Z1VA1TruthCurveAMPEnvCfg):
    """Small visual batch for verifying VA1 geometry before a formal run."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.debug_vis = True
        self.commands.ee_target.debug_vis = True


@configclass
class B2Z1VA1TruthCurveAMPEnvCfg_Play(B2Z1VA1TruthCurveAMPEnvCfg):
    """Long deterministic playback over the final VA1 curve distribution."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 3600.0
        self.reference_state_initialization_prob = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.debug_vis = True
        self.commands.ee_target.debug_vis = True

        # Playback evaluates the final curriculum envelope directly.
        command = self.commands.ee_target
        command.synthetic_straight_path_fraction = 0.40
        command.synthetic_arc_radius_range_m = (6.0, 12.0)
        command.synthetic_speed_range_mps = (0.20, 0.38)
        command.max_yaw_rate_rps = 0.18
        command.cross_track_gain = 0.50
        command.path_heading_gain = 0.90

        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_dynamics = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "roll", "pitch", "yaw")
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class B2Z1VA2TruthCurveAMPEnvCfg(B2Z1VA1TruthCurveAMPEnvCfg):
    """VA2 from-scratch task that cannot advance while standing in place."""

    rewards: VA2TruthCurveRewardsCfg = VA2TruthCurveRewardsCfg()
    curriculum: VA2TruthCurveCurriculumCfg = VA2TruthCurveCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # Acquisition starts with mostly straight, gentle trajectories.  VA2
        # deliberately contains only a small standing share so the policy has
        # to discover locomotion before learning tighter curves.
        command = self.commands.ee_target
        command.synthetic_straight_path_fraction = 0.90
        command.synthetic_arc_radius_range_m = (20.0, 30.0)
        command.synthetic_speed_range_mps = (0.18, 0.26)
        command.max_yaw_rate_rps = 0.08
        command.cross_track_gain = 0.30
        command.path_heading_gain = 0.70
        self.commands.base_velocity.rel_standing_envs = 0.05

        # VA1 used sqrt(0.20)=0.447.  At a 0.28 m/s request a stationary robot
        # still earned 68% of that reward.  std=0.12 reduces the same false
        # score below 1%, while the L2 and progress terms provide gradients.
        self.rewards.track_lin_vel_xy.weight = 5.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.12

        # Keep path/TCP objectives active, but below locomotion.  In VA1 the
        # two TCP terms could dominate while B2 stood still under the arm.
        self.rewards.end_effector_position_tracking_exp.weight = 2.0
        self.rewards.end_effector_position_tracking_fine_grained.weight = 1.0
        self.rewards.crack_centerline_tracking.weight = 0.60
        self.rewards.crack_heading_tracking.weight = 0.30

        # Do not suppress the action bandwidth needed to acquire a trot.
        self.rewards.action_rate_l2.weight = -0.015
        self.rewards.action_smoothness.weight = -0.015


@configclass
class B2Z1VA2TruthCurveAMPEnvCfg_Visual(B2Z1VA2TruthCurveAMPEnvCfg):
    """Small visual batch for checking VA2 curve generation and commands."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.debug_vis = True
        self.commands.ee_target.debug_vis = True


@configclass
class B2Z1VA2TruthCurveAMPEnvCfg_Play(B2Z1VA2TruthCurveAMPEnvCfg):
    """Deterministic VA2 playback on the final curriculum envelope."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 3600.0
        self.reference_state_initialization_prob = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.debug_vis = True
        self.commands.ee_target.debug_vis = True

        command = self.commands.ee_target
        command.synthetic_straight_path_fraction = 0.40
        command.synthetic_arc_radius_range_m = (6.0, 12.0)
        command.synthetic_speed_range_mps = (0.20, 0.38)
        command.max_yaw_rate_rps = 0.18
        command.cross_track_gain = 0.50
        command.path_heading_gain = 0.90

        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_dynamics = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "roll", "pitch", "yaw")
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class VA3TruthCurveRewardsCfg(VA2TruthCurveRewardsCfg):
    """VA2 progress rewards with explicit anti-kneeling safeguards."""

    moving_leg_posture = RewTerm(
        func=mdp.moving_leg_posture_deviation_hinge,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_joint",
                    ".*_thigh_joint",
                    ".*_calf_joint",
                ],
            ),
            "command_name": "base_velocity",
            "velocity_threshold": 0.08,
            # V19 measured 0.065 rad and the collapsed VA2 gait 0.094 rad.
            "allowed_mean_deviation": 0.08,
            "deviation_scale": 0.15,
        },
    )
    moving_calf_height = RewTerm(
        func=mdp.moving_body_height_below_penalty,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_calf"),
            "minimum_height": 0.225,
            "height_scale": 0.05,
            "command_name": "base_velocity",
            "velocity_threshold": 0.08,
        },
    )
    moving_thigh_height = RewTerm(
        func=mdp.moving_body_height_below_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_thigh"),
            "minimum_height": 0.455,
            "height_scale": 0.06,
            "command_name": "base_velocity",
            "velocity_threshold": 0.08,
        },
    )
    leg_link_contacts = RewTerm(
        func=mdp.uncontacts,
        weight=-2.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*_calf", ".*_thigh"]
            ),
        },
    )


@configclass
class B2Z1VA3TruthCurveAMPEnvCfg(B2Z1VA2TruthCurveAMPEnvCfg):
    """From-scratch truth-curve training without VA2's kneeling optimum."""

    rewards: VA3TruthCurveRewardsCfg = VA3TruthCurveRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Start gently enough to discover a real trot, while keeping the
        # measured-progress gate that prevents the old stationary solution.
        command = self.commands.ee_target
        command.synthetic_speed_range_mps = (0.16, 0.24)
        command.synthetic_straight_path_fraction = 0.92
        command.synthetic_arc_radius_range_m = (22.0, 32.0)
        command.max_yaw_rate_rps = 0.07
        self.commands.base_velocity.rel_standing_envs = 0.10

        # VA2's very sharp velocity loss made any forward displacement worth
        # more than gait quality. Keep progress observable without rewarding a
        # low-energy crawl more strongly than the AMP/contact objectives.
        self.rewards.track_lin_vel_xy.weight = 4.5
        self.rewards.track_lin_vel_xy.params["std"] = 0.16
        self.rewards.commanded_velocity_error.weight = -5.0
        self.rewards.along_path_speed_ratio.weight = 1.5

        # Nominal B2 standing height and stronger gait-cycle preservation.
        self.rewards.base_height_l2.params["target_height"] = 0.52
        self.rewards.hip_deviation.weight = -0.65
        self.rewards.trot_diagonal_sync.weight = 0.85
        self.rewards.prolonged_foot_contact.weight = -0.70

        # A calf or thigh touching the floor is an invalid locomotion state,
        # not merely a low-scoring way to make progress.
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=["base_link", ".*_calf", ".*_thigh"],
        )
        self.terminations.base_contact.params["threshold"] = 1.0
        self.terminations.bad_orientation.params["limit_angle"] = 0.80


@configclass
class B2Z1VA3TruthCurveAMPEnvCfg_Visual(B2Z1VA3TruthCurveAMPEnvCfg):
    """Small visual batch for checking VA3 anti-kneeling training."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.debug_vis = True
        self.commands.ee_target.debug_vis = True


@configclass
class B2Z1VA3TruthCurveAMPEnvCfg_Play(B2Z1VA3TruthCurveAMPEnvCfg):
    """Deterministic final-envelope playback with safety terminations active."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 3600.0
        self.reference_state_initialization_prob = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.debug_vis = True
        self.commands.ee_target.debug_vis = True

        command = self.commands.ee_target
        # Follow one deterministic, effectively unbounded arc for the entire
        # playback.  The old 14--20 s resampling replaced the world path while
        # the robot was still following it, which looked like a route switch.
        command.synthetic_straight_path_fraction = 0.0
        command.synthetic_arc_radius_range_m = (10.0, 10.0)
        command.synthetic_arc_direction = "left"
        command.synthetic_speed_range_mps = (0.22, 0.22)
        command.resampling_time_range = (1.0e6, 1.0e6)
        command.max_yaw_rate_rps = 0.18
        command.cross_track_gain = 0.50
        command.path_heading_gain = 0.90
        self.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)

        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_dynamics = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "roll", "pitch", "yaw")
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


@configclass
class VA4SmoothToolRewardsCfg(VA3TruthCurveRewardsCfg):
    """Keep VA3 gait and TCP tracking while discouraging gait-locked arm motion."""

    tcp_cross_path_velocity = RewTerm(
        func=mdp.tcp_cross_path_velocity_cost,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="dummy_target"),
            "command_name": "ee_target",
            "vertical_scale": 0.20,
            "lateral_scale": 0.20,
        },
    )


@configclass
class B2Z1VA4SmoothToolAMPEnvCfg(B2Z1VA3TruthCurveAMPEnvCfg):
    """VA3 continuation with a fixed-height path and a smooth tool objective."""

    rewards: VA4SmoothToolRewardsCfg = VA4SmoothToolRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # VA3's -0.38 m lower bound was active throughout steady playback:
        # the requested 9 cm world-height target moved up/down with the base.
        # The new range contains the nominal -0.43 m body-relative target.
        self.commands.ee_target.limit_ranges.pos_z = (-0.55, -0.30)
        # This penalizes only the arm actions. Preserve leg bandwidth for trot.
        self.rewards.arm_action_rate_l2.weight = -0.15


@configclass
class B2Z1VA4SmoothToolAMPEnvCfg_Play(B2Z1VA4SmoothToolAMPEnvCfg):
    """VA4 playback on the same uninterrupted left arc used for VA3 review."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 3600.0
        self.reference_state_initialization_prob = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.debug_vis = True
        self.commands.ee_target.debug_vis = True
        self.commands.ee_target.synthetic_straight_path_fraction = 0.0
        self.commands.ee_target.synthetic_arc_radius_range_m = (10.0, 10.0)
        self.commands.ee_target.synthetic_arc_direction = "left"
        self.commands.ee_target.synthetic_speed_range_mps = (0.22, 0.22)
        self.commands.ee_target.resampling_time_range = (1.0e6, 1.0e6)
        self.commands.ee_target.max_yaw_rate_rps = 0.18
        self.commands.ee_target.cross_track_gain = 0.50
        self.commands.ee_target.path_heading_gain = 0.90
        self.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_dynamics = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "roll", "pitch", "yaw")
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


@configclass
class VA5ElbowWristRewardsCfg(VA4SmoothToolRewardsCfg):
    """Discourage the opposed joint3/joint4 cycle seen in VA3 and VA4."""

    elbow_wrist_velocity = RewTerm(
        func=mdp.joint_group_velocity_huber,
        weight=-0.30,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint3", "joint4"]),
            "speed_scale": 1.0,
        },
    )
    elbow_wrist_action_rate = RewTerm(
        func=mdp.joint_group_action_rate_l2,
        weight=-0.50,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint3", "joint4"]),
        },
    )


@configclass
class B2Z1VA5ElbowWristAMPEnvCfg(B2Z1VA4SmoothToolAMPEnvCfg):
    """VA4 continuation targeting the arm's gait-synchronous nullspace motion."""

    rewards: VA5ElbowWristRewardsCfg = VA5ElbowWristRewardsCfg()


@configclass
class B2Z1VA5ElbowWristAMPEnvCfg_Play(B2Z1VA4SmoothToolAMPEnvCfg_Play):
    """VA5 on the identical uninterrupted left arc used for VA3/VA4 review."""

    rewards: VA5ElbowWristRewardsCfg = VA5ElbowWristRewardsCfg()


@configclass
class VA6NullspaceRewardsCfg(VA4SmoothToolRewardsCfg):
    """Penalize the ineffective opposed motion, not all Z1 joint motion."""

    elbow_wrist_opposed_action_rate = RewTerm(
        func=mdp.joint_pair_opposed_action_rate_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint3", "joint4"]),
        },
    )


@configclass
class B2Z1VA6NullspaceAMPEnvCfg(B2Z1VA4SmoothToolAMPEnvCfg):
    """VA4 continuation that targets only elbow/wrist nullspace cycling."""

    rewards: VA6NullspaceRewardsCfg = VA6NullspaceRewardsCfg()


@configclass
class B2Z1VA6NullspaceAMPEnvCfg_Play(B2Z1VA4SmoothToolAMPEnvCfg_Play):
    """VA6 on the same uninterrupted curve used to evaluate VA3 through VA5."""

    rewards: VA6NullspaceRewardsCfg = VA6NullspaceRewardsCfg()


@configclass
class VA7WholeBodyRewardsCfg(VA4SmoothToolRewardsCfg):
    """Preserve TCP accuracy while discouraging ineffective arm motion."""

    arm_nullspace_velocity = RewTerm(
        func=mdp.tcp_arm_nullspace_velocity_cost,
        weight=-0.35,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names="dummy_target",
                joint_names=[f"joint{index}" for index in range(1, 7)],
            ),
            "command_name": "ee_target",
            "velocity_scale": 0.6,
            "tracking_error_scale": 0.12,
            "damping": 0.05,
        },
    )


@configclass
class B2Z1VA7WholeBodyAMPEnvCfg(B2Z1VA4SmoothToolAMPEnvCfg):
    """From-scratch, single-policy 18-DoF B2+Z1 whole-body task."""

    rewards: VA7WholeBodyRewardsCfg = VA7WholeBodyRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep gait bandwidth during acquisition; VA4's rate and velocity
        # weights were designed for continuation from a trained VA3 policy.
        self.rewards.arm_action_rate_l2.weight = -0.08
        self.rewards.tcp_cross_path_velocity.weight = -0.20


@configclass
class B2Z1VA7WholeBodyAMPEnvCfg_Play(B2Z1VA4SmoothToolAMPEnvCfg_Play):
    """VA7 evaluation on the same continuous arc as the VA3/VA4 baseline."""

    rewards: VA7WholeBodyRewardsCfg = VA7WholeBodyRewardsCfg()


@configclass
class VA8PathCurriculumCfg(CurriculumCfg):
    """Require B2 path alignment, not just TCP and speed, before tighter arcs."""

    lin_vel_cmd_levels = None
    ang_vel_cmd_levels = None
    vel_cmd_to_pos_levels = None

    truth_curve_level = CurrTerm(
        func=mdp.truth_curve_levels,
        params={
            "command_name": "ee_target",
            "reward_term_names": (
                "track_lin_vel_xy",
                "track_ang_vel_z",
                "end_effector_position_tracking_exp",
                "along_path_speed_ratio",
                "crack_centerline_tracking",
                "crack_heading_tracking",
            ),
            "reward_thresholds": (0.55, 0.45, 0.55, 0.65, 0.55, 0.55),
            "straight_path_fractions": (0.50, 0.40, 0.30, 0.20),
            "arc_radius_ranges_m": (
                (14.0, 24.0),
                (10.0, 18.0),
                (7.0, 14.0),
                (6.0, 12.0),
            ),
            "speed_ranges_mps": (
                (0.16, 0.26),
                (0.18, 0.30),
                (0.20, 0.34),
                (0.20, 0.38),
            ),
            "max_yaw_rates_rps": (0.10, 0.14, 0.18, 0.22),
            "cross_track_gains": (0.45, 0.55, 0.65, 0.75),
            "path_heading_gains": (0.90, 1.00, 1.10, 1.20),
            "minimum_evaluation_episodes": 1024,
            "successful_evaluations_required": 2,
        },
    )


@configclass
class VA8PathRewardsCfg(VA7WholeBodyRewardsCfg):
    """Keep VA7 arm behaviour while penalizing B2 path and yaw drift."""

    base_centerline_error = RewTerm(
        func=mdp.crack_path_error_pseudo_huber,
        weight=-0.65,
        params={
            "command_name": "ee_target",
            "error_attribute": "_path_lateral_error_b",
            "scale": 0.15,
        },
    )
    base_heading_error = RewTerm(
        func=mdp.crack_path_error_pseudo_huber,
        weight=-0.50,
        params={
            "command_name": "ee_target",
            "error_attribute": "_path_heading_error_b",
            "scale": 0.20,
        },
    )


@configclass
class B2Z1VA8PathAMPEnvCfg(B2Z1VA7WholeBodyAMPEnvCfg):
    """VA7 continuation focusing on base path tracking, with the same 18 actions."""

    rewards: VA8PathRewardsCfg = VA8PathRewardsCfg()
    curriculum: VA8PathCurriculumCfg = VA8PathCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Widen the exp reward's useful range, while pseudo-Huber terms keep
        # their gradient after the base is farther from the curve.
        self.rewards.crack_centerline_tracking.weight = 1.0
        self.rewards.crack_centerline_tracking.params["std"] = 0.20
        self.rewards.crack_heading_tracking.weight = 0.60
        self.rewards.crack_heading_tracking.params["std"] = 0.35

        # Start with enough arcs to avoid relearning a straight-only gait.
        path = self.commands.ee_target
        path.synthetic_straight_path_fraction = 0.50
        path.synthetic_arc_radius_range_m = (14.0, 24.0)
        path.synthetic_speed_range_mps = (0.16, 0.26)
        path.max_yaw_rate_rps = 0.10
        path.cross_track_gain = 0.45
        path.path_heading_gain = 0.90


@configclass
class B2Z1VA8PathAMPEnvCfg_Play(B2Z1VA7WholeBodyAMPEnvCfg_Play):
    """VA8 on the same uninterrupted left arc used for VA7 review."""

    rewards: VA8PathRewardsCfg = VA8PathRewardsCfg()


@configclass
class VA9PreserveArmRewardsCfg(VA8PathRewardsCfg):
    """Lower path-reward pressure for leg-only adaptation from VA7."""


@configclass
class B2Z1VA9PreserveArmAMPEnvCfg(B2Z1VA8PathAMPEnvCfg):
    """One 18-action policy; fine-tuning is restricted to its B2 action head."""

    rewards: VA9PreserveArmRewardsCfg = VA9PreserveArmRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.crack_centerline_tracking.weight = 0.85
        self.rewards.crack_heading_tracking.weight = 0.45
        self.rewards.base_centerline_error.weight = -0.25
        self.rewards.base_heading_error.weight = -0.20


@configclass
class B2Z1VA9PreserveArmAMPEnvCfg_Play(B2Z1VA7WholeBodyAMPEnvCfg_Play):
    """Continuous-arc playback for VA9 on the VA7 comparison task."""

    rewards: VA9PreserveArmRewardsCfg = VA9PreserveArmRewardsCfg()


@configclass
class VA10SignedYawRewardsCfg(VA9PreserveArmRewardsCfg):
    """Reward the measured turn direction while retaining VA7 arm objectives."""

    signed_yaw_rate_error = RewTerm(
        func=mdp.commanded_yaw_rate_pseudo_huber,
        weight=-0.90,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "scale": 0.10,
        },
    )


@configclass
class B2Z1VA10SignedYawAMPEnvCfg(B2Z1VA9PreserveArmAMPEnvCfg):
    """One full-body actor with signed yaw, centreline and TCP objectives."""

    rewards: VA10SignedYawRewardsCfg = VA10SignedYawRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # VA8/VA9 showed that increasing path penalties alone can make the
        # robot crouch and sacrifice TCP tracking. Correct yaw response first.
        self.rewards.base_centerline_error.weight = -0.20
        self.rewards.base_heading_error.weight = -0.15
        self.rewards.crack_centerline_tracking.weight = 0.80
        self.rewards.crack_heading_tracking.weight = 0.40
        self.rewards.track_ang_vel_z.weight = 3.0
        self.rewards.track_ang_vel_z.params["std"] = 0.15


@configclass
class B2Z1VA10SignedYawAMPEnvCfg_Play(B2Z1VA7WholeBodyAMPEnvCfg_Play):
    """Use the unchanged VA7 fixed arc for direct checkpoint comparison."""

    rewards: VA10SignedYawRewardsCfg = VA10SignedYawRewardsCfg()


@configclass
class VA11SafetyCurriculumCfg(VA8PathCurriculumCfg):
    """VA8 curve schedule with explicit failure, survival and TCP gates."""


@configclass
class VA11SafetyRewardsCfg(VA10SignedYawRewardsCfg):
    """Guard body height instead of trading posture for path score."""

    moving_base_height_floor = RewTerm(
        func=mdp.moving_body_height_below_penalty,
        weight=-1.50,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "minimum_height": 0.46,
            "height_scale": 0.05,
            "command_name": "base_velocity",
            "velocity_threshold": 0.08,
        },
    )


@configclass
class B2Z1VA11SafetyAMPEnvCfg(B2Z1VA10SignedYawAMPEnvCfg):
    """Improve B2 path response while retaining VA7 Z1 and upright gait."""

    rewards: VA11SafetyRewardsCfg = VA11SafetyRewardsCfg()
    curriculum: VA11SafetyCurriculumCfg = VA11SafetyCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # VA10/8200 tracked the path but lost height and TCP accuracy.  Keep
        # the measured turn-direction signal, with lower path pressure.
        self.rewards.signed_yaw_rate_error.weight = -0.65
        self.rewards.base_centerline_error.weight = -0.15
        self.rewards.base_heading_error.weight = -0.10
        self.rewards.crack_centerline_tracking.weight = 0.75
        self.rewards.crack_heading_tracking.weight = 0.35
        self.rewards.track_ang_vel_z.weight = 2.6
        self.rewards.end_effector_position_tracking_fine_grained.weight = 1.25
        self.rewards.arm_action_rate_l2.weight = -0.12

        stages = self.curriculum.truth_curve_level.params
        stages["reward_term_names"] = (
            "track_lin_vel_xy",
            "track_ang_vel_z",
            "end_effector_position_tracking_exp",
            "along_path_speed_ratio",
            "crack_centerline_tracking",
            "crack_heading_tracking",
            "moving_base_height_floor",
        )
        stages["reward_thresholds"] = (0.55, 0.45, 0.70, 0.70, 0.50, 0.50, 0.025)
        stages["straight_path_fractions"] = (0.65, 0.50, 0.40, 0.40)
        stages["arc_radius_ranges_m"] = (
            (18.0, 26.0), (14.0, 22.0), (10.0, 18.0), (6.0, 12.0)
        )
        stages["speed_ranges_mps"] = (
            (0.16, 0.24), (0.16, 0.28), (0.18, 0.32), (0.20, 0.38)
        )
        stages["max_yaw_rates_rps"] = (0.09, 0.11, 0.14, 0.18)
        stages["cross_track_gains"] = (0.40, 0.45, 0.50, 0.50)
        stages["path_heading_gains"] = (0.85, 0.90, 0.90, 0.90)
        stages["max_terminated_fraction"] = 0.01
        stages["min_mean_episode_length_fraction"] = 0.95

        # These must match curriculum stage zero.  Later stages are applied
        # only after two consecutive batches satisfy every quality gate.
        path = self.commands.ee_target
        path.synthetic_straight_path_fraction = 0.65
        path.synthetic_arc_radius_range_m = (18.0, 26.0)
        path.synthetic_speed_range_mps = (0.16, 0.24)
        path.max_yaw_rate_rps = 0.09
        path.cross_track_gain = 0.40
        path.path_heading_gain = 0.85


@configclass
class B2Z1VA11SafetyAMPEnvCfg_Play(B2Z1VA7WholeBodyAMPEnvCfg_Play):
    """Use the same deterministic left arc as VA7/VA10 for comparison."""

    rewards: VA11SafetyRewardsCfg = VA11SafetyRewardsCfg()


@configclass
class VA12GaitPreserveRewardsCfg(VA11SafetyRewardsCfg):
    """Add leg-link safeguards for the straight-trot teacher experiment."""


@configclass
class B2Z1VA12GaitPreserveAMPEnvCfg(B2Z1VA11SafetyAMPEnvCfg):
    """Safety-gated path learning with a VA7 straight-gait action prior."""

    rewards: VA12GaitPreserveRewardsCfg = VA12GaitPreserveRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # VA11's base-height floor rarely activated; its failures involved
        # calf/thigh contact while the trunk height remained nominal.
        self.rewards.moving_calf_height.weight = -2.5
        self.rewards.moving_thigh_height.weight = -1.5
        self.rewards.leg_link_contacts.weight = -3.0


@configclass
class B2Z1VA12GaitPreserveAMPEnvCfg_Play(B2Z1VA7WholeBodyAMPEnvCfg_Play):
    """Fixed-arc playback using the VA7 comparison scene."""

    rewards: VA12GaitPreserveRewardsCfg = VA12GaitPreserveRewardsCfg()


@configclass
class VA13LowBaseTerminationsCfg(TerminationsCfg):
    """Treat a collapsed trunk as a failed episode before leg links fold."""

    low_base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.39, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class B2Z1VA13LowNoiseAMPEnvCfg(B2Z1VA12GaitPreserveAMPEnvCfg):
    """VA12 rewards and gait prior with a measured-height failure signal."""

    terminations: VA13LowBaseTerminationsCfg = VA13LowBaseTerminationsCfg()


@configclass
class B2Z1VA13LowNoiseAMPEnvCfg_Play(B2Z1VA12GaitPreserveAMPEnvCfg_Play):
    """Same fixed arc with the additional low-height termination active."""

    terminations: VA13LowBaseTerminationsCfg = VA13LowBaseTerminationsCfg()


@configclass
class B2Z1VA14FromScratchAMPEnvCfg(B2Z1VA13LowNoiseAMPEnvCfg):
    """Jointly acquire gait and steering, retaining the smooth TCP objectives."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.diagonal_contact_state_mismatch.weight = -0.55
        self.rewards.prolonged_foot_contact.weight = -0.9
        stages = self.curriculum.truth_curve_level.params
        stages["straight_path_fractions"] = (0.80, 0.65, 0.50, 0.40)
        stages["arc_radius_ranges_m"] = ((24.0, 36.0), (18.0, 26.0), (12.0, 20.0), (6.0, 12.0))
        stages["speed_ranges_mps"] = ((0.16, 0.24), (0.18, 0.28), (0.20, 0.32), (0.20, 0.38))
        stages["max_yaw_rates_rps"] = (0.08, 0.10, 0.14, 0.18)
        stages["cross_track_gains"] = (0.30, 0.40, 0.45, 0.50)
        stages["path_heading_gains"] = (0.70, 0.80, 0.85, 0.90)
        stages["reward_term_names"] += ("diagonal_contact_state_mismatch", "prolonged_foot_contact")
        stages["reward_thresholds"] += (0.10, 0.10)
        path = self.commands.ee_target
        path.synthetic_straight_path_fraction = 0.80
        path.synthetic_arc_radius_range_m = (24.0, 36.0)
        path.synthetic_arc_direction = "both"
        path.synthetic_speed_range_mps = (0.16, 0.24)
        path.max_yaw_rate_rps = 0.08
        path.cross_track_gain = 0.30
        path.path_heading_gain = 0.70


@configclass
class B2Z1VA14FromScratchAMPEnvCfg_Play(B2Z1VA13LowNoiseAMPEnvCfg_Play):
    """Uninterrupted fixed-arc comparison with no training teacher required."""


@configclass
class B2Z1VA15CurriculumAMPEnvCfg(B2Z1VA14FromScratchAMPEnvCfg):
    """Stage-dependent admission with explicit diagnostics; hard resets remain."""

    def __post_init__(self):
        super().__post_init__()
        stages = self.curriculum.truth_curve_level.params
        stages["max_terminated_fractions"] = (0.08, 0.06, 0.04, 0.03)
        stages["min_mean_episode_length_fractions"] = (0.90, 0.92, 0.94, 0.95)
        stages["emit_diagnostics"] = True


@configclass
class B2Z1VA15CurriculumAMPEnvCfg_Play(B2Z1VA14FromScratchAMPEnvCfg_Play):
    """Fixed comparison scene without a training curriculum during playback."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.truth_curve_level = None
