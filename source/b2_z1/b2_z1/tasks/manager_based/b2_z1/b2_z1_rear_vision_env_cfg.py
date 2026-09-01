"""Rear-facing Z1 training and RGB-D visual-trajectory playback configs."""

from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from . import mdp
from .b2_z1_amp_env_cfg import B2Z1AMPEnvCfg, B2Z1SceneCfg


RGB_WIDTH = 640
RGB_HEIGHT = 480


def _camera_model(fx: float, fy: float, clipping_range=(0.07, 6.0)):
    return sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=[fx, 0.0, 319.5, 0.0, fy, 239.5, 0.0, 0.0, 1.0],
        width=RGB_WIDTH,
        height=RGB_HEIGHT,
        clipping_range=clipping_range,
    )


@configclass
class B2Z1RearVisionSceneCfg(B2Z1SceneCfg):
    """B2 built-in front/rear RGB-D and optional front-down D435i."""

    # A light road strip and a narrow dark crack make the visual playback
    # genuinely observable by the front-down camera.  Both are visual-only:
    # contact and robot dynamics still come from B2Z1SceneCfg.terrain.
    road_surface = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RoadSurface",
        spawn=sim_utils.CuboidCfg(
            size=(30.0, 8.0, 0.002),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.42, 0.42, 0.42), roughness=0.95
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(14.5, 0.0, 0.001)),
    )
    simulated_crack = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SimulatedCrack",
        spawn=sim_utils.CuboidCfg(
            # Kept wider than a hairline crack so it remains several pixels
            # wide at the far end of the 640x480 training camera image.
            size=(30.0, 0.050, 0.006),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.003, 0.003, 0.003), roughness=1.0
            ),
        ),
        # A small visual-only lift avoids z-fighting with the road strip.
        init_state=AssetBaseCfg.InitialStateCfg(pos=(14.5, 0.0, 0.008)),
    )

    # Built-in optical cameras. Their transforms come from joint_f_oc and
    # joint_r_oc in the B2 URDF, expressed directly relative to base_link so
    # fixed-joint collapsing in the URDF converter cannot remove the parent.
    front_rgb = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_rgb",
        update_period=1.0 / 15.0,
        width=RGB_WIDTH,
        height=RGB_HEIGHT,
        data_types=["rgb"],
        spawn=_camera_model(460.0, 460.0),
        offset=CameraCfg.OffsetCfg(
            pos=(0.39079, 0.0, -0.013433),
            rot=(0.49999816, -0.5, 0.50000184, -0.5),
            convention="ros",
        ),
    )
    rear_rgb = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/rear_rgb",
        update_period=1.0 / 15.0,
        width=RGB_WIDTH,
        height=RGB_HEIGHT,
        data_types=["rgb"],
        spawn=_camera_model(460.0, 460.0),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.38451, 0.0, -0.023249),
            rot=(0.49999816, -0.5, -0.50000184, 0.5),
            convention="ros",
        ),
    )

    # Built-in depth cameras are pitched down approximately 45 degrees.
    front_depth = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_depth",
        update_period=1.0 / 15.0,
        width=RGB_WIDTH,
        height=RGB_HEIGHT,
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="zero",
        spawn=_camera_model(430.0, 430.0),
        offset=CameraCfg.OffsetCfg(
            pos=(0.41251, 0.024997, 0.04765),
            rot=(0.27059575, -0.65328103, 0.65328343, -0.27059675),
            convention="ros",
        ),
    )
    rear_depth = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/rear_depth",
        update_period=1.0 / 15.0,
        width=RGB_WIDTH,
        height=RGB_HEIGHT,
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="zero",
        spawn=_camera_model(430.0, 430.0),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.4081, -0.025, 0.024861),
            rot=(0.27075907, -0.65321336, -0.65321576, 0.27076006),
            convention="ros",
        ),
    )

    # Optional D435i (the requested "D35i" is treated as D435i until the
    # exact model is confirmed). Mount it ahead of the B2 head and above the
    # chassis, looking about 45 degrees down.  The forward offset is
    # intentional: a mount near x=0.30 m is inside the arm/chassis silhouette
    # and loses most of the crack image to self-occlusion.
    d435i = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/d435i",
        update_period=1.0 / 15.0,
        width=RGB_WIDTH,
        height=RGB_HEIGHT,
        data_types=["rgb", "distance_to_image_plane"],
        depth_clipping_behavior="zero",
        # Nominal square-pixel intrinsics for the 640x480 simulation stream.
        # Replace with the calibrated matrix for the physical D435i.
        spawn=_camera_model(392.5, 392.5),
        offset=CameraCfg.OffsetCfg(
            # Requires a short front/top bracket.  At this height the arm and
            # front shell stay outside most of the road-looking field of view.
            pos=(0.48, 0.0, 0.34),
            rot=(0.27059805, -0.65328148, 0.65328148, -0.27059805),
            convention="ros",
        ),
    )


def _rear_command(mode: str, debug_vis: bool):
    exchange_dir = os.environ.get("B2_Z1_VISION_EXCHANGE", "/tmp/b2_z1_vision_bridge")
    ranges = mdp.command_cfg.UniformPoseWBCCommandCfg.Ranges(
        pos_x=(-1.0, -0.55),
        pos_y=(-0.15, 0.15),
        pos_z=(-0.38, -0.30),
        roll=(0.0, 0.0),
        pitch=(math.pi / 2.0, math.pi / 2.0),
        yaw=(math.pi, math.pi),
    )
    return mdp.RearVisionTrajectoryCommandCfg(
        asset_name="robot",
        body_name="dummy_target",
        link_name="base_link",
        base_name="base_link",
        resampling_time_range=(8.0, 10.0),
        debug_vis=debug_vis,
        mode=mode,
        trajectory_file=os.environ.get(
            "B2_Z1_VISION_TRAJECTORY",
            os.path.join(exchange_dir, "latest_trajectory.json"),
        ),
        vision_camera_name="d435i",
        vision_exchange_directory=exchange_dir,
        ranges=ranges,
        limit_ranges=ranges,
        curriculum_enabled=False,
        rear_work_x_range_m=(-1.0, -0.55),
        tool_height_m=0.10,
        synthetic_ground_z_b_m=-0.45,
        path_speed_mps=0.25,
        synthetic_speed_range_mps=(0.10, 0.60),
    )


@configclass
class B2Z1RearSyntheticAMPEnvCfg(B2Z1AMPEnvCfg):
    """Efficient rear-arm training with synthetic crack tangents (no RTX)."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_target = _rear_command("synthetic", debug_vis=False)
        # EEDerivedVelocityCommand delegates to the trajectory tangent exposed
        # by RearVisionTrajectoryCommand, so it no longer points toward a rear
        # target and accidentally commands backward walking.
        self.commands.base_velocity.curriculum_enabled = False


@configclass
class B2Z1RearSyntheticAMPEnvCfg_Play(B2Z1RearSyntheticAMPEnvCfg):
    """Deterministic playback using the synthetic path command from training."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 120.0
        self.commands.ee_target = _rear_command("synthetic", debug_vis=True)
        self.commands.base_velocity.curriculum_enabled = False
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
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.reference_state_initialization_prob = 1.0
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class B2Z1RearVisionEnvCfg_Play(B2Z1RearSyntheticAMPEnvCfg):
    """Single-environment RGB-D playback fed by RHACrackNet trajectory JSON."""

    scene: B2Z1RearVisionSceneCfg = B2Z1RearVisionSceneCfg(num_envs=1, env_spacing=4.0)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 120.0
        self.commands.ee_target = _rear_command("file", debug_vis=True)
        self.commands.base_velocity.curriculum_enabled = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_dynamics = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # The simulated crack is aligned with world +X. Deterministic playback
        # must therefore start B2 at the origin with zero yaw and velocity.
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.reference_state_initialization_prob = 1.0
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"
        self.sim.render_interval = self.decimation
