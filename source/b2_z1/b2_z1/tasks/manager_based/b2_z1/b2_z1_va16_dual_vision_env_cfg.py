"""VA16: constant-speed B2 steering and rear RGB-D tool tracking.

The policy interface remains VA15's 70 observations / 18 joint actions.  The
high-throughput training task uses synthetic measurements; the Play and
Visual-Train tasks render both cameras for the external perception bridge.
Camera intrinsics and mounting transforms below are initial simulation
parameters, not a calibration of the physical D435i or its bracket.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from .b2_z1_amp_env_cfg import B2Z1SceneCfg
from .b2_z1_rear_vision_env_cfg import B2Z1RearVisionSceneCfg, _camera_model
from .b2_z1_va_env_cfg import (
    B2Z1VA15CurriculumAMPEnvCfg,
    B2Z1VA15CurriculumAMPEnvCfg_Play,
)
from .mdp.commands.dual_vision_command import DualVisionTrajectoryCommandCfg
from .mdp.commands.dual_camera_visuals import make_camera_model_cfg


FORWARD_SPEED_MPS = 0.20
CAMERA_PERIOD_S = 1.0 / 25.0
# Centre the additional camera on the rear shell, just outside the existing
# r_dc_link mount (-0.4081, -0.025, 0.024861). ROS optical +Z points backward
# and 45 degrees down; this is a shell mount, not an elevated side bracket.
REAR_CAMERA_POSITION_B = (-0.445, 0.0, 0.005)
REAR_CAMERA_QUATERNION_ROS = (
    0.27059805007309856, -0.6532814824381883,
    -0.6532814824381882, 0.2705980500730985,
)
# Marker centres and TCP are expressed in the unchanged link_gun frame.
# The bridge fits each sphere's centre from its visible depth surface, then
# estimates the TCP as 2 * cyan - magenta; no robot TCP truth is needed.
TOOL_MARKER_RADIUS_M = 0.035
TOOL_MARKER_AXIAL_SPACING_M = 0.10
TOOL_MARKER_TIP_EXTENSION_M = 0.10
TOOL_MARKER_CYAN_GUN_M = (0.0, -0.03, -0.52)
TOOL_MARKER_MAGENTA_GUN_M = (0.0, -0.03, -0.42)
TOOL_TCP_GUN_M = (0.0, -0.03, -0.62)


def _tool_marker(name: str, position: tuple[float, float, float], color):
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/link_gun/{name}",
        spawn=sim_utils.SphereCfg(
            radius=TOOL_MARKER_RADIUS_M,
            # These are optical fiducials only: they add neither contact
            # geometry nor another rigid body/mass to the policy's tool.
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, metallic=0.0, roughness=0.4,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
    )


@configclass
class VA16DualVisionSceneCfg(B2Z1RearVisionSceneCfg):
    """Front optical RGB and a D435i looking rearward and down at the tool.

    The inherited welding_gun is a contact-enabled compound cylinder under
    link_gun.  It adds no rigid body or inertia; link_gun retains the explicit
    0.001 kg mass in the original no_stick URDF.  The loaded 3.259 kg gun
    variant and its hose dynamics are deliberately separate tasks.
    """

    rear_rgb = None
    front_depth = None
    rear_depth = None
    d435i = None

    front_rgb = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_rgb",
        update_period=CAMERA_PERIOD_S,
        update_latest_camera_pose=True,
        width=640,
        height=480,
        data_types=["rgb"],
        spawn=_camera_model(460.0, 460.0),
        offset=CameraCfg.OffsetCfg(
            pos=(0.39079, 0.0, -0.013433),
            rot=(0.49999816, -0.5, 0.50000184, -0.5),
            convention="ros",
        ),
    )

    # Rear-centre optical axis: yaw=180 deg, 45 deg below horizontal.
    # Keep the requested shell mount even if tool/marker occlusion is found;
    # visibility is checked separately and never fixed by moving it sideways.
    rear_d435i = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/rear_d435i",
        update_period=CAMERA_PERIOD_S,
        width=640,
        height=480,
        data_types=["rgb", "distance_to_image_plane"],
        depth_clipping_behavior="zero",
        update_latest_camera_pose=True,
        spawn=_camera_model(392.5, 392.5, clipping_range=(0.10, 6.0)),
        offset=CameraCfg.OffsetCfg(
            pos=REAR_CAMERA_POSITION_B,
            rot=REAR_CAMERA_QUATERNION_ROS,
            convention="ros",
        ),
    )

    rear_d435i_model = make_camera_model_cfg(
        "{ENV_REGEX_NS}/Robot/base_link/rear_d435i_model",
        REAR_CAMERA_POSITION_B, REAR_CAMERA_QUATERNION_ROS,
    )

    tool_marker_cyan = _tool_marker(
        "tool_marker_cyan", TOOL_MARKER_CYAN_GUN_M, (0.0, 1.0, 1.0)
    )
    tool_marker_magenta = _tool_marker(
        "tool_marker_magenta", TOOL_MARKER_MAGENTA_GUN_M, (1.0, 0.0, 1.0)
    )


@configclass
class VA16SyntheticSceneCfg(VA16DualVisionSceneCfg):
    """Scalable proxy stage: retain the contact tool without RTX cameras.

    Synthetic straight/arc paths differ independently in each environment;
    therefore the single fixed physical crack used for camera validation is
    replaced with VA15's flat road in this stage.
    """

    terrain = B2Z1SceneCfg(num_envs=1, env_spacing=4.0).terrain.copy()
    road_left = None
    road_right = None
    front_rgb = None
    rear_d435i = None
    rear_d435i_model = None


def _dual_command(source, mode: str):
    """Retain trained pose/workspace/filter settings when changing the source."""
    values = {
        field.name: deepcopy(getattr(source, field.name))
        for field in fields(source)
        if field.name != "class_type"
    }
    command = DualVisionTrajectoryCommandCfg(**values)
    command.mode = mode
    command.front_camera_name = "front_rgb"
    command.rear_camera_name = "rear_d435i"
    command.path_speed_mps = FORWARD_SPEED_MPS
    command.synthetic_speed_range_mps = (FORWARD_SPEED_MPS, FORWARD_SPEED_MPS)
    command.camera_export_period_steps = 2  # 50 Hz policy -> 25 Hz camera.
    command.camera_ring_slots = 8
    command.vision_exchange_directory = os.environ.get(
        "B2_Z1_DUAL_VISION_EXCHANGE", "/tmp/b2_z1_va16_dual_vision"
    )
    command.trajectory_file = os.path.join(
        command.vision_exchange_directory, "latest_dual_trajectory.json"
    )
    if mode == "file":
        command.resampling_time_range = (1.0e6, 1.0e6)
        command.debug_vis = True
    return command


def _configure_dual_cameras(env_cfg):
    """Keep both cameras; never use VA15's single selected-camera pruning."""
    command = env_cfg.commands.ee_target
    front = getattr(env_cfg.scene, command.front_camera_name, None)
    rear = getattr(env_cfg.scene, command.rear_camera_name, None)
    if front is None or front.data_types != ["rgb"]:
        raise ValueError("VA16 front camera must supply optical RGB only")
    if rear is None or not {"rgb", "distance_to_image_plane"}.issubset(rear.data_types):
        raise ValueError("VA16 rear D435i must supply aligned RGB-D")
    env_cfg.sim.render_interval = env_cfg.decimation * command.camera_export_period_steps
    period = env_cfg.sim.dt * env_cfg.sim.render_interval
    front.update_period = period
    rear.update_period = period
    env_cfg.rerender_on_reset = True
    env_cfg.sim.render.antialiasing_mode = "DLAA"
    env_cfg.sim.render.enable_dl_denoiser = True
    env_cfg.sim.render.enable_shadows = False
    # Start behind the robot, looking at the shell-mounted camera and its
    # downward field of view. Follow translation as the robot walks.
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (-2.0, 1.7, 1.1)
    env_cfg.viewer.lookat = (-0.35, 0.0, -0.15)


@configclass
class B2Z1VA16DualVisionAMPEnvCfg(B2Z1VA15CurriculumAMPEnvCfg):
    """Fixed-speed proxy training; this alone does not train image perception."""

    scene: VA16SyntheticSceneCfg = VA16SyntheticSceneCfg(num_envs=1024, env_spacing=4.0)

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_target = _dual_command(self.commands.ee_target, "synthetic")
        # One shared trajectory owns start/stop and the working speed. An
        # independent standing mask would stop B2 while EE guidance advances.
        self.commands.base_velocity.rel_standing_envs = 0.0
        stages = self.curriculum.truth_curve_level.params
        stages["speed_ranges_mps"] = ((FORWARD_SPEED_MPS, FORWARD_SPEED_MPS),) * len(
            stages["speed_ranges_mps"]
        )


@configclass
class B2Z1VA16DualVisionAMPEnvCfg_Play(B2Z1VA15CurriculumAMPEnvCfg_Play):
    """One robot with actual rendered front RGB and rear RGB-D feedback."""

    scene: VA16DualVisionSceneCfg = VA16DualVisionSceneCfg(num_envs=1, env_spacing=4.0)

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_target = _dual_command(self.commands.ee_target, "file")
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.debug_vis = True
        _configure_dual_cameras(self)


@configclass
class B2Z1VA16DualVisionAMPEnvCfg_VisualTrain(B2Z1VA16DualVisionAMPEnvCfg):
    """Optional one-environment image-feedback fine-tuning with the bridge.

    File-based live perception currently supports exactly one environment;
    a larger rendered training batch needs a batched perception transport.
    """

    scene: VA16DualVisionSceneCfg = VA16DualVisionSceneCfg(num_envs=1, env_spacing=4.0)

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_target = _dual_command(self.commands.ee_target, "file")
        self.curriculum.truth_curve_level = None
        _configure_dual_cameras(self)
