# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from isaaclab.envs.mdp.commands import (
    UniformPoseCommandCfg as UniformPoseCommandBaseCfg,
)
from isaaclab.envs.mdp.commands import (
    UniformVelocityCommandCfg as UniformVelocityCommandBaseCfg,
)
from .curve_velocity_command import CurveVelocityCommand
from .pose_command_wbc import (
    UniformPoseWBCCommand,
    _build_interpolation_curve_visualizer_cfg,
    _build_tracked_trajectory_visualizer_cfg,
    _build_tracking_pose_visualizer_cfg,
    _build_trajectory_visualizer_cfg,
)


def _build_ground_curve_visualizer_cfg():
    """A unit box instanced as a thin ground strip between consecutive curve samples."""
    return VisualizationMarkersCfg(
        prim_path="/Visuals/Command/base_ground_curve",
        markers={
            "ground_curve": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.05)),
            )
        },
    )


# -- pose
@configclass
class UniformPoseWBCCommandCfg(UniformPoseCommandBaseCfg):

    class_type: type  = UniformPoseWBCCommand

    ranges: UniformPoseCommandBaseCfg.Ranges = None  # type: ignore
    """The initial range"""
    limit_ranges: UniformPoseCommandBaseCfg.Ranges = None  # type: ignore
    """The range limit """
    link_name: str = "link00"
    """The name of the root body, which serves as the reference frame for the command."""
    base_name: str = "base_link"
    """The name of the robot base body used as the output command frame."""
    curriculum_enabled: bool = False
    trajectory_num_points: int = 20
    """Number of points used to visualize the interpolation curve and tracked path."""
    trajectory_point_spacing: float = 0.02
    """Minimum distance in meters between two recorded end-effector trajectory markers."""
    trajectory_curve_offset_scale: float = 0.1
    """Relative scale of the random Bezier control-point offsets used to bend the interpolated trajectory."""
    tracking_pose_visualizer_cfg = _build_tracking_pose_visualizer_cfg()
    interpolation_curve_visualizer_cfg = _build_interpolation_curve_visualizer_cfg()
    tracked_trajectory_visualizer_cfg = _build_tracked_trajectory_visualizer_cfg()
    trajectory_visualizer_cfg = _build_trajectory_visualizer_cfg()


# -- velocity
@configclass
class UniformVelocityCommandCfg(UniformVelocityCommandBaseCfg):
    """Configuration for the uniform velocity command generator."""

    ranges: UniformVelocityCommandBaseCfg.Ranges = None  # type: ignore
    """The initial range"""

    limit_ranges: UniformVelocityCommandBaseCfg.Ranges = None  # type: ignore
    """The range limit """
    curriculum_enabled: bool = False


@configclass
class CurveVelocityCommandCfg(UniformVelocityCommandCfg):
    """Fixed-speed navigation along an independent, world-fixed ground Bezier curve."""

    class_type: type = CurveVelocityCommand
    base_name: str = "base_link"
    forward_speed: float = 0.4
    lookahead_distance: float = 0.6
    """Distance scale (m) for the lateral-error correction to the tangent heading."""
    heading_control_stiffness: float = 1.5
    # Yaw is derived from camera observations rather than the parent's random heading sampler.
    heading_command: bool = False
    rel_standing_envs: float = 0.0
    trajectory_num_points: int = 96
    trajectory_curve_offset_scale: float = 0.1
    """Maximum lateral control-point offset as a fraction of the curve length."""
    ground_height: float = 0.0
    """Ground plane height relative to each environment origin."""
    ground_curve_width: float = 0.04
    """Width (m) of the rendered ground curve strip."""
    ground_curve_thickness: float = 0.002
    """Thickness (m) of the rendered ground curve strip; visualization only, without collision."""
    ground_curve_z_offset: float = 0.002
    """Strip center height above the ground (m), avoiding coplanar rendering artifacts."""

    camera_position_b: tuple[float, float, float] = (0.25, 0.0, 0.05)
    camera_rpy_b: tuple[float, float, float] = (0.0, math.radians(35.0), 0.0)
    """Camera mount RPY (rad) in x-forward, y-left, z-up convention; positive pitch looks down."""
    camera_intrinsics: tuple[float, float, float, float] = (320.0, 320.0, 320.0, 240.0)
    """Ideal pinhole (fx, fy, cx, cy), in pixels; configure for the actual camera."""
    camera_image_size: tuple[int, int] = (640, 480)
    camera_depth_range: tuple[float, float] = (0.05, 5.0)
    """Visible forward depth range (m) in camera coordinates."""
    camera_update_period: float = 0.1
    """RGB sensor update period (s) in Play; geometry commands still update every control step."""

    ground_curve_visualizer_cfg = _build_ground_curve_visualizer_cfg()
    heading_visualizer_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/base_heading")


class EEDerivedVelocityCommandCfg(UniformVelocityCommandCfg):

    arrival_threshold: float = 0.1
    """Distance threshold (meters) to the ee_target world position """
