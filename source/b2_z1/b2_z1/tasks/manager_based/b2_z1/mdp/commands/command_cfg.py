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

from isaaclab.utils import configclass

from isaaclab.envs.mdp.commands import (
    UniformPoseCommandCfg as UniformPoseCommandBaseCfg,
)
from isaaclab.envs.mdp.commands import (
    UniformVelocityCommandCfg as UniformVelocityCommandBaseCfg,
)
from .pose_command_wbc import (
    UniformPoseWBCCommand,
    _build_interpolation_curve_visualizer_cfg,
    _build_tracked_trajectory_visualizer_cfg,
    _build_tracking_pose_visualizer_cfg,
    _build_trajectory_visualizer_cfg,
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
    trajectory_waypoint_count: int = 3
    """Number of intermediate waypoints used to shape the interpolation path."""
    trajectory_point_spacing: float = 0.02
    """Minimum distance in meters between two recorded end-effector trajectory markers."""
    trajectory_waypoint_offset_range_m: tuple[float, float] = (0.1, 0.2)
    """Waypoint offset distance range in meters used to bend the interpolated trajectory."""
    trajectory_start_pos: tuple[float, float, float] | None = None
    """Optional fixed trajectory start ``(x, y, z)``.

    The ``x`` and ``y`` coordinates are expressed in ``link_name`` frame,
    while ``z`` is expressed in world frame, matching the pose target
    convention. When unset, the trajectory starts at the current end-effector
    position.
    """
    
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
