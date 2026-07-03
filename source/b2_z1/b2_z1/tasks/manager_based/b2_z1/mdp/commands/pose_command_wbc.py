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

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, POSITION_GOAL_MARKER_CFG
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    quat_unique,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .command_cfg import UniformPoseWBCCommandCfg


class UniformPoseWBCCommand(CommandTerm):
    """Command generator for pose tracking with interpolated target motion."""

    cfg: UniformPoseWBCCommandCfg

    def __init__(self, cfg: UniformPoseWBCCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.link0_idx = self.robot.find_bodies(cfg.link_name)[0][0]
        self.base_idx = self.robot.find_bodies(cfg.base_name)[0][0]

        self.pose_command = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command[:, 3] = 1.0
        self.pose_start = torch.zeros_like(self.pose_command)
        self.pose_start[:, 3] = 1.0
        self.pose_target = torch.zeros_like(self.pose_command)
        self.pose_target[:, 3] = 1.0
        self.pose_start_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.pose_target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self.pose_command_w = torch.zeros_like(self.pose_command)
        self.pose_command_w[:, 3] = 1.0
        self.target_pose_w = torch.zeros_like(self.pose_command)
        self.target_pose_w[:, 3] = 1.0
        self.tracking_pose_w = torch.zeros_like(self.pose_command)
        self.tracking_pose_w[:, 3] = 1.0
        self._tracking_pose_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._tracking_pose_quat_w[:, 3] = 1.0
        self._trajectory_waypoint_count = max(1, getattr(cfg, "trajectory_waypoint_count", 3))
        self._trajectory_waypoints_w = torch.zeros(self.num_envs, self._trajectory_waypoint_count, 3, device=self.device)

        self._trajectory_num_points = max(2, getattr(cfg, "trajectory_num_points", 10))
        self._trajectory_samples = torch.linspace(0.0, 1.0, self._trajectory_num_points, device=self.device)
        self._visualization_num_points = max(2, getattr(cfg, "trajectory_visualization_num_points", 100))
        self._visualization_samples = torch.linspace(0.0, 1.0, self._visualization_num_points, device=self.device)
        self._tracked_trajectory_num_points = self._visualization_num_points
        self._command_duration = torch.ones(self.num_envs, device=self.device)
        self._trajectory_point_spacing = getattr(cfg, "trajectory_point_spacing", 0.02)
        self.interpolation_curve_positions_w = torch.zeros(
            self.num_envs, self._visualization_num_points, 3, device=self.device
        )
        self.interpolation_curve_orientations_w = torch.zeros(
            self.num_envs, self._visualization_num_points, 4, device=self.device
        )
        self.interpolation_curve_orientations_w[..., 0] = 1.0
        self.tracked_trajectory_base_positions = torch.zeros(
            self.num_envs, self._tracked_trajectory_num_points, 3, device=self.device
        )
        self.tracked_trajectory_positions_w = torch.zeros(
            self.num_envs, self._tracked_trajectory_num_points, 3, device=self.device
        )
        self.tracked_trajectory_orientations_w = torch.zeros(
            self.num_envs, self._tracked_trajectory_num_points, 4, device=self.device
        )
        self.tracked_trajectory_orientations_w[..., 0] = 1.0
        self._tracked_trajectory_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.env = env

    def __str__(self) -> str:
        msg = "UniformPoseWBCCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tTrajectory points: {self._trajectory_num_points}\n"
        msg += f"\tTrajectory waypoints: {self._trajectory_waypoint_count}\n"
        msg += f"\tVisualization points: {self._visualization_num_points}\n"
        msg += f"\tTracked trajectory points: {self._tracked_trajectory_num_points}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command

    def _update_metrics(self):
        ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
        base_pos_w = self.robot.data.body_pos_w[:, self.base_idx]
        base_quat_w = self.robot.data.body_state_w[:, self.base_idx, 3:7]

        ee_pos_base = quat_apply_inverse(base_quat_w, ee_pos_w - base_pos_w)
        pos_error = ee_pos_base - self.pose_command[:, :3]

        des_quat_w = quat_mul(base_quat_w, self.pose_command[:, 3:7])
        curr_quat_w = self.robot.data.body_state_w[:, self.body_idx, 3:7]
        source_quat_norm = quat_mul(des_quat_w, quat_conjugate(des_quat_w))[:, 0]
        source_quat_inv = quat_conjugate(des_quat_w) / source_quat_norm.unsqueeze(-1)
        quat_error = quat_mul(curr_quat_w, source_quat_inv)
        rot_error = axis_angle_from_quat(quat_error)

        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        self._capture_current_pose_as_start(env_ids)
        self._sample_target_pose(env_ids)
        self.pose_target_pos_w[env_ids] = self._get_target_pos_w(env_ids)
        self._sample_trajectory_waypoints(env_ids)
        self._tracking_pose_quat_w[env_ids] = self._get_target_quat_w(env_ids)
        self._command_duration[env_ids] = self.time_left[env_ids].clamp_min(self.env.step_dt)
        self._update_command_from_world_pose(
            self._get_curve_start_pos_w(env_ids), self._tracking_pose_quat_w[env_ids], self.pose_command[env_ids], env_ids
        )
        self.target_pose_w[env_ids, :3] = self.pose_target_pos_w[env_ids]
        self.target_pose_w[env_ids, 3:] = self._tracking_pose_quat_w[env_ids]
        self.tracking_pose_w[env_ids, :3] = self._get_curve_start_pos_w(env_ids)
        self.tracking_pose_w[env_ids, 3:] = self._tracking_pose_quat_w[env_ids]
        self._reset_tracked_trajectory(env_ids)
        self._cache_interpolation_curve_visualization(env_ids)

    def _update_command(self):
        interpolation_progress = 1.0 - self.time_left / self._command_duration.clamp_min(self.env.step_dt)
        interpolation_progress = interpolation_progress.clamp_(0.0, 1.0).unsqueeze(-1)

        current_position_w = self._evaluate_curve_w(interpolation_progress)

        self._update_command_from_world_pose(
            current_position_w, self._tracking_pose_quat_w, self.pose_command
        )
        self.tracking_pose_w[:, :3] = current_position_w
        self.tracking_pose_w[:, 3:] = self._tracking_pose_quat_w

    def _sample_target_pose(self, env_ids: torch.Tensor):
        num_resample = len(env_ids)

        x_lim = torch.tensor(self.cfg.ranges.pos_x, device=self.device)
        y_lim = torch.tensor(self.cfg.ranges.pos_y, device=self.device)
        z_lim = torch.tensor(self.cfg.ranges.pos_z, device=self.device)
        rand_pos = torch.rand(num_resample, 3, device=self.device)

        self.pose_target[env_ids, 0] = x_lim[0] + rand_pos[:, 0] * (x_lim[1] - x_lim[0])
        self.pose_target[env_ids, 1] = y_lim[0] + rand_pos[:, 1] * (y_lim[1] - y_lim[0])
        self.pose_target[env_ids, 2] = z_lim[0] + rand_pos[:, 2] * (z_lim[1] - z_lim[0])

        roll_lim = torch.tensor(self.cfg.ranges.roll, device=self.device)
        pitch_lim = torch.tensor(self.cfg.ranges.pitch, device=self.device)
        yaw_lim = torch.tensor(self.cfg.ranges.yaw, device=self.device)
        rand_euler = torch.rand(num_resample, 3, device=self.device)
        euler_angles = torch.zeros(num_resample, 3, device=self.device)

        euler_angles[:, 0] = roll_lim[0] + rand_euler[:, 0] * (roll_lim[1] - roll_lim[0])
        euler_angles[:, 1] = pitch_lim[0] + rand_euler[:, 1] * (pitch_lim[1] - pitch_lim[0])
        euler_angles[:, 2] = yaw_lim[0] + rand_euler[:, 2] * (yaw_lim[1] - yaw_lim[0])

        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        if self.cfg.make_quat_unique:
            self.pose_target[env_ids, 3:] = quat_unique(quat)
        else:
            self.pose_target[env_ids, 3:] = quat

    def _capture_current_pose_as_start(self, env_ids: torch.Tensor):
        link0_pos_w = self.robot.data.body_pos_w[env_ids, self.link0_idx]
        link0_quat_w = self.robot.data.body_state_w[env_ids, self.link0_idx, 3:7]
        ee_pos_w = self.robot.data.body_pos_w[env_ids, self.body_idx]
        ee_quat_w = self.robot.data.body_state_w[env_ids, self.body_idx, 3:7]

        self.pose_start_pos_w[env_ids] = ee_pos_w
        start_pos_link0 = quat_apply_inverse(link0_quat_w, ee_pos_w - link0_pos_w)
        self.pose_start[env_ids, :2] = start_pos_link0[:, :2]
        self.pose_start[env_ids, 2] = ee_pos_w[:, 2]
        self.pose_start[env_ids, 3:] = quat_mul(quat_conjugate(link0_quat_w), ee_quat_w)

    def _sample_trajectory_waypoints(self, env_ids: torch.Tensor):
        start_pos = self.pose_start_pos_w[env_ids]
        target_pos = self.pose_target_pos_w[env_ids]
        delta = target_pos - start_pos
        length = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1.0e-4)
        tangent = delta / length

        z_axis = torch.zeros_like(tangent)
        z_axis[:, 2] = 1.0
        x_axis = torch.zeros_like(tangent)
        x_axis[:, 0] = 1.0
        use_x_axis = torch.abs(torch.sum(tangent * z_axis, dim=-1, keepdim=True)) > 0.9
        reference_axis = torch.where(use_x_axis, x_axis, z_axis)

        normal_1 = torch.linalg.cross(tangent, reference_axis)
        normal_1 = normal_1 / torch.linalg.norm(normal_1, dim=-1, keepdim=True).clamp_min(1.0e-4)
        normal_2 = torch.linalg.cross(tangent, normal_1)
        normal_2 = normal_2 / torch.linalg.norm(normal_2, dim=-1, keepdim=True).clamp_min(1.0e-4)

        waypoint_progress = torch.linspace(
            1.0 / (self._trajectory_waypoint_count + 1),
            float(self._trajectory_waypoint_count) / (self._trajectory_waypoint_count + 1),
            self._trajectory_waypoint_count,
            device=self.device,
        ).view(1, self._trajectory_waypoint_count, 1)
        base_waypoints = start_pos.unsqueeze(1) + waypoint_progress * delta.unsqueeze(1)

        waypoint_offsets = (torch.rand(len(env_ids), self._trajectory_waypoint_count, 2, device=self.device) * 2.0) - 1.0
        waypoint_offset_range_m = getattr(self.cfg, "trajectory_waypoint_offset_range_m", (0.1, 0.2))
        offset_min_m = float(waypoint_offset_range_m[0])
        offset_max_m = float(waypoint_offset_range_m[1])
        offset_magnitude = torch.rand(len(env_ids), self._trajectory_waypoint_count, 1, device=self.device)
        offset_magnitude = offset_magnitude * (offset_max_m - offset_min_m) + offset_min_m
        offset_direction = waypoint_offsets / torch.linalg.norm(
            waypoint_offsets, dim=-1, keepdim=True
        ).clamp_min(1.0e-4)

        self._trajectory_waypoints_w[env_ids] = base_waypoints
        self._trajectory_waypoints_w[env_ids] += offset_magnitude * (
            offset_direction[:, :, 0:1] * normal_1.unsqueeze(1)
            + offset_direction[:, :, 1:2] * normal_2.unsqueeze(1)
        )
        self._trajectory_waypoints_w[env_ids, :, 2] = base_waypoints[:, :, 2]

    def _reset_tracked_trajectory(self, env_ids: torch.Tensor):
        current_trajectory_point_w = self._get_current_trajectory_points(env_ids)
        self.tracked_trajectory_base_positions[env_ids] = current_trajectory_point_w.unsqueeze(1)
        self.tracked_trajectory_positions_w[env_ids] = 0.0
        self.tracked_trajectory_orientations_w[env_ids] = 0.0
        self.tracked_trajectory_orientations_w[env_ids, :, 0] = 1.0
        self._tracked_trajectory_counts[env_ids] = 1

    def _evaluate_curve(self, progress: torch.Tensor) -> torch.Tensor:
        return self._evaluate_curve_w(progress)

    def _evaluate_curve_w(self, progress: torch.Tensor) -> torch.Tensor:
        squeeze_output = progress.ndim == 2 and progress.shape[-1] == 1
        if progress.ndim == 1:
            progress = progress.view(1, -1, 1)
        elif progress.ndim == 2:
            progress = progress.unsqueeze(-1)

        if progress.shape[0] == 1 and self.num_envs > 1:
            progress = progress.expand(self.num_envs, -1, -1)

        knots = torch.cat(
            (
                self.pose_start_pos_w.unsqueeze(1),
                self._trajectory_waypoints_w,
                self.pose_target_pos_w.unsqueeze(1),
            ),
            dim=1,
        )
        num_segments = knots.shape[1] - 1

        segment_progress = progress.clamp(0.0, 1.0) * num_segments
        segment_idx = torch.floor(segment_progress).long().clamp(max=num_segments - 1)
        local_progress = segment_progress - segment_idx.to(segment_progress.dtype)

        gather_idx = segment_idx.expand(-1, -1, knots.shape[-1])
        segment_start = torch.gather(knots[:, :-1, :], 1, gather_idx)
        segment_end = torch.gather(knots[:, 1:, :], 1, gather_idx)

        positions_w = (1.0 - local_progress) * segment_start + local_progress * segment_end
        if squeeze_output:
            return positions_w[:, 0]
        return positions_w

    def _get_target_quat_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            link0_quat_w = self.robot.data.body_state_w[:, self.link0_idx, 3:7]
            pose_target = self.pose_target
        else:
            link0_quat_w = self.robot.data.body_state_w[env_ids, self.link0_idx, 3:7]
            pose_target = self.pose_target[env_ids]

        return quat_mul(link0_quat_w, pose_target[:, 3:])

    def _update_command_from_world_pose(
        self,
        position_w: torch.Tensor,
        quat_w: torch.Tensor,
        pose_command: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ):
        if env_ids is None:
            base_pos_w = self.robot.data.body_pos_w[:, self.base_idx]
            base_quat_w = self.robot.data.body_state_w[:, self.base_idx, 3:7]
        else:
            base_pos_w = self.robot.data.body_pos_w[env_ids, self.base_idx]
            base_quat_w = self.robot.data.body_state_w[env_ids, self.base_idx, 3:7]

        pose_command[:, :3] = quat_apply_inverse(base_quat_w, position_w - base_pos_w)
        pose_command[:, 3:] = quat_mul(quat_conjugate(base_quat_w), quat_w)

    def _get_target_pos_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            link0_pos_w = self.robot.data.body_pos_w[:, self.link0_idx]
            link0_quat_w = self.robot.data.body_state_w[:, self.link0_idx, 3:7]
            pose_target = self.pose_target
        else:
            link0_pos_w = self.robot.data.body_pos_w[env_ids, self.link0_idx]
            link0_quat_w = self.robot.data.body_state_w[env_ids, self.link0_idx, 3:7]
            pose_target = self.pose_target[env_ids]

        target_xy_w = link0_pos_w + quat_apply(
            link0_quat_w, torch.cat((pose_target[:, :2], torch.zeros_like(pose_target[:, 2:3])), dim=-1)
        )
        target_pos_w = torch.empty_like(pose_target[:, :3])
        target_pos_w[:, :2] = target_xy_w[:, :2]
        target_pos_w[:, 2] = pose_target[:, 2]
        return target_pos_w

    def _get_curve_start_pos_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            link0_pos_w = self.robot.data.body_pos_w[:, self.link0_idx]
            link0_quat_w = self.robot.data.body_state_w[:, self.link0_idx, 3:7]
            pose_start = self.pose_start
        else:
            link0_pos_w = self.robot.data.body_pos_w[env_ids, self.link0_idx]
            link0_quat_w = self.robot.data.body_state_w[env_ids, self.link0_idx, 3:7]
            pose_start = self.pose_start[env_ids]

        start_xy_w = link0_pos_w + quat_apply(
            link0_quat_w, torch.cat((pose_start[:, :2], torch.zeros_like(pose_start[:, 2:3])), dim=-1)
        )
        start_pos_w = torch.empty_like(pose_start[:, :3])
        start_pos_w[:, :2] = start_xy_w[:, :2]
        start_pos_w[:, 2] = pose_start[:, 2]
        return start_pos_w

    def _cache_interpolation_curve_visualization(self, env_ids: torch.Tensor | None = None):
        sample_t = self._visualization_samples.view(1, -1, 1)
        interpolation_curve_positions_w = self._evaluate_curve_w(sample_t)

        if env_ids is None:
            self.interpolation_curve_positions_w[:] = interpolation_curve_positions_w
            self.interpolation_curve_orientations_w[:] = 0.0
            self.interpolation_curve_orientations_w[..., 0] = 1.0
        else:
            self.interpolation_curve_positions_w[env_ids] = interpolation_curve_positions_w[env_ids]
            self.interpolation_curve_orientations_w[env_ids] = 0.0
            self.interpolation_curve_orientations_w[env_ids, :, 0] = 1.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
                self.tracking_pose_visualizer = VisualizationMarkers(self.cfg.tracking_pose_visualizer_cfg)
                self.interpolation_curve_visualizer = VisualizationMarkers(self.cfg.interpolation_curve_visualizer_cfg)
                self.tracked_trajectory_visualizer = VisualizationMarkers(self.cfg.tracked_trajectory_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
            self.tracking_pose_visualizer.set_visibility(True)
            self.interpolation_curve_visualizer.set_visibility(True)
            self.tracked_trajectory_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)
                self.tracking_pose_visualizer.set_visibility(False)
                self.interpolation_curve_visualizer.set_visibility(False)
                self.tracked_trajectory_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        link0_pos_w = self.robot.data.body_pos_w[:, self.link0_idx]
        link0_quat_w = self.robot.data.body_state_w[:, self.link0_idx, 3:7]
        self._append_current_ee_trajectory_points()

        self._update_trajectory_visualization(link0_pos_w, link0_quat_w)

        self.goal_pose_visualizer.visualize(self.target_pose_w[:, :3], self.target_pose_w[:, 3:])
        self.tracking_pose_visualizer.visualize(self.tracking_pose_w[:, :3], self.tracking_pose_w[:, 3:])
        self.interpolation_curve_visualizer.visualize(
            self.interpolation_curve_positions_w.reshape(-1, 3),
            self.interpolation_curve_orientations_w.reshape(-1, 4),
        )
        self.tracked_trajectory_visualizer.visualize(
            self.tracked_trajectory_positions_w.reshape(-1, 3),
            self.tracked_trajectory_orientations_w.reshape(-1, 4),
        )

        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])

    def _update_trajectory_visualization(self, link0_pos_w: torch.Tensor, link0_quat_w: torch.Tensor):
        self.tracked_trajectory_positions_w[:] = self._get_tracked_trajectory_points_w()
        self.tracked_trajectory_orientations_w[:] = 0.0
        self.tracked_trajectory_orientations_w[..., 0] = 1.0

    def _append_current_ee_trajectory_points(self):
        current_trajectory_points = self._get_current_trajectory_points()

        for env_id in range(self.num_envs):
            count = int(self._tracked_trajectory_counts[env_id].item())
            current_pos = current_trajectory_points[env_id]
            if count == 0:
                self.tracked_trajectory_base_positions[env_id, 0] = current_pos
                self._tracked_trajectory_counts[env_id] = 1
                continue

            last_pos = self.tracked_trajectory_base_positions[env_id, count - 1]
            if torch.linalg.norm(current_pos - last_pos).item() < self._trajectory_point_spacing:
                continue

            if count < self._tracked_trajectory_num_points:
                self.tracked_trajectory_base_positions[env_id, count] = current_pos
                self.tracked_trajectory_base_positions[env_id, count + 1 :] = current_pos
                self._tracked_trajectory_counts[env_id] = count + 1
            else:
                self.tracked_trajectory_base_positions[env_id, :-1] = self.tracked_trajectory_base_positions[
                    env_id, 1:
                ].clone()
                self.tracked_trajectory_base_positions[env_id, -1] = current_pos

    def _get_current_trajectory_points(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
        else:
            ee_pos_w = self.robot.data.body_pos_w[env_ids, self.body_idx]

        return ee_pos_w

    def _get_tracked_trajectory_points_w(self) -> torch.Tensor:
        return self.tracked_trajectory_base_positions

    def _convert_base_pose_to_world(
        self,
        pose_base: torch.Tensor,
        base_pos_w: torch.Tensor,
        base_quat_w: torch.Tensor,
        pose_world: torch.Tensor,
    ):
        pose_world[:, :3] = base_pos_w + quat_apply(base_quat_w, pose_base[:, :3])
        pose_world[:, 3:] = quat_mul(base_quat_w, pose_base[:, 3:])

def _build_interpolation_curve_visualizer_cfg():
    trajectory_cfg = POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/Command/interpolation_curve")
    trajectory_cfg.markers = {
        "trajectory": sim_utils.SphereCfg(
            radius=0.0075,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 1.0)),
        )
    }
    return trajectory_cfg


def _build_tracked_trajectory_visualizer_cfg():
    trajectory_cfg = POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/Command/tracked_trajectory")
    trajectory_cfg.markers = {
        "trajectory": sim_utils.SphereCfg(
            radius=0.009,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.3)),
        )
    }
    return trajectory_cfg


def _build_tracking_pose_visualizer_cfg():
    tracking_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/tracking_pose")
    tracking_cfg.markers["frame"].scale = (0.085, 0.085, 0.085)
    return tracking_cfg


def _build_trajectory_visualizer_cfg():
    return _build_interpolation_curve_visualizer_cfg()