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
        self._curve_control_points = torch.zeros(self.num_envs, 2, 3, device=self.device)

        self._trajectory_num_points = max(2, getattr(cfg, "trajectory_num_points", 10))
        self._trajectory_samples = torch.linspace(0.0, 1.0, self._trajectory_num_points, device=self.device)
        self._command_duration = torch.ones(self.num_envs, device=self.device)
        self._trajectory_point_spacing = getattr(cfg, "trajectory_point_spacing", 0.02)
        self.interpolation_curve_positions_w = torch.zeros(
            self.num_envs, self._trajectory_num_points, 3, device=self.device
        )
        self.interpolation_curve_orientations_w = torch.zeros(
            self.num_envs, self._trajectory_num_points, 4, device=self.device
        )
        self.interpolation_curve_orientations_w[..., 0] = 1.0
        self.tracked_trajectory_base_positions = torch.zeros(
            self.num_envs, self._trajectory_num_points, 3, device=self.device
        )
        self.tracked_trajectory_positions_w = torch.zeros(
            self.num_envs, self._trajectory_num_points, 3, device=self.device
        )
        self.tracked_trajectory_orientations_w = torch.zeros(
            self.num_envs, self._trajectory_num_points, 4, device=self.device
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
        self._sample_curve_control_points(env_ids)
        self._command_duration[env_ids] = self.time_left[env_ids].clamp_min(self.env.step_dt)
        self._update_command_from_world_pose(
            self._get_curve_start_pos_w(env_ids), self._get_target_quat_w(env_ids), self.pose_command[env_ids], env_ids
        )
        self._reset_tracked_trajectory(env_ids)

    def _update_command(self):
        interpolation_progress = 1.0 - self.time_left / self._command_duration.clamp_min(self.env.step_dt)
        interpolation_progress = interpolation_progress.clamp_(0.0, 1.0).unsqueeze(-1)

        self._update_command_from_world_pose(
            self._evaluate_curve(interpolation_progress), self._get_target_quat_w(), self.pose_command
        )

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

        start_pos_link0 = self._project_world_point_to_link0_mixed_frame(ee_pos_w, link0_pos_w, link0_quat_w)
        self.pose_start_pos_w[env_ids] = ee_pos_w
        self.pose_start[env_ids, :2] = start_pos_link0[:, :2]
        self.pose_start[env_ids, 2] = ee_pos_w[:, 2]
        self.pose_start[env_ids, 3:] = quat_mul(quat_conjugate(link0_quat_w), ee_quat_w)

    def _sample_curve_control_points(self, env_ids: torch.Tensor):
        start_pos = self.pose_start[env_ids, :3]
        target_pos = self.pose_target[env_ids, :3]
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

        random_offsets = (torch.rand(len(env_ids), 2, 2, device=self.device) * 2.0) - 1.0
        curve_offset_scale = getattr(self.cfg, "trajectory_curve_offset_scale", 0.1)
        offset_scale = curve_offset_scale * length.unsqueeze(1)
        first_offset = offset_scale[:, 0] * (
            random_offsets[:, 0, 0:1] * normal_1 + 0.5 * random_offsets[:, 0, 1:2] * normal_2
        )
        second_offset = offset_scale[:, 0] * (
            random_offsets[:, 1, 0:1] * normal_1 + 0.5 * random_offsets[:, 1, 1:2] * normal_2
        )

        self._curve_control_points[env_ids, 0] = start_pos + 0.33 * delta + first_offset
        self._curve_control_points[env_ids, 1] = start_pos + 0.66 * delta + second_offset

    def _reset_tracked_trajectory(self, env_ids: torch.Tensor):
        current_trajectory_point_mixed = self._get_current_trajectory_points(env_ids)
        self.tracked_trajectory_base_positions[env_ids] = current_trajectory_point_mixed.unsqueeze(1)
        self.tracked_trajectory_positions_w[env_ids] = 0.0
        self.tracked_trajectory_orientations_w[env_ids] = 0.0
        self.tracked_trajectory_orientations_w[env_ids, :, 0] = 1.0
        self._tracked_trajectory_counts[env_ids] = 1

    def _evaluate_curve(self, progress: torch.Tensor) -> torch.Tensor:
        if progress.ndim == 1:
            progress = progress.unsqueeze(-1)

        is_trajectory_sampling = progress.ndim == 3
        one_minus_t = 1.0 - progress
        if is_trajectory_sampling:
            start_pos = self._get_curve_start_pos_w().unsqueeze(1)
            control_point_1 = self._get_curve_control_points_w()[:, 0].unsqueeze(1)
            control_point_2 = self._get_curve_control_points_w()[:, 1].unsqueeze(1)
            target_pos = self._get_target_pos_w().unsqueeze(1)
        else:
            start_pos = self._get_curve_start_pos_w()
            control_point_1 = self._get_curve_control_points_w()[:, 0]
            control_point_2 = self._get_curve_control_points_w()[:, 1]
            target_pos = self._get_target_pos_w()

        return (
            (one_minus_t**3) * start_pos
            + 3.0 * (one_minus_t**2) * progress * control_point_1
            + 3.0 * one_minus_t * (progress**2) * control_point_2
            + (progress**3) * target_pos
        )

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

    def _get_curve_control_points_w(self) -> torch.Tensor:
        link0_pos_w = self.robot.data.body_pos_w[:, self.link0_idx]
        link0_quat_w = self.robot.data.body_state_w[:, self.link0_idx, 3:7]

        control_points_xy = torch.cat(
            (self._curve_control_points[..., :2], torch.zeros_like(self._curve_control_points[..., 2:3])), dim=-1
        )
        control_points_w = link0_pos_w.unsqueeze(1) + quat_apply(
            link0_quat_w.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4),
            control_points_xy.reshape(-1, 3),
        ).reshape(-1, 2, 3)
        control_points_w[..., 2] = self._curve_control_points[..., 2]
        return control_points_w

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
        base_pos_w = self.robot.data.body_pos_w[:, self.base_idx]
        base_quat_w = self.robot.data.body_state_w[:, self.base_idx, 3:7]
        self._append_current_ee_trajectory_points()

        self.target_pose_w[:, :3] = self._get_target_pos_w()
        self.target_pose_w[:, 3:] = self._get_target_quat_w()
        self._convert_base_pose_to_world(self.pose_command, base_pos_w, base_quat_w, self.tracking_pose_w)
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
        sample_t = self._trajectory_samples.view(1, -1, 1)

        self.interpolation_curve_positions_w[:] = self._evaluate_curve(sample_t)
        self.interpolation_curve_orientations_w[:] = quat_mul(
            link0_quat_w.unsqueeze(1).expand(-1, self._trajectory_num_points, -1).reshape(-1, 4),
            self.pose_target[:, 3:].unsqueeze(1).expand(-1, self._trajectory_num_points, -1).reshape(-1, 4),
        ).reshape(-1, self._trajectory_num_points, 4)
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

            if count < self._trajectory_num_points:
                self.tracked_trajectory_base_positions[env_id, count] = current_pos
                self.tracked_trajectory_base_positions[env_id, count + 1 :] = current_pos
                self._tracked_trajectory_counts[env_id] = count + 1
            else:
                self.tracked_trajectory_base_positions[env_id, :-1] = self.tracked_trajectory_base_positions[
                    env_id, 1:
                ].clone()
                self.tracked_trajectory_base_positions[env_id, -1] = current_pos

    def _project_world_point_to_link0_mixed_frame(
        self,
        point_w: torch.Tensor,
        link0_pos_w: torch.Tensor,
        link0_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        delta_xy_w = point_w[:, :2] - link0_pos_w[:, :2]

        local_x = torch.zeros((point_w.shape[0], 3), device=self.device)
        local_y = torch.zeros((point_w.shape[0], 3), device=self.device)
        local_x[:, 0] = 1.0
        local_y[:, 1] = 1.0

        basis_x_w = quat_apply(link0_quat_w, local_x)
        basis_y_w = quat_apply(link0_quat_w, local_y)

        transform_xy = torch.stack(
            (
                torch.stack((basis_x_w[:, 0], basis_y_w[:, 0]), dim=-1),
                torch.stack((basis_x_w[:, 1], basis_y_w[:, 1]), dim=-1),
            ),
            dim=1,
        )

        local_xy = torch.linalg.solve(transform_xy, delta_xy_w.unsqueeze(-1)).squeeze(-1)

        projected_point = torch.empty_like(point_w)
        projected_point[:, :2] = local_xy
        projected_point[:, 2] = point_w[:, 2]
        return projected_point

    def _get_current_trajectory_points(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            link0_pos_w = self.robot.data.body_pos_w[:, self.link0_idx]
            link0_quat_w = self.robot.data.body_state_w[:, self.link0_idx, 3:7]
            ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
        else:
            link0_pos_w = self.robot.data.body_pos_w[env_ids, self.link0_idx]
            link0_quat_w = self.robot.data.body_state_w[env_ids, self.link0_idx, 3:7]
            ee_pos_w = self.robot.data.body_pos_w[env_ids, self.body_idx]

        return self._project_world_point_to_link0_mixed_frame(ee_pos_w, link0_pos_w, link0_quat_w)

    def _get_tracked_trajectory_points_w(self) -> torch.Tensor:
        link0_pos_w = self.robot.data.body_pos_w[:, self.link0_idx]
        link0_quat_w = self.robot.data.body_state_w[:, self.link0_idx, 3:7]

        tracked_points_xy = torch.cat(
            (
                self.tracked_trajectory_base_positions[..., :2],
                torch.zeros_like(self.tracked_trajectory_base_positions[..., 2:3]),
            ),
            dim=-1,
        )
        tracked_points_w = link0_pos_w.unsqueeze(1) + quat_apply(
            link0_quat_w.unsqueeze(1).expand(-1, self._trajectory_num_points, -1).reshape(-1, 4),
            tracked_points_xy.reshape(-1, 3),
        ).reshape(-1, self._trajectory_num_points, 3)
        tracked_points_w[..., 2] = self.tracked_trajectory_base_positions[..., 2]
        return tracked_points_w

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