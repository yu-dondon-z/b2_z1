# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed forward speed and curve-derived yaw commands, independent of EE commands."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_mul,
    yaw_quat,
)

from .curve_geometry import (
    cubic_bezier,
    curve_tracking_errors,
    limit_ground_curve_curvature,
    multi_turn_path,
    perceive_ground_curve,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .command_cfg import CurveVelocityCommandCfg


class CurveVelocityCommand(UniformVelocityCommand):
    """Follow independent, world-fixed ground paths through an ideal base camera.

    The policy command remains ``[forward_speed, 0, yaw_rate]``. Ground paths use
    Bezier interpolation or multiple circular turns with straight gaps, independently of the EE
    command. Camera intrinsics and the mount pose are configurable;
    image extraction can later replace the ideal pixel observations.
    """

    cfg: CurveVelocityCommandCfg

    def __init__(self, cfg: CurveVelocityCommandCfg, env: ManagerBasedRLEnv):
        if cfg.forward_speed <= 0.0 or cfg.lookahead_distance <= 0.0:
            raise ValueError("forward_speed and lookahead_distance must be positive.")
        if cfg.trajectory_num_points < 3:
            raise ValueError("At least three curve samples are required.")
        if min(cfg.ground_curve_width, cfg.ground_curve_thickness) <= 0.0:
            raise ValueError("Ground curve width and thickness must be positive.")
        if cfg.camera_depth_range[0] <= 0.0 or cfg.camera_depth_range[1] <= cfg.camera_depth_range[0]:
            raise ValueError("camera_depth_range must satisfy 0 < near < far.")
        if min(cfg.camera_intrinsics[:2]) <= 0.0 or min(cfg.camera_image_size) <= 0:
            raise ValueError("Camera focal lengths and image dimensions must be positive.")
        if cfg.ranges.ang_vel_z[0] >= 0.0 or cfg.ranges.ang_vel_z[1] <= 0.0:
            raise ValueError("Yaw-rate limits must allow both left and right turns.")
        if not 0.0 <= cfg.trajectory_curve_offset_scale <= cfg.limit_trajectory_curve_offset_scale < float("inf"):
            raise ValueError("Curve offset scales must satisfy 0 <= initial <= limit < infinity.")
        if cfg.path_type not in ("bezier", "turn"):
            raise ValueError("path_type must be 'bezier' or 'turn'.")
        if (
            len(cfg.turn_count_range) != 2
            or any(not isinstance(count, int) or isinstance(count, bool) for count in cfg.turn_count_range)
            or not 1 <= cfg.turn_count_range[0] <= cfg.turn_count_range[1] <= 3
        ):
            raise ValueError("turn_count_range must contain integers satisfying 1 <= initial <= maximum <= 3.")
        if len(cfg.turn_angle_range) != 2 or not 0.0 <= cfg.turn_angle_range[0] <= cfg.turn_angle_range[1] <= math.pi:
            raise ValueError("turn_angle_range must satisfy 0 <= min <= max <= pi radians.")
        if not 0.0 < cfg.turn_curvature < float("inf"):
            raise ValueError("turn_curvature must be positive and finite.")
        if not 0.0 <= cfg.turn_start_distance < float("inf"):
            raise ValueError("turn_start_distance must be nonnegative and finite.")
        if not 0.0 <= cfg.turn_gap_distance < float("inf"):
            raise ValueError("turn_gap_distance must be nonnegative and finite.")
        if cfg.max_curvature is not None:
            if not 0.0 <= cfg.max_curvature < float("inf"):
                raise ValueError("max_curvature must be nonnegative and finite, or None.")
            if cfg.forward_speed * cfg.max_curvature > min(-cfg.ranges.ang_vel_z[0], cfg.ranges.ang_vel_z[1]):
                raise ValueError("forward_speed * max_curvature must fit within both yaw-rate limits.")
        if cfg.path_type == "turn":
            curvature = (
                min(cfg.turn_curvature, cfg.max_curvature) if cfg.max_curvature is not None else cfg.turn_curvature
            )
            if cfg.forward_speed * curvature > min(-cfg.ranges.ang_vel_z[0], cfg.ranges.ang_vel_z[1]):
                raise ValueError("forward_speed * turn curvature must fit within both yaw-rate limits.")
        if not cfg.curriculum_enabled:
            cfg.trajectory_curve_offset_scale = cfg.limit_trajectory_curve_offset_scale
        super().__init__(cfg, env)

        self._last_curve_curriculum_step = 0
        self.base_idx = self.robot.find_bodies(cfg.base_name)[0][0]
        self.curve_control_points_w = (
            torch.zeros(self.num_envs, 4, 3, device=self.device) if cfg.path_type == "bezier" else None
        )
        self.curve_points_w = torch.zeros(self.num_envs, cfg.trajectory_num_points, 3, device=self.device)
        self.curve_turn_angle = torch.zeros(self.num_envs, device=self.device)
        self.curve_turn_angles = torch.zeros(self.num_envs, cfg.turn_count_range[1], device=self.device)
        self.curve_turn_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.curve_path_length = torch.zeros(self.num_envs, device=self.device)
        self.camera_pixels = torch.zeros(self.num_envs, cfg.trajectory_num_points, 2, device=self.device)
        self.camera_visible = torch.zeros(
            self.num_envs, cfg.trajectory_num_points, device=self.device, dtype=torch.bool
        )
        self.perceived_curve_points_w = torch.zeros_like(self.curve_points_w)
        self.lateral_error = torch.zeros(self.num_envs, device=self.device)
        self.heading_error = torch.zeros_like(self.lateral_error)
        self.curvature = torch.zeros_like(self.lateral_error)
        self.path_visible = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._samples = torch.linspace(0.0, 1.0, cfg.trajectory_num_points, device=self.device).view(1, -1, 1)
        self._ground_height_w = env.scene.env_origins[:, 2] + cfg.ground_height
        self._camera_offset = torch.tensor(cfg.camera_position_b, device=self.device).expand(self.num_envs, -1)
        camera_rpy = torch.tensor(cfg.camera_rpy_b, device=self.device).expand(self.num_envs, -1)
        self._camera_quat_b = quat_from_euler_xyz(*camera_rpy.unbind(dim=-1))
        self._up = torch.zeros(self.num_envs, 3, device=self.device)
        self._up[:, 2] = 1.0
        for name in ("cross_track_error", "heading_error", "path_visible", "turn_angle", "turn_count", "path_length"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            f"CurveVelocityCommand: path={self.cfg.path_type}, speed={self.cfg.forward_speed} m/s, "
            f"yaw limits={self.cfg.ranges.ang_vel_z} rad/s, "
            f"max curvature={self.cfg.max_curvature} 1/m, "
            f"lookahead={self.cfg.lookahead_distance} m, ideal camera projection"
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None or isinstance(env_ids, slice):
            all_ids = torch.arange(self.num_envs, device=self.device)
            env_ids = all_ids if env_ids is None else all_ids[env_ids]
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        count = len(env_ids)
        base_pos = self.robot.data.body_pos_w[env_ids, self.base_idx]
        base_quat = self.robot.data.body_quat_w[env_ids, self.base_idx]
        rotation = yaw_quat(base_quat)

        # Cover the full command duration plus a forward perception margin.
        length = self.cfg.forward_speed * self.time_left[env_ids] + 2.0 * self.cfg.lookahead_distance
        if self.cfg.path_type == "turn":
            difficulty = (
                self.cfg.trajectory_curve_offset_scale / self.cfg.limit_trajectory_curve_offset_scale
                if self.cfg.limit_trajectory_curve_offset_scale > 0.0
                else 0.0
            )
            initial_turns, maximum_turns = self.cfg.turn_count_range
            num_turns = initial_turns + min(
                int(difficulty * (maximum_turns - initial_turns + 1)), maximum_turns - initial_turns
            )
            random = torch.rand(count, num_turns + 1, device=self.device)
            min_angle, max_angle = self.cfg.turn_angle_range
            magnitudes = (min_angle + (max_angle - min_angle) * random[:, :num_turns]) * difficulty
            direction = torch.where(random[:, -1] < 0.5, -1.0, 1.0)
            # Alternate between a sideward heading and the initial forward heading.
            # Paired corners cancel, keeping the path from looping back on itself.
            sideward = torch.arange(num_turns, device=self.device) % 2 == 0
            headings = magnitudes * direction.unsqueeze(-1) * sideward
            angles = torch.diff(torch.cat((torch.zeros(count, 1, device=self.device), headings), dim=-1), dim=-1)
            curvature = self.cfg.turn_curvature
            if self.cfg.max_curvature is not None:
                curvature = min(curvature, self.cfg.max_curvature)
            local_points, actual_angles = multi_turn_path(
                length,
                angles,
                curvature,
                self.cfg.turn_start_distance,
                self.cfg.turn_gap_distance,
                self.cfg.lookahead_distance,
                self._samples,
            )
            world_points = quat_apply(
                rotation.unsqueeze(1).expand(-1, self.cfg.trajectory_num_points, -1), local_points
            )
            world_points += base_pos.unsqueeze(1)
            world_points[:, :, 2] = self._ground_height_w[env_ids].unsqueeze(-1)
            self.curve_points_w[env_ids] = world_points
            self.curve_turn_angles[env_ids] = 0.0
            self.curve_turn_angles[env_ids, :num_turns] = actual_angles
            self.curve_turn_angle[env_ids] = actual_angles.sum(dim=-1)
            self.curve_turn_count[env_ids] = (actual_angles.abs() > 1e-6).sum(dim=-1)
        else:
            local_points = torch.zeros(count, 4, 3, device=self.device)
            local_points[:, :, 0] = length.unsqueeze(-1) * torch.tensor(
                (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0), device=self.device
            )
            offsets = 2.0 * torch.rand(count, 3, device=self.device) - 1.0
            local_points[:, 1:, 1] = offsets * self.cfg.trajectory_curve_offset_scale * length.unsqueeze(-1)
            if self.cfg.max_curvature is not None:
                local_points = limit_ground_curve_curvature(local_points, self.cfg.max_curvature)
            world_points = quat_apply(rotation.unsqueeze(1).expand(-1, 4, -1), local_points) + base_pos.unsqueeze(1)
            world_points[:, :, 2] = self._ground_height_w[env_ids].unsqueeze(-1)
            self.curve_control_points_w[env_ids] = world_points
            self.curve_points_w[env_ids] = cubic_bezier(*(world_points[:, i : i + 1] for i in range(4)), self._samples)
        self.curve_path_length[env_ids] = length
        self.metrics["turn_angle"][env_ids] = self.curve_turn_angles[env_ids].abs().amax(dim=-1)
        self.metrics["turn_count"][env_ids] = self.curve_turn_count[env_ids].float()
        self.metrics["path_length"][env_ids] = length
        self.is_standing_env[env_ids] = False
        self.is_heading_env[env_ids] = True
        self._update_command(env_ids)

    def _update_command(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        base_pos = self.robot.data.body_pos_w[env_ids, self.base_idx]
        base_quat = self.robot.data.body_quat_w[env_ids, self.base_idx]
        camera_pos = base_pos + quat_apply(base_quat, self._camera_offset[env_ids])
        camera_quat = quat_mul(base_quat, self._camera_quat_b[env_ids])
        points_c = quat_apply_inverse(
            camera_quat.unsqueeze(1).expand(-1, self.cfg.trajectory_num_points, -1),
            self.curve_points_w[env_ids] - camera_pos.unsqueeze(1),
        )
        recovered_c, pixels, visible = perceive_ground_curve(
            points_c,
            quat_apply_inverse(camera_quat, self._up[env_ids]),
            camera_pos[:, 2] - self._ground_height_w[env_ids],
            self.cfg.camera_intrinsics,
            self.cfg.camera_image_size,
            self.cfg.camera_depth_range,
        )
        recovered_w = camera_pos.unsqueeze(1) + quat_apply(
            camera_quat.unsqueeze(1).expand(-1, self.cfg.trajectory_num_points, -1), recovered_c
        )
        horizontal_quat = yaw_quat(base_quat)
        points_h = quat_apply_inverse(
            horizontal_quat.unsqueeze(1).expand(-1, self.cfg.trajectory_num_points, -1),
            recovered_w - base_pos.unsqueeze(1),
        )
        lateral, heading, curvature, path_visible = curve_tracking_errors(
            points_h, visible, self.cfg.lookahead_distance
        )
        yaw_rate = self.cfg.heading_control_stiffness * heading + self.cfg.forward_speed * curvature
        self.vel_command_b[env_ids, 0] = self.cfg.forward_speed
        self.vel_command_b[env_ids, 1] = 0.0
        # With no visible segment, retain fixed speed and request straight travel.
        self.vel_command_b[env_ids, 2] = torch.where(path_visible, yaw_rate.clamp(*self.cfg.ranges.ang_vel_z), 0.0)
        base_yaw = 2.0 * torch.atan2(horizontal_quat[:, 3], horizontal_quat[:, 0])
        self.heading_target[env_ids] = torch.atan2(torch.sin(base_yaw + heading), torch.cos(base_yaw + heading))
        self.camera_pixels[env_ids] = pixels
        self.camera_visible[env_ids] = visible
        self.perceived_curve_points_w[env_ids] = recovered_w
        self.lateral_error[env_ids] = lateral
        self.heading_error[env_ids] = heading
        self.curvature[env_ids] = curvature
        self.path_visible[env_ids] = path_visible
        self.metrics["cross_track_error"][env_ids] = lateral.abs()
        self.metrics["heading_error"][env_ids] = heading.abs()
        self.metrics["path_visible"][env_ids] = path_visible.float()

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis and not hasattr(self, "ground_curve_visualizer"):
            self.ground_curve_visualizer = VisualizationMarkers(self.cfg.ground_curve_visualizer_cfg)
            self.heading_visualizer = VisualizationMarkers(self.cfg.heading_visualizer_cfg)
        if hasattr(self, "ground_curve_visualizer"):
            self.ground_curve_visualizer.set_visibility(debug_vis)
            self.heading_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        super()._debug_vis_callback(event)
        self._visualize_ground_curve()
        zeros = torch.zeros_like(self.heading_target)
        self.heading_visualizer.visualize(
            self.robot.data.body_pos_w[:, self.base_idx],
            quat_from_euler_xyz(zeros, zeros, self.heading_target),
        )

    def _visualize_ground_curve(self):
        """Draw overlapping, collision-free boxes as a continuous line on the ground."""
        start = self.curve_points_w[:, :-1]
        end = self.curve_points_w[:, 1:]
        segment = end - start
        centers = 0.5 * (start + end)
        centers[..., 2] += self.cfg.ground_curve_z_offset
        yaw = torch.atan2(segment[..., 1], segment[..., 0]).reshape(-1)
        zeros = torch.zeros_like(yaw)
        orientations = quat_from_euler_xyz(zeros, zeros, yaw)
        scales = torch.empty_like(centers)
        # A small overlap closes gaps where neighboring segments change direction.
        scales[..., 0] = torch.linalg.vector_norm(segment[..., :2], dim=-1) + self.cfg.ground_curve_thickness
        scales[..., 1] = self.cfg.ground_curve_width
        scales[..., 2] = self.cfg.ground_curve_thickness
        self.ground_curve_visualizer.visualize(centers.reshape(-1, 3), orientations, scales.reshape(-1, 3))
