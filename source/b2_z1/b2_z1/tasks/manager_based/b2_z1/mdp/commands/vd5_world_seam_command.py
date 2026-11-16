"""Use the simulated base seam path as an EE target in the moving base frame."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from .pose_command_wbc import UniformPoseWBCCommand
from .vd5_geometry import (
    base_position_to_world,
    blend_target_position,
    sample_polyline_behind_position,
    world_position_to_base,
)


class VD5WorldSeamCommand(UniformPoseWBCCommand):
    """Blend a reachable nominal TCP pose into a ground seam target in base coordinates."""

    def __init__(self, cfg, env):
        self._vd5_ground_blend = 1.0 if cfg.vd5_ground_transition_steps <= 0 else 0.0
        self._vd5_last_ground_curriculum_step = int(env.common_step_counter)
        self._vd5_ground_transition_steps = int(cfg.vd5_ground_transition_steps)
        self.trailing_distance = float(cfg.vd5_seam_trailing_distance)
        if self._vd5_ground_transition_steps < 0:
            raise ValueError("VD5 ground transition steps cannot be negative.")
        if self.trailing_distance <= 0.0:
            raise ValueError("VD5 seam trailing distance must be positive.")
        super().__init__(cfg, env)
        self._vd5_target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        # Resample the nominal transition pose and refresh the seam target at reset.
        super()._resample_command(env_ids)
        if len(env_ids):
            self._set_seam_target(env_ids)

    def _update_command(self):
        super()._update_command()
        self._advance_ground_curriculum()
        self._set_seam_target(slice(None))

    def _advance_ground_curriculum(self):
        if self._vd5_ground_transition_steps == 0 or self._vd5_ground_blend >= 1.0:
            return
        current_step = int(self.env.common_step_counter)
        elapsed = max(0, current_step - self._vd5_last_ground_curriculum_step)
        if elapsed:
            self._vd5_ground_blend = min(
                1.0, self._vd5_ground_blend + elapsed / self._vd5_ground_transition_steps
            )
            self._vd5_last_ground_curriculum_step = current_step

    def _set_seam_target(self, env_ids):
        base_command = self.env.command_manager.get_term("base_velocity")
        seam_points_w = base_command.curve_points_w[env_ids]
        base_pos_w = self.robot.data.body_pos_w[env_ids, self.base_idx]
        base_quat_w = self.robot.data.body_state_w[env_ids, self.base_idx, 3:7]
        seam_target_pos_w = sample_polyline_behind_position(
            seam_points_w, base_pos_w, self.trailing_distance
        )
        seam_target_pos_b = world_position_to_base(seam_target_pos_w, base_pos_w, base_quat_w)
        effective_target_pos_b = blend_target_position(
            self.pose_command[env_ids, :3], seam_target_pos_b, self._vd5_ground_blend
        )
        self.pose_command[env_ids, :3] = effective_target_pos_b
        self.pose_command[env_ids, 3:] = self.pose_target[env_ids, 3:]
        self._vd5_target_pos_w[env_ids] = base_position_to_world(
            effective_target_pos_b, base_pos_w, base_quat_w
        )
        self.pose_target_pos_w[env_ids] = self._vd5_target_pos_w[env_ids]

    def _get_target_pos_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if not hasattr(self, "_vd5_target_pos_w"):
            return super()._get_target_pos_w(env_ids)
        return self._vd5_target_pos_w if env_ids is None else self._vd5_target_pos_w[env_ids]

    def _update_trajectory_visualization(self):
        base_command = self.env.command_manager.get_term("base_velocity")
        path = base_command.curve_points_w
        sample_at = torch.linspace(0.0, path.shape[1] - 1, self._trajectory_num_points, device=self.device)
        left = sample_at.floor().long()
        right = (left + 1).clamp(max=path.shape[1] - 1)
        fraction = (sample_at - left).view(1, -1, 1)
        self.interpolation_curve_positions_w[:] = path[:, left] * (1.0 - fraction) + path[:, right] * fraction
        self.interpolation_curve_orientations_w[:] = self._get_target_quat_w().unsqueeze(1)
        self.tracked_trajectory_positions_w[:] = self._get_tracked_trajectory_points_w()
        self.tracked_trajectory_orientations_w[:] = 0.0
        self.tracked_trajectory_orientations_w[..., 0] = 1.0
