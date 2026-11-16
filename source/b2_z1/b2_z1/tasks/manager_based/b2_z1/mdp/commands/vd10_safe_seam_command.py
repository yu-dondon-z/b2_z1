"""VD10 high-clearance startup, returning to VD5's original seam attitude."""

import math

import torch
from isaaclab.utils.math import quat_from_euler_xyz

from ..tcp_frame import quat_apply
from ..vd10_geometry import cylinder_tip_height_above_plane
from .vd5_geometry import world_position_to_base
from .vd5_startup_command import VD5StartupSeamCommand


SAFE_PITCH_RAD = 0.30
SEAM_PITCH_RAD = math.pi / 2 - 0.45
WORKING_SURFACE_CLEARANCE_M = 0.005


class VD10SafeSeamCommand(VD5StartupSeamCommand):
    """Blend only the temporary pitch with the existing ground-seam curriculum.

    World XY still follows the parent's seam. At the final working angle,
    raise the end-face centre so the actual pipe rim clears the ground by 5 mm.
    """

    vd10_startup_profile = {
        "version": 2,
        "safe_pitch_rad": SAFE_PITCH_RAD,
        "seam_pitch_rad": SEAM_PITCH_RAD,
        "working_surface_clearance_m": WORKING_SURFACE_CLEARANCE_M,
        "tube_length_m": 1.1,
        "tube_outer_radius_m": 0.028,
        "reset_commands_after_forward": True,
    }

    def _set_seam_target(self, env_ids):
        target = self.pose_target[env_ids]
        pitch = target[:, 0].new_full(
            (target.shape[0],), SAFE_PITCH_RAD + self._vd5_ground_blend * (SEAM_PITCH_RAD - SAFE_PITCH_RAD)
        )
        self.pose_target[env_ids, 3:] = quat_from_euler_xyz(
            torch.zeros_like(pitch), pitch, torch.full_like(pitch, math.pi)
        )
        super()._set_seam_target(env_ids)
        target_quat_w = self._get_target_quat_w(env_ids)
        axis_w = quat_apply(target_quat_w, torch.zeros_like(target[:, :3])
                            + target.new_tensor((1.0, 0.0, 0.0)))
        centre_height = cylinder_tip_height_above_plane(axis_w, WORKING_SURFACE_CLEARANCE_M)
        # Preserve the measured TCP exactly at the first command of a reset.
        offset = centre_height * self._vd5_ground_blend * self.metrics["startup_fraction"][env_ids]
        position_w = self._vd5_target_pos_w[env_ids].clone()
        position_w[:, 2] += offset
        self._vd5_target_pos_w[env_ids] = position_w
        self.pose_target_pos_w[env_ids] = position_w
        self.pose_command[env_ids, :3] = world_position_to_base(
            position_w, self.robot.data.body_pos_w[env_ids, self.base_idx],
            self.robot.data.body_state_w[env_ids, self.base_idx, 3:7]
        )
