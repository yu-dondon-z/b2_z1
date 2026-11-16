"""VD5 startup adaptation without changing the original seam command task."""

from __future__ import annotations

import math

import torch

from ..tcp_frame import quat_mul, tcp_pose_w
from .curve_velocity_command import CurveVelocityCommand
from .vd5_geometry import world_position_to_base
from .vd5_world_seam_command import VD5WorldSeamCommand


def startup_fraction(episode_steps, step_dt, duration):
    """Quintic time ramp with zero speed and acceleration at both endpoints."""
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("Startup duration must be finite and positive.")
    time = (episode_steps * step_dt / duration).clamp(0.0, 1.0)
    return time**3 * (10.0 + time * (-15.0 + 6.0 * time))


def interpolate_quaternion(first, second, fraction):
    """Shortest normalized quaternion interpolation; inputs and output are wxyz."""
    sign = torch.where((first * second).sum(-1, keepdim=True) < 0.0, -1.0, 1.0)
    result = first + fraction.unsqueeze(-1) * (sign * second - first)
    return result / result.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


class VD5StartupCurveCommand(CurveVelocityCommand):
    """Use the original front path controller, smoothly entering its commands."""

    def _update_command(self, env_ids=None):
        super()._update_command(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        fraction = startup_fraction(self._env.episode_length_buf[ids], self._env.step_dt,
                                    self.cfg.vd5_startup_duration_s)
        self.vel_command_b[ids] *= fraction.unsqueeze(-1)


class VD5StartupSeamCommand(VD5WorldSeamCommand):
    """Enter the moving seam target from the measured TCP at episode reset."""

    def __init__(self, cfg, env):
        if not math.isfinite(cfg.vd5_startup_duration_s) or cfg.vd5_startup_duration_s <= 0.0:
            raise ValueError("Startup duration must be finite and positive.")
        super().__init__(cfg, env)
        self._startup_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._startup_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._startup_quat_w[:, 0] = 1.0
        self._startup_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["startup_fraction"] = torch.zeros(self.num_envs, device=self.device)

    def _set_seam_target(self, env_ids):
        super()._set_seam_target(env_ids)
        steps = self.env.episode_length_buf[env_ids]
        capture = (steps == 0) | ~self._startup_initialized[env_ids]
        if capture.any():
            ids = torch.arange(self.num_envs, device=self.device)[env_ids][capture]
            position, quaternion = tcp_pose_w(self.robot)
            self._startup_pos_w[ids] = position[ids]
            self._startup_quat_w[ids] = quaternion[ids]
            self._startup_initialized[ids] = True
        fraction = startup_fraction(steps, self.env.step_dt, self.cfg.vd5_startup_duration_s)
        self.metrics["startup_fraction"][env_ids] = fraction
        position_w = self._startup_pos_w[env_ids] + fraction.unsqueeze(-1) * (
            self._vd5_target_pos_w[env_ids] - self._startup_pos_w[env_ids]
        )
        base_pos = self.robot.data.body_pos_w[env_ids, self.base_idx]
        base_quat = self.robot.data.body_state_w[env_ids, self.base_idx, 3:7]
        inverse = base_quat * base_quat.new_tensor((1.0, -1.0, -1.0, -1.0))
        start_quat_b = quat_mul(inverse, self._startup_quat_w[env_ids])
        self.pose_command[env_ids, :3] = world_position_to_base(position_w, base_pos, base_quat)
        self.pose_command[env_ids, 3:] = interpolate_quaternion(
            start_quat_b, self.pose_target[env_ids, 3:], fraction
        )
        self._vd5_target_pos_w[env_ids] = position_w
        self.pose_target_pos_w[env_ids] = position_w

    def _get_target_quat_w(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        return quat_mul(self.robot.data.body_state_w[ids, self.base_idx, 3:7],
                        self.pose_command[ids, 3:])
