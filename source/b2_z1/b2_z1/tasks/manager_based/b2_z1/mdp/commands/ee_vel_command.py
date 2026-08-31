from __future__ import annotations


from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi
from isaaclab.utils import configclass

import torch

from .command_cfg import UniformVelocityCommandCfg


class EEDerivedVelocityCommand(UniformVelocityCommand):
    """Derives the base velocity from the ee target pose.

    The command is a 3D vector [vx, vy, wz] in the robot base frame. Linear
    velocity is aligned with the direction from the base to the ee target and
    is scaled by the distance to the target. Yaw is computed from the heading
    offset toward the target. Both linear and angular outputs are clamped by
    the current command ranges so curriculum updates can change the maximum
    motion speed over time.
    """

    cfg: EEDerivedVelocityCommandCfg

    def __init__(self, cfg, env, **kwargs):
        super().__init__(cfg, env, **kwargs)
        self.env = env
        self.base_idx = self.robot.find_bodies(cfg.base_name)[0][0]
        self.ee_body_idx = self.robot.find_bodies(cfg.ee_body_name)[0][0]
        self.metrics["distance_to_target"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "EEDerivedVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tArrival threshold: {self.cfg.arrival_threshold}\n"
        msg += f"\tTarget command: {self.cfg.ee_command_name}\n"
        msg += f"\tCurrent linear range: {self.cfg.ranges.lin_vel_x}, {self.cfg.ranges.lin_vel_y}\n"
        msg += f"\tCurrent yaw range: {self.cfg.ranges.ang_vel_z}"
        return msg

    def _update_metrics(self):
        super()._update_metrics()
        ee_cmd_term = self.env.command_manager.get_term(self.cfg.ee_command_name)
        tracking_pos_w = ee_cmd_term.tracking_pose_w[:, :3]  # current trajectory target point
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_idx]
        self.metrics["distance_to_target"] = torch.norm(
            tracking_pos_w[:, :2] - ee_pos_w[:, :2], dim=1
        )

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return

        super()._resample_command(env_ids)

    def _update_command(self):
        ee_cmd_term = self.env.command_manager.get_term(self.cfg.ee_command_name)
        # Rear-facing tool control must follow the forward crack tangent, not
        # the vector from B2 to a target that is physically behind the base.
        # Vision/synthetic rear-path command terms expose that tangent-derived
        # velocity explicitly while preserving the original 3-D observation.
        path_velocity_b = getattr(ee_cmd_term, "base_velocity_command_b", None)
        if path_velocity_b is not None:
            self.vel_command_b[:] = path_velocity_b
            standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
            if len(standing_env_ids) > 0:
                self.vel_command_b[standing_env_ids, :] = 0.0
            return

        # tracking_pose_w[:, :3]: current interpolated trajectory point in world frame
        tracking_pos_w = ee_cmd_term.tracking_pose_w[:, :3]

        # Current ee body position in world frame (e.g. dummy_target)
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_idx]

        # Base state
        base_pos_w = self.robot.data.body_pos_w[:, self.base_idx]
        base_quat_w = self.robot.data.body_state_w[:, self.base_idx, 3:7]

        # Direction from ee (dummy_target) to the current trajectory tracking point, xy only
        delta_w = tracking_pos_w[:, :2] - ee_pos_w[:, :2]
        distance = torch.norm(delta_w, dim=1, keepdim=True)
        safe_distance = distance.clamp_min(1e-6)
        direction_w = delta_w / safe_distance  # unit vector in world frame

        # Speed: zero when within arrival_threshold; ramps up linearly beyond it.
        # speed_scale = clamp((distance - threshold) / threshold, 0, 1)
        max_lin_vel_x = max(abs(self.cfg.ranges.lin_vel_x[0]), abs(self.cfg.ranges.lin_vel_x[1]))
        max_lin_vel_y = max(abs(self.cfg.ranges.lin_vel_y[0]), abs(self.cfg.ranges.lin_vel_y[1]))
        max_lin_speed = max(max_lin_vel_x, max_lin_vel_y)
        arrival_threshold = max(self.cfg.arrival_threshold, 1e-6)
        dist_scalar = distance.squeeze(-1)
        speed_scale = ((dist_scalar - arrival_threshold) / arrival_threshold).clamp(0.0, 1.0)
        speed = max_lin_speed * speed_scale

        # Project direction into base frame for vx/vy
        direction_w_3d = torch.zeros(self.num_envs, 3, device=self.device)
        direction_w_3d[:, :2] = direction_w
        direction_base = quat_apply_inverse(base_quat_w, direction_w_3d)

        self.vel_command_b[:, 0] = (direction_base[:, 0] * speed).clamp(
            min=self.cfg.ranges.lin_vel_x[0], max=self.cfg.ranges.lin_vel_x[1]
        )
        self.vel_command_b[:, 1] = (direction_base[:, 1] * speed).clamp(
            min=self.cfg.ranges.lin_vel_y[0], max=self.cfg.ranges.lin_vel_y[1]
        )

        # Yaw: also zero within arrival_threshold; ramps with the same speed_scale
        base_heading_w = self.robot.data.heading_w  # (num_envs,)
        desired_heading_w = torch.atan2(direction_w[:, 1], direction_w[:, 0])
        heading_error = wrap_to_pi(desired_heading_w - base_heading_w)
        yaw_cmd = self.cfg.heading_gain * heading_error * speed_scale
        self.vel_command_b[:, 2] = yaw_cmd.clamp(
            min=self.cfg.ranges.ang_vel_z[0], max=self.cfg.ranges.ang_vel_z[1]
        )

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if len(standing_env_ids) > 0:
            self.vel_command_b[standing_env_ids, :] = 0.0


@configclass
class EEDerivedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the ee-target-derived base velocity command generator.

    Velocity direction and magnitude are derived from the vector between the
    current ee body (e.g. dummy_target) and the live trajectory tracking point
    of the ee command term. Yaw is computed from the offset between the base
    heading and the world-frame direction of that vector.
    """

    class_type: type = EEDerivedVelocityCommand

    ee_command_name: str = "ee_target"
    """Name of the command term that provides the ee target pose."""

    # ee_body_name: str = "dummy_target" # --> ee tracking -- > 0.1
    ee_body_name: str = "base_link" # --> base tracking --> 0.6
    """Name of the robot body used as the current end-effector position reference."""

    base_name: str = "base_link"
    """Name of the robot base body, used to read base state."""

    arrival_threshold: float = 0.6 # 0.1
    """Distance (m) from ee body to tracking point below which speed tapers to zero."""

    heading_gain: float = 1.0
    """Proportional gain converting heading error (rad) into yaw-rate command."""
