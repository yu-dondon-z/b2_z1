"""VD3 arm regularization in the commanded, full robot-base frame.

The terms are stateless: no numerical command derivative or episode history is
retained. A rigid TCP/base motion and an intentionally moving TCP target do not
incur a velocity error. All returned values are nonnegative penalties, intended
for negative reward weights.
"""

from __future__ import annotations

import torch

from .tcp_frame import (
    TCP_POS_IN_LINK06, TCP_QUAT_IN_LINK06, link_origin_velocity_w, link_pose_w,
    quat_apply, quat_mul, tcp_pose_w, tcp_velocity_w,
)


def _quat_apply_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate a vector into a unit, scalar-first quaternion's local frame."""
    imaginary = -quaternion[..., 1:]
    cross = 2.0 * torch.linalg.cross(imaginary, vector, dim=-1)
    return vector + quaternion[..., :1] * cross + torch.linalg.cross(imaginary, cross, dim=-1)


def relative_twist_in_base(
    base_quaternion_w: torch.Tensor,
    base_position_w: torch.Tensor,
    tcp_position_w: torch.Tensor,
    base_linear_velocity_w: torch.Tensor,
    base_angular_velocity_w: torch.Tensor,
    tcp_linear_velocity_w: torch.Tensor,
    tcp_angular_velocity_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the TCP's base-coordinate position derivative and relative omega.

    Positions and linear velocities must refer to the same link origins, rather
    than mixing link positions with center-of-mass velocities. The cross-product
    term removes velocity induced by rotating the offset from base to TCP.
    """
    offset_w = tcp_position_w - base_position_w
    linear_w = tcp_linear_velocity_w - base_linear_velocity_w - torch.linalg.cross(
        base_angular_velocity_w, offset_w, dim=-1
    )
    angular_w = tcp_angular_velocity_w - base_angular_velocity_w
    return _quat_apply_inverse(base_quaternion_w, linear_w), _quat_apply_inverse(
        base_quaternion_w, angular_w
    )


def cubic_bezier_velocity(
    start: torch.Tensor,
    control_one: torch.Tensor,
    control_two: torch.Tensor,
    target: torch.Tensor,
    progress: torch.Tensor,
    duration: torch.Tensor,
) -> torch.Tensor:
    """Analytic base-frame velocity for the command generator's cubic curve.

    ``progress`` is unclipped normalized time. Its endpoint values use one-sided
    curve derivatives, while times outside [0, 1] return zero. Invalid/nonpositive
    durations also return zero. Inputs ending in XYZ support batch dimensions.
    """
    progress = torch.as_tensor(progress, dtype=start.dtype, device=start.device)
    duration = torch.as_tensor(duration, dtype=start.dtype, device=start.device)
    if progress.ndim == start.ndim - 1:
        progress = progress.unsqueeze(-1)
    if duration.ndim == start.ndim - 1:
        duration = duration.unsqueeze(-1)
    active = (progress >= 0.0) & (progress <= 1.0) & (duration > 0.0)
    t = progress.clamp(0.0, 1.0)
    derivative = 3.0 * (
        (1.0 - t).square() * (control_one - start)
        + 2.0 * (1.0 - t) * t * (control_two - control_one)
        + t.square() * (target - control_two)
    )
    safe_duration = duration.clamp_min(torch.finfo(start.dtype).eps)
    return torch.where(active, derivative / safe_duration, torch.zeros_like(derivative))


def command_linear_velocity_b(command, step_dt: float) -> torch.Tensor:
    """Differentiate the same Bézier sample currently exposed to the actor.

    Rewards run before the next command update, so both pose and analytic target
    velocity here correspond to the command used for the just-completed action.
    A resampled path is differentiated on its own curve; the old/new endpoints
    are never subtracted across a resample or an episode reset.
    """
    duration = command._command_duration.clamp_min(step_dt)
    progress = 1.0 - command.time_left / duration
    return cubic_bezier_velocity(
        command.pose_start[:, :3],
        command._curve_control_points[:, 0],
        command._curve_control_points[:, 1],
        command.pose_target[:, :3],
        progress,
        duration,
    )


def bounded_deadzone_quadratic(
    residual: torch.Tensor, scale: float, deadzone: float, max_penalty: float
) -> torch.Tensor:
    """Squared excess speed, normalized in physical units and bounded above."""
    if scale <= 0.0 or deadzone < 0.0 or max_penalty <= 0.0:
        raise ValueError("Require scale > 0, deadzone >= 0, and max_penalty > 0.")
    excess = (torch.linalg.vector_norm(residual, dim=-1) - deadzone).clamp_min(0.0)
    return (excess / scale).square().clamp_max(max_penalty)


def tcp_velocity_errors(env, command_name: str = "ee_target", base_name: str = "base_link"):
    """Return linear/angular velocity errors and update diagnostic command metrics."""
    command = env.command_manager.get_term(command_name)
    asset = env.scene[command.cfg.asset_name]
    base_ids, _ = asset.find_bodies(base_name)
    if len(base_ids) != 1:
        raise ValueError("VD3 TCP stabilization requires one base link.")
    base_id = base_ids[0]
    tcp_position, _ = tcp_pose_w(asset)
    tcp_linear, tcp_angular = tcp_velocity_w(asset)
    base_linear, base_angular = link_origin_velocity_w(asset, base_id)
    base_position, base_quaternion = link_pose_w(asset, base_id)
    linear_b, angular_b = relative_twist_in_base(
        base_quaternion,
        base_position,
        tcp_position,
        base_linear,
        base_angular,
        tcp_linear,
        tcp_angular,
    )
    linear_error = linear_b - command_linear_velocity_b(command, env.step_dt)
    # VD3's configured command orientation is fixed in the full base frame.
    # A future moving-orientation command must provide its angular derivative.
    angular_error = angular_b
    command.metrics["tcp_linear_velocity_error"] = torch.linalg.vector_norm(linear_error, dim=-1)
    command.metrics["tcp_angular_velocity_error"] = torch.linalg.vector_norm(angular_error, dim=-1)
    command.metrics["tcp_relative_linear_speed"] = torch.linalg.vector_norm(linear_b, dim=-1)
    return linear_error, angular_error


def tcp_linear_velocity_residual(
    env,
    command_name: str = "ee_target",
    base_name: str = "base_link",
    scale: float = 0.12,
    deadzone: float = 0.01,
    max_penalty: float = 4.0,
) -> torch.Tensor:
    """Penalize unnecessary TCP translation, including oscillation during gait."""
    linear, _ = tcp_velocity_errors(env, command_name, base_name)
    return bounded_deadzone_quadratic(linear, scale, deadzone, max_penalty)


def tcp_angular_velocity_residual(
    env,
    command_name: str = "ee_target",
    base_name: str = "base_link",
    scale: float = 0.5,
    deadzone: float = 0.05,
    max_penalty: float = 4.0,
) -> torch.Tensor:
    """Penalize relative TCP rotation about a fixed base-frame target attitude."""
    _, angular = tcp_velocity_errors(env, command_name, base_name)
    return bounded_deadzone_quadratic(angular, scale, deadzone, max_penalty)


def flange_target_from_tcp_pose(position_b, quaternion_b, linear_velocity_b, angular_velocity_b):
    """Invert the fixed link06->TCP transform for the commanded flange pose/twist.

    The flange follows the requested tool trajectory; it is never anchored to a
    single location while the commanded TCP moves. All velocities are expressed
    in the same base coordinates as the supplied target pose.
    """
    rotation = quaternion_b.new_tensor(TCP_QUAT_IN_LINK06)
    inverse_rotation = rotation * rotation.new_tensor((1., -1., -1., -1.))
    flange_quaternion = quat_mul(quaternion_b, inverse_rotation.expand_as(quaternion_b))
    offset = position_b.new_tensor(TCP_POS_IN_LINK06).expand_as(position_b)
    offset_b = quat_apply(flange_quaternion, offset)
    return (
        position_b - offset_b,
        flange_quaternion,
        linear_velocity_b - torch.linalg.cross(angular_velocity_b, offset_b, dim=-1),
        angular_velocity_b,
    )


def _flange_state_and_target(env, command_name, base_name, flange_name):
    command = env.command_manager.get_term(command_name)
    asset = env.scene[command.cfg.asset_name]
    indices = []
    for name in (base_name, flange_name):
        body_ids, _ = asset.find_bodies(name)
        if len(body_ids) != 1:
            raise ValueError(f"VD3 flange stabilization requires exactly one {name!r} body.")
        indices.append(body_ids[0])
    base_id, flange_id = indices
    base_position, base_quaternion = link_pose_w(asset, base_id)
    flange_position, _ = link_pose_w(asset, flange_id)
    position_b = _quat_apply_inverse(base_quaternion, flange_position - base_position)
    desired_tcp_velocity = command_linear_velocity_b(command, env.step_dt)
    # The configured TCP attitude is constant throughout each trajectory. Its
    # instantaneous resample is handled by pose tracking, not a finite difference.
    desired_tcp_angular = torch.zeros_like(desired_tcp_velocity)
    target = flange_target_from_tcp_pose(
        command.command[:, :3], command.command[:, 3:7],
        desired_tcp_velocity, desired_tcp_angular,
    )
    position_error = position_b - target[0]
    command.metrics['flange_position_error'] = torch.linalg.vector_norm(position_error, dim=-1)
    return command, asset, base_id, flange_id, position_error, target


def flange_velocity_errors(env, command_name='ee_target', base_name='base_link', flange_name='link06'):
    """Measure flange motion even when rotation about the TCP keeps its position still."""
    command, asset, base_id, flange_id, _, target = _flange_state_and_target(
        env, command_name, base_name, flange_name,
    )
    base_position, base_quaternion = link_pose_w(asset, base_id)
    flange_position, _ = link_pose_w(asset, flange_id)
    base_linear, base_angular = link_origin_velocity_w(asset, base_id)
    flange_linear, flange_angular = link_origin_velocity_w(asset, flange_id)
    linear_b, angular_b = relative_twist_in_base(
        base_quaternion, base_position, flange_position,
        base_linear, base_angular, flange_linear, flange_angular,
    )
    linear_error, angular_error = linear_b - target[2], angular_b - target[3]
    command.metrics['flange_linear_velocity_error'] = torch.linalg.vector_norm(linear_error, dim=-1)
    command.metrics['flange_angular_velocity_error'] = torch.linalg.vector_norm(angular_error, dim=-1)
    command.metrics['flange_relative_linear_speed'] = torch.linalg.vector_norm(linear_b, dim=-1)
    command.metrics['flange_world_linear_speed'] = torch.linalg.vector_norm(flange_linear, dim=-1)
    # Keep TCP diagnostics for the comparison requested by the user. These
    # measurements are no longer the selected linear/angular velocity penalties.
    tcp_velocity_errors(env, command_name, base_name)
    return linear_error, angular_error


def flange_position_residual(
    env, command_name='ee_target', base_name='base_link', flange_name='link06',
    scale=0.05, deadzone=0.005, max_penalty=4.0,
):
    """Discourage flange excursion around the pose implied by the TCP target."""
    _, _, _, _, error, _ = _flange_state_and_target(env, command_name, base_name, flange_name)
    return bounded_deadzone_quadratic(error, scale, deadzone, max_penalty)


def flange_linear_velocity_residual(
    env, command_name='ee_target', base_name='base_link', flange_name='link06',
    scale=0.12, deadzone=0.01, max_penalty=4.0,
):
    """Penalize unnecessary flange translation relative to its moving target."""
    linear, _ = flange_velocity_errors(env, command_name, base_name, flange_name)
    return bounded_deadzone_quadratic(linear, scale, deadzone, max_penalty)


def flange_angular_velocity_residual(
    env, command_name='ee_target', base_name='base_link', flange_name='link06',
    scale=0.5, deadzone=0.05, max_penalty=4.0,
):
    """Penalize flange rotation once; a rigid TCP has the same angular velocity."""
    _, angular = flange_velocity_errors(env, command_name, base_name, flange_name)
    return bounded_deadzone_quadratic(angular, scale, deadzone, max_penalty)


def arm_joint_velocity_l2(env, asset_cfg, command_name: str = "ee_target") -> torch.Tensor:
    """Mild internal-arm-motion regularization using resolved joint IDs.

    The entity config must select exactly the six Z1 arm joints. No assumption
    is made about their positions in the articulation or action vector.
    """
    asset = env.scene[asset_cfg.name]
    velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    if velocity.shape[-1] != 6:
        raise ValueError("VD3 arm velocity regularization requires exactly six selected arm joints.")
    command = env.command_manager.get_term(command_name)
    command.metrics["arm_joint_velocity_rms"] = velocity.square().mean(dim=-1).sqrt()
    return velocity.square().sum(dim=-1)
