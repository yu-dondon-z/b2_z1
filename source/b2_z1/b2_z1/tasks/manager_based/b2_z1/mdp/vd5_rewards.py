"""VD5 rewards for TCP position and welding-gun-axis alignment."""

from __future__ import annotations

import torch

from .tcp_frame import quat_apply, tcp_pose_b


def tcp_gun_axis_command_error(env, command_name: str = "ee_target", asset_cfg=None,
                               base_name: str = "base_link") -> torch.Tensor:
    """Return the TCP +X gun-axis angle to the commanded gun axis, in radians.

    Rotation around the nozzle axis is deliberately ignored: it does not change
    the point or direction that the gun tip presents to the seam.
    """
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    _, current_quaternion = tcp_pose_b(asset, base_name)
    desired_quaternion = env.command_manager.get_command(command_name)[:, 3:7]
    current_quaternion = torch.nn.functional.normalize(current_quaternion, dim=-1)
    desired_quaternion = torch.nn.functional.normalize(desired_quaternion, dim=-1)
    gun_axis = current_quaternion.new_tensor((1.0, 0.0, 0.0)).expand_as(current_quaternion[..., 1:])
    current_axis = quat_apply(current_quaternion, gun_axis)
    desired_axis = quat_apply(desired_quaternion, gun_axis)
    cosine = torch.sum(current_axis * desired_axis, dim=-1).clamp(-1.0, 1.0)
    error = torch.acos(cosine)
    command = env.command_manager.get_term(command_name)
    command.metrics["gun_axis_error"] = error
    return error
