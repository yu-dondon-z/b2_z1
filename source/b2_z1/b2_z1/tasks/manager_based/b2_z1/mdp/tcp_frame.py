"""VD3 gun TCP as a massless frame attached to a real articulation link.

Only torch is required. Older assets with a real ``dummy_target`` body remain
supported, while the VD3 asset uses the unchanged link06 inertia and a fixed TCP.
"""

from __future__ import annotations

import torch


# Composed from joint_mounter, joint_gun_mounter and link_dummy_target_joint in
# the preserved source URDF, including its rounded RPY values.
TCP_POS_IN_LINK06 = (0.10499999999979762, 1.1019615309915131e-7, -0.8199999999999998)
TCP_QUAT_IN_LINK06 = (
    0.707108079857527, -1.6591560751643165e-6,
    0.7071054825092897, 1.6591690618375932e-6,
)


def quat_mul(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    a, b = first[..., 1:], second[..., 1:]
    return torch.cat((
        first[..., :1] * second[..., :1] - (a * b).sum(dim=-1, keepdim=True),
        first[..., :1] * b + second[..., :1] * a + torch.cross(a, b, dim=-1),
    ), dim=-1)


def quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    cross = 2.0 * torch.cross(quat[..., 1:], vector, dim=-1)
    return vector + quat[..., :1] * cross + torch.cross(quat[..., 1:], cross, dim=-1)


def quat_apply_inverse(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat * quat.new_tensor((1.0, -1.0, -1.0, -1.0)), vector)


def _find_body(asset, name: str) -> int | None:
    try:
        indices, _ = asset.find_bodies(name)
    except (KeyError, ValueError):
        return None
    return int(indices[0]) if len(indices) else None


def tcp_parent(asset) -> tuple[int, bool]:
    """Return (body index, needs fixed offset), cached on the articulation."""
    cached = getattr(asset, "_vd3_tcp_parent", None)
    if cached is None:
        index = _find_body(asset, "dummy_target")
        if index is not None:
            cached = (index, False)
        else:
            index = _find_body(asset, "link06")
            if index is None:
                raise ValueError("TCP requires either dummy_target or link06 in the robot asset.")
            cached = (index, True)
        asset._vd3_tcp_parent = cached
    return cached


def link_pose_w(asset, body_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    data = asset.data
    positions = data.body_link_pos_w if hasattr(data, "body_link_pos_w") else data.body_pos_w
    if hasattr(data, "body_link_quat_w"):
        quaternions = data.body_link_quat_w
    elif hasattr(data, "body_quat_w"):
        quaternions = data.body_quat_w
    else:
        quaternions = data.body_state_w[..., 3:7]
    return positions[:, body_index], quaternions[:, body_index]


def link_origin_velocity_w(asset, body_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear velocity at the link origin and its world angular velocity.

    IsaacLab's body_lin_vel_w/body_state_w velocities are at the COM. Prefer
    body_link_*; the compatibility branch explicitly shifts from COM to origin.
    """
    data = asset.data
    if hasattr(data, "body_link_lin_vel_w") and hasattr(data, "body_link_ang_vel_w"):
        return data.body_link_lin_vel_w[:, body_index], data.body_link_ang_vel_w[:, body_index]
    if hasattr(data, "body_com_lin_vel_w"):
        lin, ang = data.body_com_lin_vel_w[:, body_index], data.body_com_ang_vel_w[:, body_index]
    elif hasattr(data, "body_lin_vel_w"):
        lin, ang = data.body_lin_vel_w[:, body_index], data.body_ang_vel_w[:, body_index]
    else:
        lin, ang = data.body_state_w[:, body_index, 7:10], data.body_state_w[:, body_index, 10:13]
    origin, quat = link_pose_w(asset, body_index)
    if hasattr(data, "body_com_pos_w"):
        offset = origin - data.body_com_pos_w[:, body_index]
    elif hasattr(data, "body_com_pos_b"):
        offset = -quat_apply(quat, data.body_com_pos_b[:, body_index])
    else:
        raise ValueError("COM positions are required to convert body COM velocity to link-origin velocity.")
    return lin + torch.cross(ang, offset, dim=-1), ang


def tcp_pose_w(asset) -> tuple[torch.Tensor, torch.Tensor]:
    """World TCP position and quaternion in wxyz convention."""
    index, needs_offset = tcp_parent(asset)
    pos, quat = link_pose_w(asset, index)
    if not needs_offset:
        return pos, quat
    offset = pos.new_tensor(TCP_POS_IN_LINK06).expand_as(pos)
    rotation = quat.new_tensor(TCP_QUAT_IN_LINK06).expand_as(quat)
    return pos + quat_apply(quat, offset), quat_mul(quat, rotation)


def tcp_velocity_w(asset) -> tuple[torch.Tensor, torch.Tensor]:
    """World linear velocity at TCP and world angular velocity."""
    index, needs_offset = tcp_parent(asset)
    lin, ang = link_origin_velocity_w(asset, index)
    if not needs_offset:
        return lin, ang
    pos, quat = link_pose_w(asset, index)
    offset_w = quat_apply(quat, pos.new_tensor(TCP_POS_IN_LINK06).expand_as(pos))
    return lin + torch.cross(ang, offset_w, dim=-1), ang


def tcp_pose_b(asset, base_name: str = "base_link") -> tuple[torch.Tensor, torch.Tensor]:
    """TCP pose relative to the complete moving base frame."""
    base_id = _find_body(asset, base_name)
    if base_id is None:
        raise ValueError(f"TCP base body {base_name!r} was not found.")
    base_pos, base_quat = link_pose_w(asset, base_id)
    pos, quat = tcp_pose_w(asset)
    inverse = base_quat * base_quat.new_tensor((1.0, -1.0, -1.0, -1.0))
    return quat_apply_inverse(base_quat, pos - base_pos), quat_mul(inverse, quat)
