"""Pure geometry for VD5's simulated seam-to-base command conversion."""

from __future__ import annotations

import math

import torch


def sample_polyline_behind_position(
    points_w: torch.Tensor, position_w: torch.Tensor, trailing_distance: float
) -> torch.Tensor:
    """Sample a world path trailing a position, extending its first tangent at startup."""
    if points_w.ndim != 3 or position_w.ndim != 2 or points_w.shape[0] != position_w.shape[0]:
        raise ValueError("Expected path (batch, points, 3) and position (batch, 3).")
    if points_w.shape[1] < 2 or points_w.shape[2] != 3 or position_w.shape[1] != 3:
        raise ValueError("A 3D path needs at least two points.")
    if trailing_distance < 0.0:
        raise ValueError("trailing_distance must be nonnegative.")

    starts = points_w[:, :-1]
    segments = points_w[:, 1:] - starts
    lengths = torch.linalg.vector_norm(segments, dim=-1)
    safe_lengths_sq = lengths.square().clamp_min(1.0e-12)
    relative = position_w.unsqueeze(1) - starts
    fraction = ((relative * segments).sum(dim=-1) / safe_lengths_sq).clamp(0.0, 1.0)
    projected = starts + fraction.unsqueeze(-1) * segments
    nearest_segment = torch.linalg.vector_norm(projected - position_w.unsqueeze(1), dim=-1).argmin(dim=1)

    cumulative = torch.cat((lengths.new_zeros(lengths.shape[0], 1), lengths.cumsum(dim=1)), dim=1)
    nearest = nearest_segment.unsqueeze(1)
    base_distance = cumulative.gather(1, nearest).squeeze(1)
    base_distance += fraction.gather(1, nearest).squeeze(1) * lengths.gather(1, nearest).squeeze(1)
    target_distance = base_distance - trailing_distance

    segment_index = (target_distance.unsqueeze(1) >= cumulative[:, 1:]).sum(dim=1)
    segment_index = segment_index.clamp(max=points_w.shape[1] - 2)
    segment_start_distance = cumulative.gather(1, segment_index.unsqueeze(1)).squeeze(1)
    segment_length = lengths.gather(1, segment_index.unsqueeze(1)).squeeze(1)
    target_fraction = ((target_distance - segment_start_distance) / segment_length.clamp_min(1.0e-8)).clamp(0.0, 1.0)
    batch = torch.arange(points_w.shape[0], device=points_w.device)
    target = starts[batch, segment_index] + target_fraction.unsqueeze(-1) * segments[batch, segment_index]

    first_tangent = segments[:, 0] / lengths[:, :1].clamp_min(1.0e-8)
    extrapolated = points_w[:, 0] + target_distance.unsqueeze(-1) * first_tangent
    return torch.where((target_distance < 0.0).unsqueeze(-1), extrapolated, target)


def world_position_to_base(position_w: torch.Tensor, base_pos_w: torch.Tensor, base_quat_w: torch.Tensor):
    """Transform positions from world to base coordinates (quaternions are wxyz)."""
    relative = position_w - base_pos_w
    quat = base_quat_w * base_quat_w.new_tensor((1.0, -1.0, -1.0, -1.0))
    cross = 2.0 * torch.cross(quat[..., 1:], relative, dim=-1)
    return relative + quat[..., :1] * cross + torch.cross(quat[..., 1:], cross, dim=-1)


def base_position_to_world(position_b: torch.Tensor, base_pos_w: torch.Tensor, base_quat_w: torch.Tensor):
    """Transform positions from base coordinates to world coordinates (quaternions are wxyz)."""
    cross = 2.0 * torch.cross(base_quat_w[..., 1:], position_b, dim=-1)
    rotated = position_b + base_quat_w[..., :1] * cross + torch.cross(base_quat_w[..., 1:], cross, dim=-1)
    return base_pos_w + rotated


def blend_target_position(nominal_b: torch.Tensor, seam_b: torch.Tensor, blend: float) -> torch.Tensor:
    """Interpolate an established base-frame target toward the seam-derived target."""
    if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be finite and in [0, 1].")
    if nominal_b.shape != seam_b.shape:
        raise ValueError("Nominal and seam target tensors must have matching shapes.")
    return nominal_b + blend * (seam_b - nominal_b)
