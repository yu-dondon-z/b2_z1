# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched curve and ideal ground-camera geometry, independent of the simulator."""

import torch


def cubic_bezier(
    start: torch.Tensor,
    control_1: torch.Tensor,
    control_2: torch.Tensor,
    end: torch.Tensor,
    progress: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a cubic Bezier curve with broadcastable points and progress."""
    remaining = 1.0 - progress
    return (
        remaining**3 * start
        + 3.0 * remaining**2 * progress * control_1
        + 3.0 * remaining * progress**2 * control_2
        + progress**3 * end
    )


def limit_ground_curve_curvature(control_points: torch.Tensor, max_curvature: float) -> torch.Tensor:
    """Bound the whole local ground Bezier curve's absolute curvature in 1/m.

    The four control points must have equally spaced, increasing x coordinates
    and constant z, as in CurveVelocityCommand's sampler. Then x'(t) = length
    and x''(t) = 0, so |curvature| <= |y''(t)| / length**2. Since y'' is linear,
    its largest absolute value occurs at an endpoint. Scaling all lateral
    offsets by this conservative bound guarantees the limit for every t in [0, 1].
    """
    length = control_points[:, -1, 0] - control_points[:, 0, 0]
    second_difference = control_points[:, 2:, 1] - 2.0 * control_points[:, 1:-1, 1] + control_points[:, :-2, 1]
    curvature_bound = 6.0 * second_difference.abs().amax(dim=-1) / length.square()
    # Leave straight curves unchanged, including when max_curvature is zero.
    scale = torch.where(
        curvature_bound > 0.0,
        (max_curvature / curvature_bound.clamp_min(torch.finfo(control_points.dtype).tiny)).clamp(max=1.0),
        1.0,
    )
    limited = control_points.clone()
    lateral_start = control_points[:, :1, 1]
    limited[:, :, 1] = lateral_start + scale.unsqueeze(-1) * (control_points[:, :, 1] - lateral_start)
    return limited


def straight_turn_straight(
    length: torch.Tensor,
    turn_angle: torch.Tensor,
    curvature: float,
    lead_in_distance: float,
    lead_out_distance: float,
    progress: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-turn interface to multi_turn_path; return points and signed angle."""
    points, angles = multi_turn_path(
        length, turn_angle.unsqueeze(-1), curvature, lead_in_distance, 0.0, lead_out_distance, progress
    )
    return points, angles[:, 0]


def multi_turn_path(
    length: torch.Tensor,
    turn_angles: torch.Tensor,
    curvature: float,
    lead_in_distance: float,
    turn_gap_distance: float,
    lead_out_distance: float,
    progress: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample tangent-continuous circular turns separated by straight segments.

    turn_angles has shape (environments, turns), with signed radians. Return
    equal-arc-length ground samples and actual angles. Reserve lead-in, gaps,
    and lead-out before uniformly shrinking angles to fit the length budget.
    This preserves curvature bounds and the relative turn pattern on short paths.
    """
    distance = length.unsqueeze(-1) * progress.reshape(1, -1)
    points = torch.zeros(*distance.shape, 3, device=length.device, dtype=length.dtype)
    if curvature == 0.0:
        points[..., 0] = distance
        return points, torch.zeros_like(turn_angles)

    num_turns = turn_angles.shape[1]
    lead_in = torch.minimum(torch.full_like(length, lead_in_distance), 0.25 * length)
    available = (length - lead_in - lead_out_distance).clamp_min(0.0)
    gap = torch.minimum(
        torch.full_like(length, turn_gap_distance), available / (2.0 * max(1, num_turns - 1))
    )
    arc_budget = (available - gap * (num_turns - 1)).clamp_min(0.0)
    total_angle = turn_angles.abs().sum(dim=-1)
    scale = (curvature * arc_budget / total_angle.clamp_min(torch.finfo(length.dtype).tiny)).clamp(max=1.0)
    angles = turn_angles * scale.unsqueeze(-1)
    points[..., 0] = torch.minimum(distance, lead_in.unsqueeze(-1))
    cursor, heading = lead_in, torch.zeros_like(length)
    for index in range(num_turns):
        angle = angles[:, index]
        arc_length = angle.abs() / curvature
        arc_distance = torch.minimum((distance - cursor.unsqueeze(-1)).clamp_min(0.0), arc_length.unsqueeze(-1))
        phase = curvature * arc_distance
        dx = torch.sin(phase) / curvature
        # 2*sin(phase/2)^2 avoids cancellation for small turn angles.
        dy = angle.sign().unsqueeze(-1) * 2.0 * torch.sin(0.5 * phase).square() / curvature
        cosine, sine = heading.cos().unsqueeze(-1), heading.sin().unsqueeze(-1)
        points[..., 0] += cosine * dx - sine * dy
        points[..., 1] += sine * dx + cosine * dy
        cursor = cursor + arc_length
        heading = heading + angle
        if index + 1 < num_turns:
            straight = torch.minimum((distance - cursor.unsqueeze(-1)).clamp_min(0.0), gap.unsqueeze(-1))
            points[..., 0] += straight * heading.cos().unsqueeze(-1)
            points[..., 1] += straight * heading.sin().unsqueeze(-1)
            cursor = cursor + gap
    remaining = (distance - cursor.unsqueeze(-1)).clamp_min(0.0)
    points[..., 0] += remaining * heading.cos().unsqueeze(-1)
    points[..., 1] += remaining * heading.sin().unsqueeze(-1)
    return points, angles


def perceive_ground_curve(
    points_c: torch.Tensor,
    ground_normal_c: torch.Tensor,
    camera_height: torch.Tensor,
    intrinsics: tuple[float, float, float, float],
    image_size: tuple[int, int],
    depth_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a ground curve into an ideal camera, then intersect pixel rays with the ground.

    The camera uses x-forward, y-left, z-up axes. ``intrinsics`` is (fx, fy, cx, cy)
    and ``image_size`` is (width, height). The ground plane in camera coordinates is
    ``normal . point + camera_height = 0``. Only points inside the image and depth
    range are returned as visible. There is no RGB rendering or occlusion model.
    """
    fx, fy, cx, cy = intrinsics
    width, height = image_size
    depth = points_c[..., 0]
    safe_depth = depth.clamp_min(depth_range[0])
    pixels = torch.stack((cx - fx * points_c[..., 1] / safe_depth, cy - fy * points_c[..., 2] / safe_depth), dim=-1)
    rays = torch.stack((torch.ones_like(depth), -(pixels[..., 0] - cx) / fx, -(pixels[..., 1] - cy) / fy), dim=-1)
    denominator = torch.sum(ground_normal_c.unsqueeze(1) * rays, dim=-1)
    valid_ray = denominator < -1.0e-6
    distance = -camera_height.unsqueeze(-1) / torch.where(valid_ray, denominator, -torch.ones_like(denominator))
    visible = (
        (depth >= depth_range[0])
        & (depth <= depth_range[1])
        & (pixels[..., 0] >= 0.0)
        & (pixels[..., 0] < width)
        & (pixels[..., 1] >= 0.0)
        & (pixels[..., 1] < height)
        & valid_ray
        & (distance > 0.0)
        & torch.isfinite(points_c).all(dim=-1)
    )
    recovered_points = torch.where(visible.unsqueeze(-1), rays * distance.unsqueeze(-1), 0.0)
    return recovered_points, pixels, visible


def curve_tracking_errors(
    points_h: torch.Tensor, visible: torch.Tensor, lookahead_distance: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate signed lateral error, corrected heading error, and curvature from visible segments.

    ``points_h`` are ordered in the direction of travel in a horizontal base frame
    (x-forward, y-left). Positive lateral error means the base is left of the curve.
    The nearest visible segment's tangent line is extended towards the base because
    a forward-facing camera cannot generally see the ground directly under it.
    """
    starts = points_h[:, :-1, :2]
    segments = points_h[:, 1:, :2] - starts
    lengths_squared = torch.sum(segments**2, dim=-1)
    segment_valid = visible[:, :-1] & visible[:, 1:] & (lengths_squared > 1.0e-8)
    projection = -torch.sum(starts * segments, dim=-1) / lengths_squared.clamp_min(1.0e-8)
    closest_points = starts + projection.clamp(0.0, 1.0).unsqueeze(-1) * segments
    distances = torch.sum(closest_points**2, dim=-1).masked_fill(~segment_valid, float("inf"))
    indices = distances.argmin(dim=-1)
    rows = torch.arange(points_h.shape[0], device=points_h.device)
    valid = segment_valid.any(dim=-1)

    tangent = segments[rows, indices] / lengths_squared[rows, indices].sqrt().clamp_min(1.0e-4).unsqueeze(-1)
    point = starts[rows, indices]
    lateral_error = point[:, 0] * tangent[:, 1] - point[:, 1] * tangent[:, 0]
    tangent_heading = torch.atan2(tangent[:, 1], tangent[:, 0])
    heading_error = tangent_heading - torch.atan2(lateral_error, torch.full_like(lateral_error, lookahead_distance))
    heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))

    # Signed circumcircle curvature at each interior vertex; exclude invisible triples.
    previous, following = segments[:, :-1], segments[:, 1:]
    cross = previous[..., 0] * following[..., 1] - previous[..., 1] * following[..., 0]
    denominator = (
        torch.linalg.vector_norm(previous, dim=-1)
        * torch.linalg.vector_norm(following, dim=-1)
        * torch.linalg.vector_norm(previous + following, dim=-1)
    )
    vertex_curvature = 2.0 * cross / denominator.clamp_min(1.0e-8)
    vertex_valid = segment_valid[:, :-1] & segment_valid[:, 1:]
    vertex_curvature = torch.where(vertex_valid, vertex_curvature, 0.0)
    vertex_curvature = torch.cat((vertex_curvature[:, :1], vertex_curvature, vertex_curvature[:, -1:]), dim=-1)
    vertex_valid = torch.cat((vertex_valid[:, :1], vertex_valid, vertex_valid[:, -1:]), dim=-1)
    # A segment at the image boundary may only have one visible curvature estimate.
    weights = vertex_valid[rows, indices].float() + vertex_valid[rows, indices + 1].float()
    curvature = (vertex_curvature[rows, indices] + vertex_curvature[rows, indices + 1]) / weights.clamp_min(1.0)
    return tuple(torch.where(valid, value, 0.0) for value in (lateral_error, heading_error, curvature)) + (valid,)
