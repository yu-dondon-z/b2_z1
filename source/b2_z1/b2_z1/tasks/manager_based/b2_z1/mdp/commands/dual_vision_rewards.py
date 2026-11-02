"""Reward-only distances to the rendered road's physical crack centreline.

These helpers reproduce the *mesh stations* of SeamlessCrackedRoadHalfCfg,
including its flat outer sections. They must never feed visual control,
camera packets, or actor observations. Live perception owns those inputs.
"""

from __future__ import annotations

import math

import torch


def physical_crack_polyline_xy(*, amplitude_m: float, wavelength_m: float,
                               curve_half_length_m: float, road_half_length_m: float,
                               mesh_step_m: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return local XY centre stations identical to the cracked-road mesh."""
    values = (amplitude_m, wavelength_m, curve_half_length_m, road_half_length_m, mesh_step_m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("physical crack dimensions must be finite")
    if (wavelength_m <= 0 or mesh_step_m <= 0 or curve_half_length_m <= 0
            or road_half_length_m <= curve_half_length_m):
        raise ValueError("invalid physical crack mesh dimensions")
    count = round(2 * curve_half_length_m / mesh_step_m) + 1
    if count < 2:
        raise ValueError("physical crack needs at least two curved mesh stations")
    curved_x = torch.linspace(-curve_half_length_m, curve_half_length_m, count,
                              device=device, dtype=dtype)
    curved_y = amplitude_m * torch.sin(2 * math.pi * curved_x / wavelength_m).pow(3)
    curved = torch.stack((curved_x, curved_y), dim=-1)
    outer = torch.tensor([[-road_half_length_m, 0.], [road_half_length_m, 0.]],
                          device=device, dtype=dtype)
    return torch.cat((outer[:1], curved, outer[1:]), dim=0)


@torch.no_grad()
def physical_crack_lateral_distance_w(points_w: torch.Tensor, env_origins: torch.Tensor,
                                      centerline_local_xy: torch.Tensor) -> torch.Tensor:
    """Compute exact XY point-to-mesh-segment distance for each environment.

    ``points_w`` is ``[N, ..., 3]`` and ``env_origins`` is ``[N, 3]``.
    The local centreline is supplied by the scene parameters, rather than
    synthetic trajectory state or a nominal default curve.
    """
    if points_w.ndim < 2 or points_w.shape[-1] != 3:
        raise ValueError("world points must have shape [N, ..., 3]")
    if env_origins.shape != (points_w.shape[0], 3):
        raise ValueError("one world origin is required per environment")
    line = torch.as_tensor(centerline_local_xy, device=points_w.device, dtype=points_w.dtype)
    if line.ndim != 2 or line.shape[1] != 2 or line.shape[0] < 2:
        raise ValueError("physical crack centreline must contain at least two XY stations")
    if not torch.isfinite(line).all():
        raise ValueError("physical crack centreline must be finite")
    origin_shape = (points_w.shape[0],) + (1,) * (points_w.ndim - 2) + (2,)
    local = points_w[..., :2] - env_origins[:, :2].reshape(origin_shape)
    starts = line[:-1]
    vectors = line[1:] - starts
    denominator = vectors.square().sum(dim=-1).clamp_min(1.0e-12)
    delta = local.unsqueeze(-2) - starts
    fraction = (delta * vectors).sum(dim=-1) / denominator
    nearest = starts + fraction.clamp(0, 1).unsqueeze(-1) * vectors
    return (local.unsqueeze(-2) - nearest).square().sum(dim=-1).amin(dim=-1).sqrt()
