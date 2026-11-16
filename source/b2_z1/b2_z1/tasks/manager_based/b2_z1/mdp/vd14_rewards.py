"""Measured-geometry posture costs; no observation, target or contact changes."""
from __future__ import annotations

import math

import torch

from .tcp_frame import quat_apply_inverse


def front_width_deficit(width: torch.Tensor, minimum_width: float = .42,
                        scale: float = .10) -> torch.Tensor:
    """One-sided squared cost: no extra reward for ever wider feet.

    Width is signed left minus right. Crossed feet must not become acceptable
    just because their absolute separation is large.
    """
    if not math.isfinite(minimum_width) or minimum_width <= 0:
        raise ValueError("minimum_width must be finite and positive")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    return ((minimum_width - width).clamp_min(0.) / scale).square()


def front_stance_width_penalty(env, asset_cfg, minimum_width: float = .42,
                               scale: float = .10) -> torch.Tensor:
    """Use real link positions in the full base frame, including swing feet."""
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.body_ids
    if isinstance(ids, slice):
        ids = list(range(len(asset.body_names)))[ids]
    if [asset.body_names[i] for i in ids] != ["FL_foot", "FR_foot"]:
        raise ValueError("Front width requires ordered FL_foot, FR_foot body IDs")
    # Use the same link origins and base transform as AMP's foot observations.
    feet_w = asset.data.body_pos_w[:, ids]
    root_quat = asset.data.root_quat_w[:, None, :].expand(-1, 2, -1)
    feet_b = quat_apply_inverse(root_quat, feet_w - asset.data.root_pos_w[:, None, :])
    width = feet_b[:, 0, 1] - feet_b[:, 1, 1]
    penalty = front_width_deficit(width, minimum_width, scale)
    manager = getattr(env, "command_manager", None)
    if manager is not None:
        manager.get_term("ee_target").metrics.update({
            "vd14_front_width_m": width,
            "vd14_front_width_below_min": (width < minimum_width).to(width.dtype),
            "vd14_base_pitch_rad": torch.asin(asset.data.projected_gravity_b[:, 0].clamp(-1., 1.)),
        })
    return penalty
