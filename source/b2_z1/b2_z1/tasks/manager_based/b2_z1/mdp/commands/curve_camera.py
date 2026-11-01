# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the base-mounted cameras from their command configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    quat_from_euler_xyz,
)

if TYPE_CHECKING:
    from .command_cfg import CurveVelocityCommandCfg, UniformPoseWBCCommandCfg


def make_curve_camera_cfg(cfg: CurveVelocityCommandCfg, *, render: bool) -> AssetBaseCfg | TiledCameraCfg:
    """Build the forward camera with the ground-curve projection calibration."""
    return _make_camera_cfg(cfg, prim_name="BaseCamera", render=render)


def make_rear_camera_cfg(cfg: UniformPoseWBCCommandCfg, *, render: bool) -> AssetBaseCfg | TiledCameraCfg:
    """Build the rear camera observing the base-relative EE workspace."""
    return _make_camera_cfg(cfg, prim_name="RearCamera", render=render)


def _make_camera_cfg(
    cfg: CurveVelocityCommandCfg | UniformPoseWBCCommandCfg, *, prim_name: str, render: bool
) -> AssetBaseCfg | TiledCameraCfg:
    """Spawn a USD Camera for training, or a tiled rendering sensor for Play.

    Creating a rendering sensor instance enables RTX even with an infinite update
    period. A plain AssetBaseCfg spawns the same camera prim without a render
    product or image annotators. Both branches share the command's calibration.
    Play batches the environment cameras into one tiled render product.
    """
    width, height = cfg.camera_image_size
    fx, fy, cx, cy = cfg.camera_intrinsics
    if min(width, height, fx, fy) <= 0:
        raise ValueError("Camera image dimensions and focal lengths must be positive.")
    # Match the pinhole renderer in the installed Isaac Lab / Isaac Sim version.
    # It ignores aperture offsets and uses square pixels when reporting intrinsics.
    if abs(fx - fy) > 1.0e-4 or abs(cx - width / 2) > 1.0e-4 or abs(cy - height / 2) > 1.0e-4:
        raise ValueError("Rendered cameras require fx == fy and principal point (width/2, height/2).")
    if cfg.camera_update_period <= 0:
        raise ValueError("camera_update_period must be positive.")

    prim_path = f"{{ENV_REGEX_NS}}/Robot/{cfg.base_name}/{prim_name}"
    spawn = PinholeCameraCfg.from_intrinsic_matrix(
        [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        width=width,
        height=height,
        clipping_range=cfg.camera_depth_range,
    )
    rpy = torch.tensor(cfg.camera_rpy_b, dtype=torch.float32).unsqueeze(0)
    rotation_world = quat_from_euler_xyz(*rpy.unbind(dim=-1))
    if render:
        return TiledCameraCfg(
            prim_path=prim_path,
            update_period=cfg.camera_update_period,
            width=width,
            height=height,
            data_types=["rgb"],
            spawn=spawn,
            offset=TiledCameraCfg.OffsetCfg(
                pos=cfg.camera_position_b,
                rot=tuple(rotation_world[0].tolist()),
                convention="world",
            ),
        )

    rotation_opengl = convert_camera_frame_orientation_convention(rotation_world, origin="world", target="opengl")
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=spawn,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=cfg.camera_position_b,
            rot=tuple(rotation_opengl[0].tolist()),
        ),
    )
