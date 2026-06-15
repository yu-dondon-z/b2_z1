from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv
    from isaaclab.assets import Articulation

from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul

def base_external_force(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """observe external force applied on the base"""
    asset: Articulation = env.scene[asset_cfg.name]
    # shape: (num_envs, 3)
    return asset._external_force_b[:, asset_cfg.body_ids, :].squeeze(1).clone()


def contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    # first_contact = contact_sensor.data.net_forces_w(env.step_dt)[:, sensor_cfg.body_ids]

    contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(
        dim=-1
    )
    return contact_force

def end_effector_relative_pose(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    ee_ids, _ = asset.find_bodies("dummy_target")
    base_ids, _ = asset.find_bodies("base_link")

    if len(ee_ids) == 0 or len(base_ids) == 0:
        return torch.zeros((env.num_envs, 7), device=env.device)

    ee_pos_w = asset.data.body_pos_w[:, ee_ids[0], :]
    base_pos_w = asset.data.body_pos_w[:, base_ids[0], :]
    base_quat_w = asset.data.body_quat_w[:, base_ids[0], :]
    ee_quat_w = asset.data.body_quat_w[:, ee_ids[0], :]

    rel_pos = quat_apply_inverse(base_quat_w, ee_pos_w - base_pos_w)

    rel_quat = quat_mul(quat_conjugate(base_quat_w), ee_quat_w)

    return torch.cat([rel_pos, rel_quat], dim=-1)



