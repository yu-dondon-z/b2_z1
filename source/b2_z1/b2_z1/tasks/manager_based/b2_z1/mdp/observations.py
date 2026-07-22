from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv
    from isaaclab.assets import Articulation
    from isaaclab.sensors import RayCaster

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
    # link0_ids, _ = asset.find_bodies("link00")
    link0_ids, _ = asset.find_bodies("base_link")

    if len(ee_ids) == 0 or len(link0_ids) == 0:
        return torch.zeros((env.num_envs, 7), device=env.device)

    ee_pos_w    = asset.data.body_pos_w[:, ee_ids[0], :]
    link0_pos_w = asset.data.body_pos_w[:, link0_ids[0], :]
    link0_quat_w = asset.data.body_quat_w[:, link0_ids[0], :]
    ee_quat_w    = asset.data.body_quat_w[:, ee_ids[0], :]

    # Relative position: transform to link0 frame (not root frame)
    rel_pos_w = ee_pos_w - link0_pos_w
    rel_pos   = quat_apply_inverse(link0_quat_w, rel_pos_w)  # fixed

    # Relative orientation: q_rel = q_link0^{-1} * q_ee
    link0_quat_conj = quat_conjugate(link0_quat_w)           # fixed
    rel_quat = quat_mul(link0_quat_conj, ee_quat_w)

    return torch.cat([rel_pos, rel_quat], dim=-1)


################# AMP-specific observations #################
    
    
def foot_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """observe foot position in robot's base frame"""
    asset: Articulation = env.scene[asset_cfg.name]

    # foot positions in the world frame, relative to the root position
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_pos_w = asset.data.root_pos_w.unsqueeze(1)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, foot_pos_w.shape[1], -1)

    # rotate into the base frame
    foot_pos_b = quat_apply_inverse(root_quat_w, foot_pos_w - root_pos_w)
    return foot_pos_b.reshape(env.num_envs, -1)


def base_pos_z_terrain(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """observe base height w.r.t. terrain"""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # mean terrain height under the robot, ignoring invalid ray hits
    ray_hits = sensor.data.ray_hits_w[..., 2]
    valid_mask = ~torch.isnan(ray_hits) & ~torch.isinf(ray_hits) & (torch.abs(ray_hits) < 1e6)
    ray_hits_masked = torch.where(valid_mask, ray_hits, torch.full_like(ray_hits, float("nan")))
    terrain_height = torch.nanmean(ray_hits_masked, dim=1)

    # for envs where all hits are invalid, fall back to zero terrain height
    all_invalid_mask = torch.isnan(terrain_height)
    if all_invalid_mask.any():
        terrain_height[all_invalid_mask] = 0.0

    base_z = asset.data.root_pos_w[:, 2] - terrain_height
    return base_z.unsqueeze(-1)

# [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]  # lab-> data

def joint_pos_reindex(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), indices: list = None) -> torch.Tensor:
    """The joint positions of the asset, reindexed to match the dataset's joint order.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    if indices is None:
        indices = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]  # lab-> data
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return joint_pos[:, indices]


def joint_vel_reindex(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), indices: list = None) -> torch.Tensor:
    """The joint velocities of the asset, reindexed to match the dataset's joint order.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    if indices is None:
        indices = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]  # lab-> data
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return joint_vel[:, indices]



