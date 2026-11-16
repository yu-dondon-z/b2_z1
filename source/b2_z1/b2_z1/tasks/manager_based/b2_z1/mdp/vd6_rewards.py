"""VD6 regularization: soft landing and arm command continuity.

Training-only truth is used for rewards, never added to actor observations.
No historical Bezier derivative or full flange-attitude target is assumed.
"""
import torch


def descending_foot_penalty(height, vertical_velocity, contact, band=.10, speed_limit=.35):
    """Penalize fast descending flight only in the last 10 cm above flat ground."""
    near_ground = (height >= -.01) & (height < band) & ~contact
    excess = (-vertical_velocity-speed_limit).clamp_min(0.)
    return torch.where(near_ground, excess.square(), 0.).mean(dim=-1)


def feet_soft_landing(env, asset_cfg, sensor_cfg, band=.10, speed_limit=.35):
    asset, sensor = env.scene[asset_cfg.name], env.scene[sensor_cfg.name]
    # VD6 explicitly requires plane terrain. Foot link origins are sphere centers.
    height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]-env.scene.env_origins[:, 2:3]-.032
    velocity = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, 2]
    contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 10.
    return descending_foot_penalty(height, velocity, contact, band, speed_limit)


def feet_impact_excess(env, sensor_cfg, threshold=350., scale=200.):
    sensor = env.scene[sensor_cfg.name]
    # The latest actor interval contains four 200 Hz contact samples, including
    # impacts that may already have subsided at the 50 Hz reward boundary.
    count = max(1, round(env.step_dt/env.physics_dt))
    forces = sensor.data.net_forces_w_history[:, :count, sensor_cfg.body_ids, 2]
    peak = forces.amax(dim=1)
    return ((peak-threshold).clamp_min(0.)/scale).square().mean(dim=-1)


def arm_action_continuity(env, asset_cfg, second_order=False, deadzone=.02, scale=.05):
    """Actual joint-target changes in radians, leaving leg gait actions alone."""
    action, previous = env.action_manager.action, env.action_manager.prev_action
    if action.shape[-1] != len(env.scene[asset_cfg.name].joint_names):
        raise ValueError('VD6 requires the original all-joint 18D position action')
    delta = action-previous
    if second_order:
        delta = action-2.*previous+env.pre_pre_action
    # The VD6 position action retains the original uniform scale 0.25.
    radians = .25*delta[:, asset_cfg.joint_ids]
    return ((radians.abs()-deadzone).clamp_min(0.)/scale).square().mean(dim=-1)
