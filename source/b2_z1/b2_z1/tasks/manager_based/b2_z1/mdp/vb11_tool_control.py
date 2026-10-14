"""VB11 measured startup and flat-road tool constraints, with no action edits."""
import math
import torch


def rotate(q, v):
    t = 2 * torch.cross(q[:, 1:], v, dim=-1)
    return v + q[:, :1] * t + torch.cross(q[:, 1:], t, dim=-1)


def gun_bottom_z(asset, body_id):
    q = asset.data.body_link_quat_w[:, body_id]
    center = asset.data.body_link_pos_w[:, body_id] + rotate(
        q, q.new_tensor([.003, -.032, -.05]).expand(len(q), 3))
    axis_z = 1 - 2 * (q[:, 1].square() + q[:, 2].square())
    return center[:, 2] - .55 * axis_z.abs() - .028061 * (1 - axis_z.square()).clamp_min(0).sqrt()


def ready_mask(error, target_error, arm_speed, upright, height, force, clearance,
               position_tolerance=.08, height_tolerance=.01):
    """Final standby target, not a temporarily easy interpolated waypoint."""
    finite = (torch.isfinite(error).all(-1) & torch.isfinite(target_error).all(-1)
              & torch.isfinite(arm_speed) & torch.isfinite(upright)
              & torch.isfinite(height) & torch.isfinite(force) & torch.isfinite(clearance))
    return (finite & (error.norm(dim=-1) < position_tolerance)
            & (error[:, 2].abs() < height_tolerance) & (target_error.norm(dim=-1) < .005)
            & (arm_speed < .8) & (upright > math.cos(.10)) & (height > .45)
            & (force < 5.) & (clearance > .005))


def settle_latch(ready, elapsed, latched, dt, settle_s):
    elapsed = torch.where(ready, elapsed + dt, torch.zeros_like(elapsed))
    return elapsed, latched | (elapsed >= settle_s)


def huber(value, scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Tool cost scale must be positive and finite")
    return (torch.sqrt(1 + (value / scale).square()) - 1).clamp(max=10.)


def tcp_height_shortfall(env, command_name="ee_target", tolerance_m=.005, scale_m=.01):
    term = env.command_manager.get_term(command_name)
    actual = term.robot.data.body_pos_w[:, term.body_idx, 2]
    shortfall = (term.tracking_pose_w[:, 2] - actual - tolerance_m).clamp_min(0)
    return huber(shortfall, scale_m)


def tcp_height_precision(env, command_name="ee_target", std_m=.015):
    term = env.command_manager.get_term(command_name)
    actual = term.robot.data.body_pos_w[:, term.body_idx, 2]
    return torch.exp(-((actual - term.tracking_pose_w[:, 2]) / std_m).square())


def tcp_below_road(env, command_name="ee_target", margin_m=.003, scale_m=.01):
    term = env.command_manager.get_term(command_name)
    road = term.road_height_w()
    actual = term.robot.data.body_pos_w[:, term.body_idx, 2]
    return huber((road + margin_m - actual).clamp_min(0), scale_m)


def gun_bottom_margin(env, asset_cfg, margin_m=.005, scale_m=.005):
    term = env.command_manager.get_term("ee_target")
    bottom = gun_bottom_z(env.scene[asset_cfg.name], asset_cfg.body_ids[0])
    return huber((term.road_height_w() + margin_m - bottom).clamp_min(0), scale_m)


def contact_debt_step(force, age, debt, dt, threshold_n=5., grace_s=4., recovery_rate=.25):
    """Intermittent contact cannot evade the continuous-contact timer forever."""
    updated = (debt + torch.where(force > threshold_n, dt, -dt * recovery_rate)).clamp_min(0)
    return torch.where(age > grace_s, updated, torch.zeros_like(updated))


def intermittent_gun_contact(env, sensor_cfg, threshold_n=5., budget_s=.4,
                             grace_s=4., recovery_rate=.25):
    term = env.command_manager.get_term("ee_target")
    force = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids[0]].norm(dim=-1)
    if not hasattr(env, "_vb11_contact_debt_s"):
        env._vb11_contact_debt_s = torch.zeros_like(force)
    env._vb11_contact_debt_s = contact_debt_step(
        force, term._episode_age_s, env._vb11_contact_debt_s, env.step_dt,
        threshold_n, grace_s, recovery_rate)
    return env._vb11_contact_debt_s >= budget_s


def continuous_gun_contact(env, sensor_cfg, threshold_n=10., duration_s=.8, grace_s=4.):
    term = env.command_manager.get_term("ee_target")
    force = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids[0]].norm(dim=-1)
    if not hasattr(env, "_vb11_continuous_contact_s"):
        env._vb11_continuous_contact_s = torch.zeros_like(force)
    contact = (force > threshold_n) & (term._episode_age_s > grace_s)
    env._vb11_continuous_contact_s = torch.where(
        contact, env._vb11_continuous_contact_s + env.step_dt, 0.)
    return env._vb11_continuous_contact_s >= duration_s


def startup_not_ready(env, command_name="ee_target", deadline_s=12.):
    term = env.command_manager.get_term(command_name)
    latched = (term._startup_ready if term.cfg.mode == "synthetic"
               else torch.full_like(term._startup_ready, term._file_motion_ready))
    return (term._startup_wait_age_s > deadline_s) & ~latched
