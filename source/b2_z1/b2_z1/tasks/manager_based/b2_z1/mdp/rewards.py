from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _heading_hold_mask(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return the pure-translation mask exposed by the omni command term."""
    command_term = env.command_manager.get_term(command_name)
    mask = getattr(command_term, "is_heading_hold_env", None)
    if mask is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return mask


def _longitudinal_mask(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return the forward/backward-only mask exposed by the omni command."""
    command_term = env.command_manager.get_term(command_name)
    mask = getattr(command_term, "is_longitudinal_env", None)
    if mask is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return mask & ~command_term.is_standing_env


def command_mode_lin_vel_xy_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    mode_attribute: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return planar velocity error only for a named structured-command mode."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    mask = getattr(command_term, mode_attribute, None)
    if mask is None:
        return torch.zeros(env.num_envs, device=env.device)
    command = env.command_manager.get_command(command_name)
    error = torch.sum(torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return error * mask.float()


def command_mode_ang_vel_z_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    mode_attribute: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return yaw-rate error only for a named structured-command mode."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    mask = getattr(command_term, mode_attribute, None)
    if mask is None:
        return torch.zeros(env.num_envs, device=env.device)
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return error * mask.float()


def commanded_lin_vel_xy_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return planar command-tracking error for every environment.

    Unlike :func:`command_mode_lin_vel_xy_l2`, this term does not depend on a
    command implementation exposing mode masks.  The truth-curve task derives
    its velocity command from the path tangent, so the generic form is needed
    to make standing still under a moving command explicitly costly.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return torch.sum(
        torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1
    )


def moving_leg_posture_deviation_hinge(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    velocity_threshold: float = 0.08,
    allowed_mean_deviation: float = 0.08,
    deviation_scale: float = 0.15,
) -> torch.Tensor:
    """Penalize a collapsed leg posture without suppressing a normal trot.

    The mean absolute deviation is measured from the nominal standing pose. A
    hinge leaves small excursions of the selected V19 gait unpenalized, while
    normalized quadratic growth makes a persistent crouch expensive. The term
    is disabled for standing commands so it does not fight zero-speed rewards.
    """
    if allowed_mean_deviation < 0.0:
        raise ValueError("allowed_mean_deviation cannot be negative")
    if deviation_scale <= 0.0:
        raise ValueError("deviation_scale must be positive")

    asset: Articulation = env.scene[asset_cfg.name]
    deviation = torch.abs(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    ).mean(dim=1)
    excess = torch.clamp(
        (deviation - allowed_mean_deviation) / deviation_scale,
        min=0.0,
        max=1.0,
    )
    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.vector_norm(command[:, :2], dim=1) > velocity_threshold
    return torch.square(excess) * moving.float()


def moving_body_height_below_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    minimum_height: float,
    height_scale: float = 0.05,
    command_name: str = "base_velocity",
    velocity_threshold: float = 0.08,
) -> torch.Tensor:
    """Penalize leg links whose centres approach the flat training ground.

    VA2 kept the base high while folding its legs outward, so base-height and
    contact-only checks missed the kneeling failure. VA truth-curve training
    uses a flat plane, making each environment origin's z the local ground.
    """
    if minimum_height <= 0.0:
        raise ValueError("minimum_height must be positive")
    if height_scale <= 0.0:
        raise ValueError("height_scale must be positive")

    asset: Articulation = env.scene[asset_cfg.name]
    ground_z = env.scene.env_origins[:, 2].unsqueeze(1)
    body_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - ground_z
    deficit = torch.clamp(
        (minimum_height - body_height) / height_scale,
        min=0.0,
        max=1.0,
    )
    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.vector_norm(command[:, :2], dim=1) > velocity_threshold
    return torch.mean(torch.square(deficit), dim=1) * moving.float()


def commanded_along_path_speed_ratio(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    velocity_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward actual progress along the commanded planar direction.

    Moving environments receive ``clamp(v_actual_along / v_command, 0, 1)``.
    Therefore a stationary policy gets zero even when its TCP and posture are
    perfect, while overspeed cannot exploit the reward.  True standing
    commands receive one so the term does not destabilize zero-speed samples.
    The bounded score is also used as an unambiguous curriculum gate.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_xy = env.command_manager.get_command(command_name)[:, :2]
    target_speed = torch.linalg.norm(command_xy, dim=1)
    moving = target_speed > velocity_threshold
    direction = command_xy / target_speed.clamp_min(velocity_threshold).unsqueeze(1)
    actual_along = torch.sum(asset.data.root_lin_vel_b[:, :2] * direction, dim=1)
    ratio = torch.clamp(actual_along / target_speed.clamp_min(velocity_threshold), 0.0, 1.0)
    return torch.where(moving, ratio, torch.ones_like(ratio))


def straight_lateral_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body-frame lateral slip during pure straight translation."""
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.square(asset.data.root_lin_vel_b[:, 1])
    return error * _longitudinal_mask(env, command_name).float()


def straight_yaw_rate_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize unintended yaw rate during pure straight translation."""
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.square(asset.data.root_ang_vel_b[:, 2])
    return error * _heading_hold_mask(env, command_name).float()


def straight_heading_drift_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize accumulated world-heading drift in pure translation modes."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    target = getattr(command_term, "straight_heading_target", asset.data.heading_w)
    error = math_utils.wrap_to_pi(asset.data.heading_w - target)
    return torch.square(error) * _heading_hold_mask(env, command_name).float()


def _synthetic_straight_path_mask(
    env: ManagerBasedRLEnv,
    path_command_name: str,
    velocity_command_name: str,
    velocity_threshold: float,
) -> torch.Tensor:
    """Mask moving environments whose privileged training path is straight."""
    path_term = env.command_manager.get_term(path_command_name)
    straight = getattr(path_term, "is_straight_path", None)
    if straight is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    command = env.command_manager.get_command(velocity_command_name)
    moving = torch.linalg.norm(command[:, :2], dim=1) > velocity_threshold
    return straight & moving


def path_straight_lateral_velocity_l2(
    env: ManagerBasedRLEnv,
    path_command_name: str,
    velocity_command_name: str = "base_velocity",
    velocity_threshold: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize side-slip only on explicitly sampled straight path segments."""
    asset: Articulation = env.scene[asset_cfg.name]
    mask = _synthetic_straight_path_mask(
        env, path_command_name, velocity_command_name, velocity_threshold
    )
    return torch.square(asset.data.root_lin_vel_b[:, 1]) * mask.float()


def path_straight_yaw_rate_l2(
    env: ManagerBasedRLEnv,
    path_command_name: str,
    velocity_command_name: str = "base_velocity",
    velocity_threshold: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize unintended turning while the sampled path is straight."""
    asset: Articulation = env.scene[asset_cfg.name]
    mask = _synthetic_straight_path_mask(
        env, path_command_name, velocity_command_name, velocity_threshold
    )
    return torch.square(asset.data.root_ang_vel_b[:, 2]) * mask.float()


def path_straight_heading_drift_l2(
    env: ManagerBasedRLEnv,
    path_command_name: str,
    velocity_command_name: str = "base_velocity",
    velocity_threshold: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize accumulated yaw drift relative to the straight path heading."""
    asset: Articulation = env.scene[asset_cfg.name]
    path_term = env.command_manager.get_term(path_command_name)
    target = getattr(path_term, "straight_heading_target", asset.data.heading_w)
    mask = _synthetic_straight_path_mask(
        env, path_command_name, velocity_command_name, velocity_threshold
    )
    error = math_utils.wrap_to_pi(asset.data.heading_w - target)
    return torch.square(error) * mask.float()

"""
Joint penalties.
"""

def smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the second finite difference of actions using an L2 kernel."""
    current_action = env.action_manager.action
    previous_action = env.action_manager.prev_action
    return torch.sum(
        torch.square(current_action - 2.0 * previous_action + env.pre_pre_action),
        dim=1,
    )


def joint_group_action_rate_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize action changes only for the selected joints.

    This lets the arm receive stronger smoothing than the legs without
    suppressing the leg action bandwidth needed for a trot.
    """
    joint_ids = asset_cfg.joint_ids
    action_delta = (
        env.action_manager.action[:, joint_ids]
        - env.action_manager.prev_action[:, joint_ids]
    )
    return torch.sum(torch.square(action_delta), dim=1)


def joint_group_velocity_huber(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    speed_scale: float = 1.0,
) -> torch.Tensor:
    """Smoothly penalize fast measured joint motion without quadratic spikes."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return (torch.sqrt(1.0 + torch.square(joint_velocity / speed_scale)) - 1.0).sum(dim=1)


def joint_pair_opposed_action_rate_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize only the changing opposed component of a parallel Jacobian pair.

    For Z1 joint3/joint4, the measured end-effector Jacobian columns are
    nearly parallel. Same-direction motion can move the TCP, while opposing
    motion primarily changes the arm's internal configuration. Let the policy
    learn this tradeoff rather than altering its actions after inference.
    """
    joint_ids = asset_cfg.joint_ids
    if len(joint_ids) != 2:
        raise ValueError("joint_pair_opposed_action_rate_l2 requires exactly two joints")
    delta = env.action_manager.action[:, joint_ids] - env.action_manager.prev_action[:, joint_ids]
    return torch.square(delta[:, 0] - delta[:, 1])


def tcp_arm_nullspace_velocity_cost(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "ee_target",
    velocity_scale: float = 0.6,
    tracking_error_scale: float = 0.12,
    damping: float = 0.05,
) -> torch.Tensor:
    """Penalize arm motion that does not move the TCP toward its position task.

    The *measured* arm velocity is split using the current TCP translational
    Jacobian. Only its damped nullspace residual is penalized. The penalty
    fades when the TCP is far from its target, so the policy can first recover
    tracking. This is a training reward, never a post-policy action filter.
    """
    if velocity_scale <= 0 or tracking_error_scale <= 0 or damping <= 0:
        raise ValueError("nullspace reward scales and damping must be positive")
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if len(joint_ids) != 6 or len(asset_cfg.body_ids) != 1:
        raise ValueError("nullspace reward requires six Z1 joints and one TCP body")

    body_id = asset_cfg.body_ids[0]
    jacobian_body_id = body_id - 1 if asset.is_fixed_base else body_id
    jacobian_joint_ids = joint_ids if asset.is_fixed_base else [joint_id + 6 for joint_id in joint_ids]
    # Select joints on the final axis to preserve (env, xyz, joint) ordering.
    jacobian = asset.root_physx_view.get_jacobians()[:, jacobian_body_id, :3, :][
        ..., jacobian_joint_ids
    ]
    joint_velocity = asset.data.joint_vel[:, joint_ids]
    task_velocity = torch.bmm(jacobian, joint_velocity.unsqueeze(-1))
    metric = torch.bmm(jacobian, jacobian.transpose(1, 2))
    eye = torch.eye(3, device=metric.device, dtype=metric.dtype).unsqueeze(0)
    projected = torch.bmm(
        jacobian.transpose(1, 2),
        torch.linalg.solve(metric + damping**2 * eye, task_velocity),
    ).squeeze(-1)
    null_velocity = joint_velocity - projected
    null_cost = (
        torch.sqrt(1.0 + torch.square(null_velocity / velocity_scale)) - 1.0
    ).mean(dim=1)

    target_b = env.command_manager.get_command(command_name)[:, :3]
    base_id = asset.find_bodies("base_link")[0][0]
    tcp_b = quat_apply_inverse(
        asset.data.body_quat_w[:, base_id],
        asset.data.body_pos_w[:, body_id] - asset.data.body_pos_w[:, base_id],
    )
    tracking_error = torch.linalg.norm(tcp_b - target_b, dim=1)
    tracking_gate = torch.exp(-torch.square(tracking_error / tracking_error_scale))
    return null_cost * tracking_gate

def power_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize power consumption of the robot."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.applied_torque) * torch.abs(asset.data.joint_vel), dim=1)



def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    cur_pos = asset.data.joint_pos[..., asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[..., asset_cfg.joint_ids]
    reward = torch.linalg.norm(cur_pos - default_pos, dim=1)
    
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def joint_torques_max(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the maximum absolute torque within a joint group.

    Useful for applying different penalty strengths per joint group (hip, thigh, calf, arm).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_indices, _ = asset.find_joints(joint_names)
    torques = asset.data.applied_torque[:, joint_indices]
    max_abs_torque = torch.max(torch.abs(torques), dim=1).values
    return torch.square(max_abs_torque)


# def joint_deviation_l1(
#     env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Penalize joints deviating from their default positions using L1 kernel.

#     Active only when the robot is upright (projected_gravity_z < -0.7).
#     """
#     asset: Articulation = env.scene[asset_cfg.name]
#     angle = (
#         asset.data.joint_pos[:, asset_cfg.joint_ids]
#         - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
#     )
#     reward = torch.sum(torch.abs(angle), dim=1)
#     reward *= (-env.scene["robot"].data.projected_gravity_b[:, 2]) > 0.7
#     return reward


def joint_mirror(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]
) -> torch.Tensor:
    """Penalize asymmetry between mirrored joint pairs (e.g. front-right vs rear-left).

    Uses cached joint indices for efficiency.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if (
        not hasattr(env, "joint_mirror_joints_cache")
        or env.joint_mirror_joints_cache is None
    ):
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair]
            for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in env.joint_mirror_joints_cache:
        diff = torch.sum(
            torch.square(
                asset.data.joint_pos[:, joint_pair[0][0]]
                - asset.data.joint_pos[:, joint_pair[1][0]]
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


"""
Robot.
"""

def base_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float) -> torch.Tensor:
    """Reward the base for being close to a target height."""
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return reward

def base_height_tracking(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward maintaining the base at a target height above the ground."""
    asset: RigidObject = env.scene[asset_cfg.name]
    curr_height = asset.data.root_pos_w[:, 2]
    height_error = torch.abs(target_height - curr_height)
    return torch.exp(-height_error / std)



def uncontacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # import ipdb; ipdb.set_trace()
    # is_contact = torch.max(torch.norm(net_contact_forces[:, :5, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # return torch.sum(is_contact.float(), dim=1)
    
    is_contact = torch.norm(net_contact_forces[:, 0, sensor_cfg.body_ids], dim=-1) > threshold
    re = is_contact.float().mean(-1) # mean parts
    
    return re


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward



def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    
# def feet_slide(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Penalize lateral foot sliding when feet are in contact with the ground."""
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     contacts = (
#         contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
#         .norm(dim=-1)
#         .max(dim=1)[0]
#         > 1.0
#     )
#     asset: RigidObject = env.scene[asset_cfg.name]
#     cur_footvel_translated = asset.data.body_lin_vel_w[
#         :, asset_cfg.body_ids, :
#     ] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
#     footvel_in_body_frame = torch.zeros(
#         env.num_envs, len(asset_cfg.body_ids), 3, device=env.device
#     )
#     for i in range(len(asset_cfg.body_ids)):
#         footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
#             asset.data.root_quat_w, cur_footvel_translated[:, i, :]
#         )
#     foot_leteral_vel = torch.sqrt(
#         torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)
#     ).view(env.num_envs, -1)
#     reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    
#     reward *= (
#         torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
#     )
#     return reward



def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)





"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


def feet_long_air_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_air_time: float = 0.5,
    velocity_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize feet that stay airborne longer than max_air_time."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.data.current_air_time is None:
        return torch.zeros(env.num_envs, device=env.device)
    current_air = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    excess = torch.clamp(current_air - max_air_time, min=0.0)
    # return torch.sum(torch.square(excess), dim=1)
    
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    is_still = cmd <= velocity_threshold
    
    reward = torch.sum(torch.square(excess), dim=1)
    
    return torch.where(is_still, torch.zeros_like(reward), reward)


def prolonged_foot_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_contact_time: float = 0.40,
    maximum_excess_time: float = 0.40,
    velocity_threshold: float = 0.10,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize a foot remaining continuously planted for too long while moving.

    The term is deliberately disabled at near-zero commands so that stable
    four-foot standing is not mistaken for a gait defect.  A bounded squared
    hinge keeps normal stance intervals free of penalty and prevents a stuck
    foot late in an episode from producing an unbounded reward magnitude.
    """
    if max_contact_time < 0.0:
        raise ValueError("max_contact_time cannot be negative")
    if maximum_excess_time <= 0.0:
        raise ValueError("maximum_excess_time must be positive")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_contact_time = contact_sensor.data.current_contact_time
    if current_contact_time is None:
        return torch.zeros(env.num_envs, device=env.device)

    contact_time = current_contact_time[:, sensor_cfg.body_ids]
    excess = torch.clamp(
        contact_time - max_contact_time,
        min=0.0,
        max=maximum_excess_time,
    )
    # Sum over feet so one persistently planted foot remains observable.  The
    # configured reward weight controls the total contribution.
    penalty = torch.sum(torch.square(excess / maximum_excess_time), dim=1)

    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.norm(command, dim=1) > velocity_threshold
    return torch.where(moving, penalty, torch.zeros_like(penalty))


class DiagonalContactStateMismatchPenalty(ManagerTermBase):
    """Penalize disagreement inside each diagonal trot pair while moving.

    The existing smooth gait reward compares contact and air *timings*.  This
    complementary term acts directly on the current binary contact state, so
    a diagonal pair with only one planted foot receives an immediate signal.
    It is disabled for near-zero commands to preserve stable four-foot stance.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        pair_names = cfg.params["synced_feet_pair_names"]
        if len(pair_names) != 2 or any(len(pair) != 2 for pair in pair_names):
            raise ValueError("Expected two diagonal foot pairs with two feet per pair.")
        self.synced_feet_pairs = [
            self.contact_sensor.find_bodies(pair)[0] for pair in pair_names
        ]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        synced_feet_pair_names,
        velocity_threshold: float = 0.10,
        command_name: str = "base_velocity",
    ) -> torch.Tensor:
        del sensor_cfg, synced_feet_pair_names
        current_contact_time = self.contact_sensor.data.current_contact_time
        if current_contact_time is None:
            return torch.zeros(env.num_envs, device=env.device)

        contacts = current_contact_time > 0.0
        mismatch = torch.zeros(env.num_envs, device=env.device)
        for first_foot, second_foot in self.synced_feet_pairs:
            mismatch += torch.logical_xor(
                contacts[:, first_foot], contacts[:, second_foot]
            ).float()
        mismatch *= 0.5

        command = env.command_manager.get_command(command_name)
        moving = torch.linalg.norm(command, dim=1) > velocity_threshold
        return torch.where(moving, mismatch, torch.zeros_like(mismatch))


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        
        if not hasattr(self, "last_change_step"):
            self.last_change_step = torch.zeros_like(env.episode_length_buf).unsqueeze(-1).repeat(1, 4)
            self.step_global = torch.zeros_like(self.last_change_step)
            
            
        self.last_change_step += 1 
        self.step_global += 1
        
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        # return torch.where(
        #     torch.logical_or(cmd > 0.1, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
        # )
        
        return torch.where(
            cmd > 0.1, sync_reward * async_reward, 0.0
        )
    

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)
    



def stand_still_contact_reward(
    env,
    sensor_cfg,
    velocity_threshold: float = 0.1,
    contact_force_threshold: float = 5.0,
    target_contacts: int = 4,
):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    is_still = cmd <= velocity_threshold

    contact_force = torch.norm(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1
    )
    is_contact = contact_force > contact_force_threshold
    num_contacts = is_contact.float().sum(dim=1)

    reward = 1.0 - torch.abs(num_contacts - target_contacts) / target_contacts
    return torch.where(is_still, reward, torch.zeros_like(reward))


def stand_still_feet_air_penalty(
    env,
    sensor_cfg,
    velocity_threshold: float = 0.1,
    max_air_time: float = 0.15,
):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.data.current_air_time is None:
        return torch.zeros(env.num_envs, device=env.device)

    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    is_still = cmd <= velocity_threshold

    current_air = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    excess = torch.clamp(current_air - max_air_time, min=0.0)
    penalty = torch.sum(torch.square(excess), dim=1)

    return torch.where(is_still, penalty, torch.zeros_like(penalty))


def stand_still_load_balance_penalty(
    env,
    sensor_cfg,
    velocity_threshold: float = 0.1,
    eps: float = 1e-6,
):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    is_still = cmd <= velocity_threshold

    fz = torch.clamp(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2],
        min=0.0,
    )
    normalized_fz = fz / (torch.sum(fz, dim=1, keepdim=True) + eps)
    penalty = torch.var(normalized_fz, dim=1)

    return torch.where(is_still, penalty, torch.zeros_like(penalty))



"""
Other rewards.
"""


def position_command_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
    base_name: str = "base_link",
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    ee_id = asset_cfg.body_ids[0]
    base_id, _ = asset.find_bodies(base_name)
    base_id = base_id[0]
    command = env.command_manager.get_command(command_name)

    ee_pos_w = asset.data.body_pos_w[:, ee_id]
    base_pos_w = asset.data.body_pos_w[:, base_id]
    base_quat_w = asset.data.body_quat_w[:, base_id]

    end_effector_curr_pos_base = quat_apply_inverse(base_quat_w, ee_pos_w - base_pos_w)
    pos_error = end_effector_curr_pos_base - command[:, :3]
    
    return torch.norm(pos_error, dim=1)
    

def position_command_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
    base_name: str = "base_link",
) -> torch.Tensor:
    """Reward end-effector position tracking in the base frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ee_id = asset_cfg.body_ids[0]
    base_id, _ = asset.find_bodies(base_name)
    base_id = base_id[0]
    command = env.command_manager.get_command(command_name)

    ee_pos_w = asset.data.body_pos_w[:, ee_id]
    base_pos_w = asset.data.body_pos_w[:, base_id]
    base_quat_w = asset.data.body_quat_w[:, base_id]

    end_effector_curr_pos_base = quat_apply_inverse(base_quat_w, ee_pos_w - base_pos_w)
    pos_error = end_effector_curr_pos_base - command[:, :3]

    error_sq = torch.sum(torch.square(pos_error), dim=1)
    return torch.exp(-error_sq / (std**2))


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
    base_name: str = "base_link",
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    ee_id = asset_cfg.body_ids[0]
    base_id, _ = asset.find_bodies(base_name)
    base_id = base_id[0]
    command = env.command_manager.get_command(command_name)

    ee_pos_w = asset.data.body_pos_w[:, ee_id]
    base_pos_w = asset.data.body_pos_w[:, base_id]
    base_quat_w = asset.data.body_quat_w[:, base_id]

    end_effector_curr_pos_base = quat_apply_inverse(base_quat_w, ee_pos_w - base_pos_w)
    pos_error = end_effector_curr_pos_base - command[:, :3]

    distance = torch.norm(pos_error, dim=1)
    return 1 - torch.tanh(distance / std)


def link_stability(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    base_name: str = "base_link",
    link_name: str = "link06",
    lin_acc_std: float = 0.5,
    ang_acc_std: float = 0.5,
) -> torch.Tensor:
    """Reward the link for being stable relative to base frame (excludes base movement).
    
    Encourages smooth, stable motion of the specified link in the base frame by penalizing:
    - Linear acceleration relative to base
    - Angular acceleration deviations
    - High linear velocities relative to base
    - High angular velocities
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    link_id, _ = asset.find_bodies(link_name)
    link_idx = link_id[0]
    
    base_id, _ = asset.find_bodies(base_name)
    base_idx = base_id[0]

    # Extract accelerations and velocities in world frame
    lin_acc_w = asset.data.body_com_lin_acc_w[:, link_idx, :]
    ang_acc_w = asset.data.body_com_ang_acc_w[:, link_idx, :]
    
    # Get base frame data  
    base_lin_acc_w = asset.data.body_com_lin_acc_w[:, base_idx, :]
    base_ang_acc_w = asset.data.body_com_ang_acc_w[:, base_idx, :]
    
    # Calculate relative linear acceleration (in world frame)
    rel_lin_acc_w = lin_acc_w - base_lin_acc_w
    
    # Calculate relative angular acceleration (difference in angular acceleration)
    rel_ang_acc_w = ang_acc_w - base_ang_acc_w
    
    # Compute norms
    lin_acc_norm = torch.norm(rel_lin_acc_w, dim=1)
    ang_acc_norm = torch.norm(rel_ang_acc_w, dim=1)
    
    # Exponential penalty (lower values = higher reward, penalizes large values)
    lin_acc_reward = -1 * torch.expm1(-lin_acc_norm / lin_acc_std)
    ang_acc_reward = -1 * torch.expm1(-ang_acc_norm / ang_acc_std)
    
    # Combine rewards (equal weighting by default)
    reward = lin_acc_reward + 1.0 * ang_acc_reward 
    
    return reward


def link_world_stability(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    link_name: str = "link_gun",
    lin_acc_std: float = 2.0,
    ang_acc_std: float = 4.0,
    angular_weight: float = 0.25,
) -> torch.Tensor:
    """Penalize endpoint acceleration in the world frame with a bounded cost.

    ``link_stability`` intentionally subtracts base acceleration.  That is
    useful for suppressing arm motion relative to the torso, but it cannot see
    endpoint shake caused by torso roll/pitch during locomotion.  This term
    complements it by measuring the actual world-frame endpoint acceleration.
    The exponential saturation prevents contact impulses from dominating PPO.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    link_ids, _ = asset.find_bodies(link_name)
    link_idx = link_ids[0]

    lin_acc_norm = torch.linalg.norm(
        asset.data.body_com_lin_acc_w[:, link_idx, :], dim=1
    )
    ang_acc_norm = torch.linalg.norm(
        asset.data.body_com_ang_acc_w[:, link_idx, :], dim=1
    )

    lin_cost = -torch.expm1(-lin_acc_norm / lin_acc_std)
    ang_cost = -torch.expm1(-ang_acc_norm / ang_acc_std)
    return lin_cost + angular_weight * ang_cost


def tcp_cross_path_velocity_cost(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "ee_target",
    vertical_scale: float = 0.20,
    lateral_scale: float = 0.20,
) -> torch.Tensor:
    """Penalize tool velocity normal to the crack, not progress along it.

    The VA3 world-acceleration penalty saturates on gait impulses, while the
    arm can still oscillate rapidly enough to worsen actual TCP tracking.
    A smooth pseudo-Huber velocity cost keeps a useful gradient for those
    oscillations without rewarding a stationary tool on a moving path.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    tool_velocity_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :]
    command_term = env.command_manager.get_term(command_name)
    tangent_error = command_term._path_heading_error_b
    tangent_heading_w = asset.data.heading_w + tangent_error
    lateral_velocity = (
        -torch.sin(tangent_heading_w) * tool_velocity_w[:, 0]
        + torch.cos(tangent_heading_w) * tool_velocity_w[:, 1]
    )
    vertical_velocity = tool_velocity_w[:, 2]
    vertical_cost = torch.sqrt(1.0 + torch.square(vertical_velocity / vertical_scale)) - 1.0
    lateral_cost = torch.sqrt(1.0 + torch.square(lateral_velocity / lateral_scale)) - 1.0
    return vertical_cost + lateral_cost


def crack_centerline_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.10,
    velocity_threshold: float = 0.08,
) -> torch.Tensor:
    """Reward keeping the B2 base centre over the synthetic crack.

    The crack command owns the privileged synthetic path geometry during
    training.  Deployment still consumes only the original velocity and TCP
    commands, so this term does not change the policy observation dimension.
    """
    command_term = env.command_manager.get_term(command_name)
    lateral_error = getattr(command_term, "_path_lateral_error_b", None)
    if lateral_error is None:
        return torch.zeros(env.num_envs, device=env.device)
    command_speed = torch.linalg.norm(
        env.command_manager.get_command("base_velocity")[:, :2], dim=1
    )
    reward = torch.exp(-torch.square(lateral_error / max(std, 1.0e-6)))
    return torch.where(
        command_speed > velocity_threshold, reward, torch.zeros_like(reward)
    )


def crack_heading_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.20,
    velocity_threshold: float = 0.08,
) -> torch.Tensor:
    """Reward aligning the base heading with the local crack tangent."""
    command_term = env.command_manager.get_term(command_name)
    heading_error = getattr(command_term, "_path_heading_error_b", None)
    if heading_error is None:
        return torch.zeros(env.num_envs, device=env.device)
    command_speed = torch.linalg.norm(
        env.command_manager.get_command("base_velocity")[:, :2], dim=1
    )
    reward = torch.exp(-torch.square(heading_error / max(std, 1.0e-6)))
    return torch.where(
        command_speed > velocity_threshold, reward, torch.zeros_like(reward)
    )


def crack_path_error_pseudo_huber(
    env: ManagerBasedRLEnv,
    command_name: str,
    error_attribute: str,
    scale: float,
    velocity_threshold: float = 0.08,
) -> torch.Tensor:
    """Give centreline/heading errors a useful gradient beyond the exp reward.

    The existing exp terms are nearly zero after a large miss, allowing the
    arm to follow the crack while B2 itself drifts away. This dimensionless
    pseudo-Huber cost grows approximately linearly for larger deviations.
    """
    if scale <= 0:
        raise ValueError("crack path error scale must be positive")
    command_term = env.command_manager.get_term(command_name)
    error = getattr(command_term, error_attribute)
    command_speed = torch.linalg.vector_norm(
        env.command_manager.get_command("base_velocity")[:, :2], dim=1
    )
    cost = torch.sqrt(1.0 + torch.square(error / scale)) - 1.0
    return torch.where(command_speed > velocity_threshold, cost, torch.zeros_like(cost))


def commanded_yaw_rate_pseudo_huber(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    scale: float = 0.10,
    velocity_threshold: float = 0.08,
) -> torch.Tensor:
    """Penalize signed yaw-rate error even when its exponential reward saturates.

    This acts on the measured body yaw rate, not on a path-derived surrogate:
    a positive left-turn command with a negative actual yaw is penalized.
    """
    if scale <= 0:
        raise ValueError("yaw-rate error scale must be positive")
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = asset.data.root_ang_vel_b[:, 2] - command[:, 2]
    cost = torch.sqrt(1.0 + torch.square(error / scale)) - 1.0
    moving = torch.linalg.vector_norm(command[:, :2], dim=1) > velocity_threshold
    return torch.where(moving, cost, torch.zeros_like(cost))


def stance_feet_crack_clearance_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    safe_distance: float = 0.07,
    contact_force_threshold: float = 5.0,
    velocity_threshold: float = 0.08,
) -> torch.Tensor:
    """Penalize support feet whose centres approach the crack trench.

    Only contacting feet are penalized. Swing feet may cross the centreline,
    but the policy is taught to place each stance foot far enough from the
    2-cm groove to retain a practical sole-width safety margin.
    """
    command_term = env.command_manager.get_term(command_name)
    distance_fn = getattr(command_term, "crack_lateral_distance_w", None)
    if distance_fn is None:
        return torch.zeros(env.num_envs, device=env.device)

    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_positions_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    crack_distance = distance_fn(foot_positions_w)
    contact_force = torch.linalg.norm(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1
    )
    support_mask = contact_force > contact_force_threshold
    normalized_violation = torch.clamp(
        (safe_distance - crack_distance) / max(safe_distance, 1.0e-6), min=0.0
    )
    penalty = torch.sum(
        torch.square(normalized_violation) * support_mask.float(), dim=1
    ) / support_mask.float().sum(dim=1).clamp_min(1.0)
    command_speed = torch.linalg.norm(
        env.command_manager.get_command("base_velocity")[:, :2], dim=1
    )
    return torch.where(
        command_speed > velocity_threshold, penalty, torch.zeros_like(penalty)
    )


def swing_leg_height_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.15,
    velocity_threshold: float = 0.1,
    std: float = 0.1,
    include_body_velocity: bool = True,
) -> torch.Tensor:
    """Reward feet clearance during swing phase - encourages lifting legs when moving.
    
    This reward encourages the policy to lift feet higher during the swing phase (when not in contact),
    which helps avoid stumbling and improves gait quality.
    
    Args:
        env: The RL environment instance
        sensor_cfg: Scene entity config for contact sensor
        asset_cfg: Scene entity config for the robot
        target_height: Target height for feet during swing phase (relative to ground)
        std: Standard deviation for exponential reward decay
        
    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Extract contact sensor and asset
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Check if feet are in contact (stance phase)
    # If contact time > 0, foot is in contact
    is_in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    
    # Get foot positions
    foot_positions = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (num_envs, num_feet, 3)
    foot_heights = foot_positions[:, :, 2]  # (num_envs, num_feet)
    
    # Calculate height error for each foot (how far from target height)
    height_error = torch.abs(foot_heights - target_height)
    
    # Exponential reward for reaching target height
    # Higher reward when foot is at target height during swing
    height_reward = torch.exp(-height_error / std)
    
    # Apply mask: only reward height when foot is NOT in contact (swing phase)
    swing_phase_mask = ~is_in_contact  # Invert: swing phase = not in contact
    
    # Combine: reward height during swing phase only
    reward = torch.sum(height_reward * swing_phase_mask.float(), dim=1)
    
    # Normalize by number of feet to keep reward in reasonable range
    num_feet = swing_phase_mask.shape[1]
    if num_feet > 0:
        reward = reward / num_feet
        
    # return reward 
        
    # except from the reward when the robot is standing still 
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    # is_still = cmd <= velocity_threshold
    
    commanded_move = cmd > velocity_threshold
    if include_body_velocity:
        body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
        commanded_move = torch.logical_or(commanded_move, body_vel > velocity_threshold)
    return torch.where(commanded_move, reward, torch.zeros_like(reward))


def swing_foot_clearance_deficit(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    minimum_clearance: float = 0.055,
    foot_radius: float = 0.032,
    minimum_air_time: float = 0.04,
    velocity_threshold: float = 0.08,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize a swinging foot whose sole is too close to flat ground.

    Unlike :func:`swing_leg_height_reward`, this term measures the clearance
    of the bottom of the spherical B2 foot rather than the height of its link
    origin.  The penalty starts only after a short airborne debounce so that
    normal contact transitions do not dominate the signal.

    The current training terrain is flat and each cloned environment is
    translated by ``env_origins``.  Its local ground plane is therefore the
    environment origin's z coordinate.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    current_air_time = contact_sensor.data.current_air_time
    current_contact_time = contact_sensor.data.current_contact_time
    if current_air_time is None or current_contact_time is None:
        return torch.zeros(env.num_envs, device=env.device)

    current_air_time = current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = current_contact_time[:, sensor_cfg.body_ids]
    swing_mask = torch.logical_and(
        current_contact_time <= 0.0,
        current_air_time >= minimum_air_time,
    )

    foot_center_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    ground_z = env.scene.env_origins[:, 2].unsqueeze(1)
    sole_clearance = foot_center_z - ground_z - foot_radius
    normalized_deficit = torch.clamp(
        (minimum_clearance - sole_clearance) / max(minimum_clearance, 1.0e-6),
        min=0.0,
        max=2.0,
    )
    active_swing_feet = swing_mask.float()
    penalty = torch.sum(torch.square(normalized_deficit) * active_swing_feet, dim=1)
    penalty /= active_swing_feet.sum(dim=1).clamp_min(1.0)

    command_speed = torch.linalg.norm(
        env.command_manager.get_command(command_name)[:, :2], dim=1
    )
    return torch.where(
        command_speed > velocity_threshold,
        penalty,
        torch.zeros_like(penalty),
    )


def swing_foot_clearance_target_exp(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_clearance: float = 0.035,
    foot_radius: float = 0.032,
    minimum_air_time: float = 0.04,
    velocity_threshold: float = 0.08,
    std: float = 0.02,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward a physical sole clearance during the debounced swing phase.

    The older height term used the foot-link world z coordinate.  This version
    subtracts both the cloned environment ground height and the spherical B2
    foot radius, making ``target_clearance`` an actual sole-to-ground distance.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    current_air_time = contact_sensor.data.current_air_time
    current_contact_time = contact_sensor.data.current_contact_time
    if current_air_time is None or current_contact_time is None:
        return torch.zeros(env.num_envs, device=env.device)

    current_air_time = current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = current_contact_time[:, sensor_cfg.body_ids]
    swing_mask = torch.logical_and(
        current_contact_time <= 0.0,
        current_air_time >= minimum_air_time,
    )

    foot_center_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    ground_z = env.scene.env_origins[:, 2].unsqueeze(1)
    sole_clearance = foot_center_z - ground_z - foot_radius
    clearance_reward = torch.exp(-torch.abs(sole_clearance - target_clearance) / std)
    # Divide by the configured feet rather than active feet.  This retains a
    # useful incentive for a regular swing duty cycle as well as its height.
    reward = torch.sum(clearance_reward * swing_mask.float(), dim=1)
    reward /= max(swing_mask.shape[1], 1)

    command_speed = torch.linalg.norm(
        env.command_manager.get_command(command_name)[:, :2], dim=1
    )
    return torch.where(
        command_speed > velocity_threshold,
        reward,
        torch.zeros_like(reward),
    )
    
