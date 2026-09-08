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
    
