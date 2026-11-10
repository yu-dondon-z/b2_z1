"""Left/right policy consistency for VD3's B2 + six-axis Z1.

This is a soft constraint on leg outputs, not simultaneous left/right joint
equality. Critic, AMP transitions and PPO samples are deliberately not augmented.
Only torch is needed so geometry can be checked without starting Isaac Sim.
"""

from __future__ import annotations

import torch

from .tcp_frame import TCP_POS_IN_LINK06, TCP_QUAT_IN_LINK06, quat_apply, quat_mul


# Isaac articulation order; joint1 is interleaved with the leg joints.
JOINT_NAMES = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "joint1", "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    "joint2", "joint3", "joint4", "joint5", "joint6",
)
LEG_ACTION_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12)
_JOINT_PERMUTATION = (1, 0, 3, 2, 5, 4, 7, 6, 8, 10, 9, 12, 11, 13, 14, 15, 16, 17)
_JOINT_SIGNS = (-1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1)
_POLICY_TERMS = (
    "velocity_commands", "target_commands", "base_ang_vel", "projected_gravity",
    "joint_pos_rel", "joint_vel_rel", "actions",
)
_POLICY_DIMS = ((3,), (7,), (3,), (3,), (18,), (18,), (18,))


def mirror_joints(values: torch.Tensor) -> torch.Tensor:
    """Swap corresponding legs and reflect joint axes (including Z1 inputs)."""
    if values.shape[-1] != 18:
        raise ValueError("VD3 joint/action mirror requires 18 values.")
    return values[..., list(_JOINT_PERMUTATION)] * values.new_tensor(_JOINT_SIGNS)


def mirror_tcp_pose(position: torch.Tensor, quaternion: torch.Tensor):
    """Reflect the flange, then reapply the actual rounded flange->TCP offset.

    Quaternions are scalar-first. Choose the equivalent quaternion sign closest
    to the input, keeping the existing yaw=pi command on its learned branch.
    """
    if position.shape[-1] != 3 or quaternion.shape[-1] != 4:
        raise ValueError("TCP mirror requires XYZ and a wxyz quaternion.")
    rotation = quaternion.new_tensor(TCP_QUAT_IN_LINK06).expand_as(quaternion)
    conjugate = quaternion.new_tensor((1, -1, -1, -1))
    flange_q = quat_mul(quaternion, rotation * conjugate)
    offset = position.new_tensor(TCP_POS_IN_LINK06).expand_as(position)
    flange_p = position - quat_apply(flange_q, offset)
    mirrored_flange_q = flange_q * quaternion.new_tensor((1, -1, 1, -1))
    mirrored_q = quat_mul(mirrored_flange_q, rotation)
    mirrored_q = torch.where(
        (mirrored_q * quaternion).sum(dim=-1, keepdim=True) < 0, -mirrored_q, mirrored_q
    )
    mirrored_p = flange_p * position.new_tensor((1, -1, 1)) + quat_apply(mirrored_flange_q, offset)
    return mirrored_p, mirrored_q


def _validate_environment(env):
    """Check the indexing and affine-action assumptions once per environment."""
    if env is None:
        return
    unwrapped = env.unwrapped
    if getattr(unwrapped, "_vd3_leg_symmetry_validated", False):
        return
    robot = unwrapped.scene["robot"]
    if tuple(robot.joint_names) != JOINT_NAMES:
        raise ValueError("VD3 symmetry joint order differs from the verified 18-joint layout.")
    manager = unwrapped.observation_manager
    if tuple(manager.active_terms["policy"]) != _POLICY_TERMS or tuple(
        tuple(dim) for dim in manager.group_obs_term_dim["policy"]
    ) != _POLICY_DIMS:
        raise ValueError("VD3 symmetry requires the verified 70-dimensional policy observation layout.")
    action_manager = unwrapped.action_manager
    if list(action_manager.active_terms) != ["JointPositionAction"]:
        raise ValueError("VD3 symmetry requires one JointPositionAction term.")
    action = action_manager.get_term("JointPositionAction")
    if tuple(action._joint_names) != JOINT_NAMES or not action.cfg.use_default_offset:
        raise ValueError("VD3 symmetry requires default-offset actions in articulation order.")
    for name in ("default_joint_pos", "default_joint_vel"):
        default = getattr(robot.data, name)
        if not torch.allclose(default, mirror_joints(default), atol=1e-6, rtol=0):
            raise ValueError(f"VD3 symmetry requires mirror-symmetric {name}.")
    scale = torch.as_tensor(action._scale, device=robot.device)
    if scale.ndim and not torch.allclose(scale, scale[..., list(_JOINT_PERMUTATION)]):
        raise ValueError("VD3 symmetry requires equal scales for mirrored joints.")
    unwrapped._vd3_leg_symmetry_validated = True


def vd3_left_right(obs=None, actions=None, env=None, obs_type="policy"):
    """Return original + mirrored batches in the RSL-RL symmetry API."""
    if obs_type != "policy":
        raise ValueError("VD3 leg consistency does not support critic/data augmentation.")
    _validate_environment(env)
    augmented_obs = augmented_actions = None
    if obs is not None:
        if obs.ndim != 2 or obs.shape[-1] != 70:
            raise ValueError("VD3 symmetry requires a batch of 70-dimensional policy observations.")
        mirrored = obs.clone()
        mirrored[:, :3] = obs[:, :3] * obs.new_tensor((1, -1, -1))
        mirrored[:, 3:6], mirrored[:, 6:10] = mirror_tcp_pose(obs[:, 3:6], obs[:, 6:10])
        mirrored[:, 10:13] = obs[:, 10:13] * obs.new_tensor((-1, 1, -1))
        mirrored[:, 13:16] = obs[:, 13:16] * obs.new_tensor((1, -1, 1))
        for start in (16, 34, 52):
            mirrored[:, start:start + 18] = mirror_joints(obs[:, start:start + 18])
        augmented_obs = torch.cat((obs, mirrored), dim=0)
    if actions is not None:
        if actions.ndim != 2 or actions.shape[-1] != 18:
            raise ValueError("VD3 symmetry requires a batch of 18-dimensional actions.")
        augmented_actions = torch.cat((actions, mirror_joints(actions)), dim=0)
    return augmented_obs, augmented_actions
