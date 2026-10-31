"""Curve-tracking environment with AMP reference-state resets."""

import torch
from isaaclab.utils.math import convert_quat, quat_apply

from b2_z1.amp_config import AMP_JOINT_NAMES
from b2_z1.rsl_rl.datasets import AMPLoader
from .b2z1_rl_env import B2Z1ManagerBasedRLEnv


class B2Z1ManagerAMPBasedRLEnv(B2Z1ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self.amp_loader = None
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        if not 0.0 <= cfg.reference_state_initialization_prob <= 1.0:
            raise ValueError("reference_state_initialization_prob must be between 0 and 1.")
        if cfg.reference_state_initialization:
            self.amp_loader = AMPLoader(
                device=self.device, time_between_frames=self.step_dt, motion_files=cfg.amp_motion_files
            )
        # Resolve actual articulation IDs: the six arm joints may be interleaved.
        self.amp_joint_ids, _ = self.scene["robot"].find_joints(AMP_JOINT_NAMES, preserve_order=True)
        if len(self.amp_joint_ids) != 12:
            raise ValueError("AMP requires the twelve B2 leg joints in dataset order.")

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._pre_pre_action[env_ids] = 0.0
        if self.amp_loader is None or len(env_ids) == 0:
            return
        use_reference = torch.rand(len(env_ids), device=self.device) < self.cfg.reference_state_initialization_prob
        ref_ids = env_ids[use_reference]
        if len(ref_ids) == 0:
            return

        frames = self.amp_loader.get_full_frame_batch(len(ref_ids))
        robot = self.scene["robot"]
        robot.write_joint_state_to_sim(
            AMPLoader.get_joint_pose_batch(frames), AMPLoader.get_joint_vel_batch(frames),
            joint_ids=self.amp_joint_ids, env_ids=ref_ids,
        )
        root_pos = AMPLoader.get_root_pos_batch(frames).clone() + self.scene.env_origins[ref_ids]
        root_pos[:, 2] += self.cfg.commands.base_velocity.ground_height
        root_quat = convert_quat(AMPLoader.get_root_rot_batch(frames), to="wxyz")
        robot.write_root_pose_to_sim(torch.cat((root_pos, root_quat), dim=-1), env_ids=ref_ids)
        root_lin_vel = quat_apply(root_quat, AMPLoader.get_linear_vel_batch(frames))
        root_ang_vel = quat_apply(root_quat, AMPLoader.get_angular_vel_batch(frames))
        robot.write_root_velocity_to_sim(torch.cat((root_lin_vel, root_ang_vel), dim=-1), env_ids=ref_ids)

        self.scene.write_data_to_sim()
        self.sim.forward()
        # Both independent world-space curves must start from the final reset pose.
        self.command_manager.reset(ref_ids)
