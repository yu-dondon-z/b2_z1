import torch
from typing import Dict, Any
from isaaclab.envs import ManagerBasedRLEnv


import torch
from typing import Dict, Any
from isaaclab.envs import ManagerBasedRLEnv
from collections.abc import Sequence


from isaaclab.utils.math import convert_quat, quat_apply
from b2_z1.rsl_rl.datasets import AMPLoader


class B2Z1ManagerAMPBasedRLEnv(ManagerBasedRLEnv):
    
    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        """Initialize the HimLoco environment.
        
        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None.
            **kwargs: Additional keyword arguments (e.g., env_cfg_entry_point from gym.register).
        """
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._pre_pre_action = torch.zeros((self.num_envs, self.action_manager.total_action_dim), device=self.device)
        
        # amp init 
        if self.cfg.reference_state_initialization:
            print("motion_files dir: ")
            print(self.cfg.amp_motion_files)
            self.amp_loader = AMPLoader(motion_files=self.cfg.amp_motion_files, device=self.device, time_between_frames=self.step_dt)

            # data order:  FL(hip,thigh,calf), FR(hip,thigh,calf), RL(hip,thigh,calf), RR(hip,thigh,calf)
            # IsaacLab order: all hips (FL,FR,RL,RR), all thighs, all calves
            data_to_lab = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
            # inverse permutation: joint_ids[j] is the lab index that the j-th data column maps to,
            # so it can be passed directly to write_joint_state_to_sim without gathering the data.
            self.data_joint_ids = [0] * len(data_to_lab)
            for lab_idx, data_idx in enumerate(data_to_lab):
                self.data_joint_ids[data_idx] = lab_idx
        

    def step(self, action: torch.Tensor):
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self._pre_pre_action = self.action_manager.prev_action.clone()
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            
            
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # # -- get observations before reset
        # self.obs_buf_before_reset = self.observation_manager.compute()
        
        
        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()
        
       

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        
        # amp 
        terminal_amp_states = self.observation_manager.compute()["amp_obs"][reset_env_ids]
        
        self.extras["terminal_amp_states"] = terminal_amp_states
        self.extras["reset_env_ids"] = reset_env_ids
        
        
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            
            self._reset_idx(reset_env_ids) 
            
            #! amp reset
            # env ids whose root/joint state gets overwritten by the AMP reference frame below.
            # `_reset_idx` above already resampled the command terms (e.g. ee_target) using the
            # pre-override pose, so these ids need their commands re-sampled once the AMP pose
            # has actually been written into the simulation.
            amp_override_env_ids = None
            if self.cfg.reference_state_initialization:
                prob = self.cfg.reference_state_initialization_prob
                robot = self.scene['robot']
                
                if prob >= 1.0:
                    frames = self.amp_loader.get_full_frame_batch(len(reset_env_ids))
                    self._reset_dofs_amp(reset_env_ids, frames, robot)
                    self._reset_root_states_amp(reset_env_ids, frames, robot)
                    amp_override_env_ids = reset_env_ids
                elif (prob > 0.0 and prob < 1.0):
                    use_ref = torch.rand(len(reset_env_ids), device=self.device) < prob
                    ref_env_ids = reset_env_ids[use_ref]
                    # default_env_ids = reset_env_ids[~use_ref]
                    if len(ref_env_ids) > 0:
                        frames = self.amp_loader.get_full_frame_batch(len(ref_env_ids))
                        self._reset_dofs_amp(ref_env_ids, frames, robot)
                        self._reset_root_states_amp(ref_env_ids, frames, robot)
                        amp_override_env_ids = ref_env_ids
  
                
                
            
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # re-sample the ee target command for envs whose pose was just overwritten by the
            # AMP reference frame, so the tracked trajectory/start point start from the robot's
            # actual post-reset position instead of the stale pre-override pose.
            if amp_override_env_ids is not None and len(amp_override_env_ids) > 0:
                self.command_manager.get_term("ee_target").reset(amp_override_env_ids)

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
        # return self.obs_buf, self.obs_buf_before_reset, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    
    @property
    def pre_pre_action(self) -> torch.Tensor:
        """The previous  previous actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._pre_pre_action
    
    
    ####
    
    def _reset_dofs_amp(self, reset_env_ids: torch.Tensor, frames: torch.Tensor, robot):
        """Reset the DOFs of the specified environments to the given frames.

        Args:
            reset_env_ids: The IDs of the environments to reset.
            frames: The frames to reset the DOFs to. Shape is (num_envs, num_dofs).
        """
        dof_pos = self.amp_loader.get_joint_pose_batch(frames)
        dof_vel = self.amp_loader.get_joint_vel_batch(frames)
        # let joint_ids handle the reindexing instead of gathering the data manually
        robot.write_joint_state_to_sim(
            dof_pos, dof_vel, joint_ids=self.data_joint_ids, env_ids=reset_env_ids
        )
        
        
        
        
    def _reset_root_states_amp(self, reset_env_ids: torch.Tensor, frames: torch.Tensor,robot):
        """Reset the root states of the specified environments to the given frames.

        Args:
            reset_env_ids: The IDs of the environments to reset.
            frames: The frames to reset the root states to. Shape is (num_envs, num_root_states).
        """
        
        
        # root pose (position + quaternion). Motion files store quaternions in xyzw order.
        env_origin = self.scene.env_origins[reset_env_ids]
        root_pos = AMPLoader.get_root_pos_batch(frames) + env_origin
        root_rot_wxyz = convert_quat(AMPLoader.get_root_rot_batch(frames), to="wxyz")
        root_pose = torch.cat([root_pos, root_rot_wxyz], dim=-1)
        robot.write_root_pose_to_sim(root_pose, env_ids=reset_env_ids)

        # root velocities are stored in the base's local frame -- rotate into world frame.
        lin_vel_w = quat_apply(root_rot_wxyz, AMPLoader.get_linear_vel_batch(frames))
        ang_vel_w = quat_apply(root_rot_wxyz, AMPLoader.get_angular_vel_batch(frames))
        robot.write_root_velocity_to_sim(torch.cat([lin_vel_w, ang_vel_w], dim=-1), env_ids=reset_env_ids)
    