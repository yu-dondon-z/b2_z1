"""Isolated VB11 standby -> measured latch -> original curve/vision motion."""
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from .vision_trajectory_command import RearVisionTrajectoryCommand, RearVisionTrajectoryCommandCfg
from ..vb11_tool_control import gun_bottom_z, ready_mask, settle_latch


class VB11ReadyTrajectoryCommand(RearVisionTrajectoryCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not 0. <= cfg.vb11_standby_fraction <= 1.:
            raise ValueError("VB11 standby fraction must be within [0,1]")
        lo, hi = cfg.vb11_hold_range_s
        if not 0. <= lo <= hi < 12.:
            raise ValueError("VB11 hold range must be ordered and below the startup deadline")
        self._arm_ids = self.robot.find_joints([f"joint{i}" for i in range(1, 7)], preserve_order=True)[0]
        self._gun_id = self.robot.find_bodies("link_gun")[0][0]
        self._gun_sensor_id = env.scene["contact_forces"].find_bodies("link_gun")[0][0]
        self._episode_age_s = torch.zeros(self.num_envs, device=self.device)
        self._startup_wait_age_s = torch.zeros_like(self._episode_age_s)
        self._startup_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._settle_s = torch.zeros_like(self._episode_age_s)
        self._standby_only = torch.zeros_like(self._startup_ready)
        self._hold_until_s = torch.zeros_like(self._episode_age_s)
        self._capture_pending = torch.zeros_like(self._startup_ready)
        self._reset_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._episode_age_s[ids] = 0.
        self._startup_wait_age_s[ids] = 0.
        self._startup_ready[ids] = False
        self._settle_s[ids] = 0.
        self._standby_only[ids] = False
        self._hold_until_s[ids] = 0.
        self._capture_pending[ids] = True
        self._reset_step[ids] = self.env.common_step_counter
        if self.cfg.mode == "synthetic":
            self._standby_only[ids] = torch.rand(len(ids), device=self.device) < self.cfg.vb11_standby_fraction
            lo, hi = self.cfg.vb11_hold_range_s
            self._hold_until_s[ids] = lo + (hi - lo) * torch.rand(len(ids), device=self.device)

    def road_height_w(self):
        z = self.cfg.file_road_surface_z_w_m if self.cfg.mode == "file" else None
        if z is None:
            z = self.cfg.synthetic_ground_height_w_m
        return self.env.scene.env_origins[:, 2] + z

    def standby_position_w(self):
        base = self.robot.data.body_pos_w[:, self.base_idx]
        quat = self.robot.data.body_state_w[:, self.base_idx, 3:7]
        rear = torch.zeros_like(base)
        rear[:, 0] = sum(self.cfg.rear_work_x_range_m) * .5
        target = base + quat_apply(quat, rear)
        target[:, 2] = self.road_height_w() + self.cfg.tool_height_m
        return target

    def _measured_ready(self):
        target = self.standby_position_w()
        actual = self.robot.data.body_pos_w[:, self.body_idx]
        sensor = self.env.scene["contact_forces"]
        force = sensor.data.net_forces_w[:, self._gun_sensor_id].norm(dim=-1)
        return ready_mask(
            target - actual, target - self.tracking_pose_w[:, :3],
            self.robot.data.joint_vel[:, self._arm_ids].abs().amax(-1),
            -self.robot.data.projected_gravity_b[:, 2],
            self.robot.data.body_pos_w[:, self.base_idx, 2] - self.env.scene.env_origins[:, 2],
            force, gun_bottom_z(self.robot, self._gun_id) - self.road_height_w(),
            self.cfg.file_standby_position_tolerance_m, self.cfg.file_standby_height_tolerance_m)

    def _hold_file_tool_pose(self):
        # After readiness, preserve the original rear-cache hold behavior.
        if self._file_motion_ready:
            return super()._hold_file_tool_pose()
        base = self.robot.data.body_pos_w[:, self.base_idx]
        quat = self.robot.data.body_state_w[:, self.base_idx, 3:7]
        self._set_file_pose_command_b_smooth(quat_apply_inverse(quat, self.standby_position_w() - base))

    def _gate_file_startup(self):
        if self._file_motion_ready:
            return True
        self._hold_file_tool_pose()
        self._file_settle_elapsed = (self._file_settle_elapsed + self.env.step_dt
                                     if bool(self._measured_ready().all()) else 0.)
        if self._file_settle_elapsed >= self.cfg.file_standby_settle_s:
            self._file_motion_ready = True
            self._command_transition_elapsed.zero_()
        return self._file_motion_ready

    def _update_command(self):
        self._episode_age_s += self.env.step_dt
        self._startup_wait_age_s += self.env.step_dt
        self._capture_after_reset_physics()
        previous_pose = self.pose_command.clone()
        previous_world = self.tracking_pose_w.clone()
        super()._update_command()
        if self.cfg.mode == "file":
            return
        blocked = ~self._startup_ready | self._standby_only | (self._episode_age_s < self._hold_until_s)
        # Keep the moving environments' original truth-curve target. Waiting
        # environments use exactly the same world-height filter as vision.
        desired_b = self.pose_command[:, :3].clone()
        base = self.robot.data.body_pos_w[:, self.base_idx]
        quat = self.robot.data.body_state_w[:, self.base_idx, 3:7]
        standby_b = quat_apply_inverse(quat, self.standby_position_w() - base)
        desired_b[blocked] = standby_b[blocked]
        self.pose_command[blocked] = previous_pose[blocked]
        self.tracking_pose_w[blocked] = previous_world[blocked]
        self._set_file_pose_command_b_smooth(desired_b)
        self.base_velocity_command_b[blocked] = 0.
        self._filtered_velocity_command_b[blocked] = 0.
        self._command_transition_elapsed[blocked] = 0.
        before = self._startup_ready.clone()
        self._settle_s, self._startup_ready = settle_latch(
            self._measured_ready() & (self._episode_age_s >= self._hold_until_s),
            self._settle_s, before, self.env.step_dt, self.cfg.file_standby_settle_s)
        # Release only after both measured settling and sampled zero-speed
        # hold. Capture the *command*, never the biased measured TCP, once.
        release = blocked & self._startup_ready & ~self._standby_only & (self._episode_age_s >= self._hold_until_s)
        self._command_transition_start_position_b[release] = self.pose_command[release, :3]
        self._command_transition_start_velocity_b[release] = 0.

    def _capture_after_reset_physics(self):
        # Link transforms at command reset can still represent the old arm
        # pose. Wait for a genuine physics update, then capture ONCE. Never
        # re-anchor a live target to the biased measured TCP each step.
        ids = (self._capture_pending & (self.env.common_step_counter > self._reset_step)).nonzero().flatten()
        if len(ids):
            self._hold_current_tool_pose(ids)
            self._command_transition_start_position_b[ids] = self.pose_command[ids, :3]
            self._file_hold_position_b[ids] = self.pose_command[ids, :3]
            self._command_transition_elapsed[ids] = 0.
            self._capture_pending[ids] = False

    def _handle_file_command_loss(self):
        was_ready = self._file_motion_ready
        super()._handle_file_command_loss()
        if was_ready and not self._file_motion_ready:
            self._startup_wait_age_s.zero_()


@configclass
class VB11ReadyTrajectoryCommandCfg(RearVisionTrajectoryCommandCfg):
    class_type: type = VB11ReadyTrajectoryCommand
    vb11_standby_fraction: float = .20
    vb11_hold_range_s: tuple[float, float] = (4., 7.)
