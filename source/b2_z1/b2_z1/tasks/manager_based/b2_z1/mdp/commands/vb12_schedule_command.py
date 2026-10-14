"""VB12 external training schedule, strict evaluation, world-height cache hold."""
import math
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from .vb11_ready_command import VB11ReadyTrajectoryCommand, VB11ReadyTrajectoryCommandCfg
from .vision_trajectory_command import RearVisionTrajectoryCommand
from ..vb11_tool_control import settle_latch


def scheduled_standby(age, release_time, standby_only):
    return standby_only | (age < release_time)


class VB12ScheduleTrajectoryCommand(VB11ReadyTrajectoryCommand):
    def __init__(self, cfg, env):
        lo, hi = cfg.vb12_warm_hold_range_s
        if not all(math.isfinite(v) for v in (lo, hi, cfg.vb12_ready_deadline_after_hold_s)) or not (
                0. <= lo <= hi and cfg.vb12_ready_deadline_after_hold_s > 0.):
            raise ValueError("VB12 preparation/deadline durations must be finite and ordered")
        super().__init__(cfg, env)
        self._success_event = torch.zeros_like(self._startup_ready)
        self._failure_event = torch.zeros_like(self._startup_ready)
        self._deadline_checked = torch.zeros_like(self._startup_ready)
        self._phase_moving = torch.zeros_like(self._startup_ready)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._success_event[ids] = False
        self._failure_event[ids] = False
        self._deadline_checked[ids] = False
        self._phase_moving[ids] = False
        if self.cfg.vb12_training_schedule:
            # Warm-start motion domains enter the same smooth velocity ramp
            # early; nominal cold-start domains retain a longer preparation.
            warm = torch.rand(len(ids), device=self.device) < .50
            lo, hi = self.cfg.vb12_warm_hold_range_s
            self._hold_until_s[ids[warm]] = lo + (hi - lo) * torch.rand(int(warm.sum()), device=self.device)

    def _hold_file_tool_pose(self):
        if not self._file_motion_ready:
            return super()._hold_file_tool_pose()
        # Preserve the last body-relative XY command until a real cached rear
        # point is available. NEVER hold Z in the body frame after readiness.
        base = self.robot.data.body_pos_w[:, self.base_idx]
        quat = self.robot.data.body_state_w[:, self.base_idx, 3:7]
        desired_w = base + quat_apply(quat, self._file_hold_position_b)
        desired_w[:, 2] = self.road_height_w() + self.cfg.tool_height_m
        self._set_file_pose_command_b_smooth(quat_apply_inverse(quat, desired_w - base))

    def _update_command(self):
        if not self.cfg.vb12_training_schedule:
            # Validation is still measured-gate controlled, including reset
            # and stale recovery. This flag is enabled ONLY for training.
            return super()._update_command()
        self._episode_age_s += self.env.step_dt
        self._startup_wait_age_s += self.env.step_dt
        self._capture_after_reset_physics()
        previous_pose = self.pose_command.clone()
        previous_world = self.tracking_pose_w.clone()
        was_blocked = ~self._phase_moving
        # Skip VB11's performance-dependent command gate, not its collision,
        # load model, action filters or termination safety checks.
        RearVisionTrajectoryCommand._update_command(self)
        blocked = scheduled_standby(self._episode_age_s, self._hold_until_s, self._standby_only)
        self._phase_moving[:] = ~blocked
        desired_b = self.pose_command[:, :3].clone()
        base = self.robot.data.body_pos_w[:, self.base_idx]
        quat = self.robot.data.body_state_w[:, self.base_idx, 3:7]
        desired_b[blocked] = quat_apply_inverse(quat, self.standby_position_w() - base)[blocked]
        self.pose_command[blocked] = previous_pose[blocked]
        self.tracking_pose_w[blocked] = previous_world[blocked]
        self._set_file_pose_command_b_smooth(desired_b)
        self.base_velocity_command_b[blocked] = 0.
        self._filtered_velocity_command_b[blocked] = 0.
        self._command_transition_elapsed[blocked] = 0.
        released = was_blocked & ~blocked
        # Start the path/TCP ramp at the actual previous standby command. Do
        # not expose one parent-curve step based on the old reset pose.
        if released.any():
            self._command_transition_start_position_b[released] = previous_pose[released, :3]
            self._command_transition_start_velocity_b[released] = 0.
            self.pose_command[released] = previous_pose[released]
            self.tracking_pose_w[released] = previous_world[released]
            self._command_transition_elapsed[released] = 0.
            self.base_velocity_command_b[released] = 0.
            self._filtered_velocity_command_b[released] = 0.
        before = self._startup_ready.clone()
        self._settle_s, self._startup_ready = settle_latch(
            self._measured_ready(), self._settle_s, before, self.env.step_dt, self.cfg.file_standby_settle_s)
        self._success_event[:] = ~before & self._startup_ready
        deadline = self._hold_until_s + self.cfg.vb12_ready_deadline_after_hold_s
        check = ~self._deadline_checked & (self._episode_age_s >= deadline)
        self._failure_event[:] = check & ~self._startup_ready
        self._deadline_checked |= check


@configclass
class VB12ScheduleTrajectoryCommandCfg(VB11ReadyTrajectoryCommandCfg):
    class_type: type = VB12ScheduleTrajectoryCommand
    vb12_training_schedule: bool = False
    vb12_warm_hold_range_s: tuple[float, float] = (1., 2.)
    vb12_ready_deadline_after_hold_s: float = 3.
