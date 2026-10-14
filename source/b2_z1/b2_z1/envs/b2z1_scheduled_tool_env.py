"""VB12 scheduled-motion telemetry layered on unchanged VB11 physics."""
from .b2z1_ready_tool_env import B2Z1ReadyToolAMPEnv


class B2Z1ScheduledToolAMPEnv(B2Z1ReadyToolAMPEnv):
    def step(self, action):
        result = super().step(action)
        term = self.command_manager.get_term("ee_target")
        cmd = term.base_velocity_command_b[:, 0].abs() > .05
        actual = self.scene["robot"].data.root_lin_vel_b[:, 0] > .05
        expected = term._phase_moving if term.cfg.vb12_training_schedule else cmd
        road_z = term.road_height_w() + term.cfg.tool_height_m
        log = self.extras.setdefault("log", {})
        log.update({
            "VB12/scheduled_motion_fraction": expected.float().mean(),
            "VB12/requested_motion_fraction": cmd.float().mean(),
            "VB12/actual_commanded_motion_fraction": (actual & cmd).float().mean(),
            "VB12/movement_given_request": (actual & cmd).float().sum() / cmd.sum().clamp_min(1),
            "VB12/preparation_success_events": term._success_event.float().sum(),
            "VB12/preparation_failure_events": term._failure_event.float().sum(),
            "VB12/world_height_command_error_m": (term.tracking_pose_w[:, 2] - road_z).abs().mean(),
        })
        if hasattr(self, "_vb11_nominal_load_mask") and self._vb11_nominal_load_mask.any():
            mask = self._vb11_nominal_load_mask
            log["VB12/full_load_scheduled_motion_fraction"] = expected[mask].float().mean()
            log["VB12/full_load_requested_motion_fraction"] = cmd[mask].float().mean()
            log["VB12/full_load_actual_commanded_motion_fraction"] = (actual[mask] & cmd[mask]).float().mean()
        return result
