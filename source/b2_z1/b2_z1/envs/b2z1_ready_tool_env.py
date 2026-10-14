"""VB11 load quotas/diagnostics; original gun/hose dynamics and actor actions."""
import torch
from .b2z1_gun_clearance_env import B2Z1GunClearanceAMPEnv


class B2Z1ReadyToolAMPEnv(B2Z1GunClearanceAMPEnv):
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # Reset events can install the clear-arm pose before ActionManager
        # clears its EMA. Seed once AFTER all manager resets, from actual q.
        term = self.action_manager.get_term("JointPositionAction")
        if term._alpha != .20 or term._scale != .25:
            raise RuntimeError("VB11 preserves the original EMA=.20/action scale=.25")
        robot = self.scene["robot"]
        if len(term._joint_names) != 18 or list(term._joint_names) != list(robot.joint_names):
            raise RuntimeError("VB11 reset seeding requires all 18 joints in articulation order")
        q = robot.data.joint_pos[env_ids]
        offset = term._offset if term._offset.ndim == 1 else term._offset[env_ids]
        normalized = (q - offset) / term._scale
        term._raw_actions[env_ids] = normalized
        term._processed_actions[env_ids] = q
        self.action_manager._action[env_ids] = normalized
        self.action_manager._prev_action[env_ids] = normalized

    def _sample_hose_parameters(self, env_ids):
        super()._sample_hose_parameters(env_ids)
        if hasattr(self, "_vb11_nominal_load_mask"):
            nominal_ids = env_ids[self._vb11_nominal_load_mask[env_ids]]
            self._hose_mass_kg[nominal_ids] = self.cfg.vb11_nominal_hose_mass_kg
            self._hose_drag[nominal_ids] = 4.

    def step(self, action):
        result = super().step(action)
        term = self.command_manager.get_term("ee_target")
        ready = (term._startup_ready if term.cfg.mode == "synthetic"
                 else torch.full_like(term._startup_ready, term._file_motion_ready))
        error_z = self.scene["robot"].data.body_pos_w[:, term.body_idx, 2] - term.tracking_pose_w[:, 2]
        log = self.extras.setdefault("log", {})
        log.update({
            "VB11/startup_ready_fraction": ready.float().mean(),
            "VB11/standby_only_fraction": term._standby_only.float().mean(),
            "VB11/tcp_height_error_mean_m": error_z.mean(),
            "VB11/tcp_height_error_abs_mean_m": error_z.abs().mean(),
            "VB11/commanded_moving_fraction": (term.base_velocity_command_b[:, 0].abs() > .05).float().mean(),
            "VB11/actual_moving_fraction": (self.scene["robot"].data.root_lin_vel_b[:, 0].abs() > .05).float().mean(),
            "VB11/qualified_moving_fraction": ((self.scene["robot"].data.root_lin_vel_b[:, 0].abs() > .05)
                & (term.base_velocity_command_b[:, 0].abs() > .05) & ready).float().mean(),
        })
        if hasattr(self, "_vb11_nominal_load_mask"):
            mask = self._vb11_nominal_load_mask
            log["VB11/nominal_full_load_fraction"] = mask.float().mean()
            if mask.any():
                log.update({
                    "VB11/full_load_ready_fraction": ready[mask].float().mean(),
                    "VB11/full_load_tcp_height_abs_error_m": error_z[mask].abs().mean(),
                    "VB11/full_load_gun_contact_fraction": (self.gun_contact_force_w[mask].norm(dim=-1) > 1.).float().mean(),
                    "VB11/full_load_support_force_mean_n": self.gun_contact_force_w[mask, 2].clamp_min(0).mean(),
                })
        if hasattr(self, "_vb11_contact_debt_s"):
            log["VB11/contact_debt_mean_s"] = self._vb11_contact_debt_s.mean()
        return result
