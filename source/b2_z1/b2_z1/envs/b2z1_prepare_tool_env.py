"""Preparation telemetry, unchanged gun/hose physics and student actuation."""
from .b2z1_scheduled_tool_env import B2Z1ScheduledToolAMPEnv


class B2Z1PrepareToolAMPEnv(B2Z1ScheduledToolAMPEnv):
    def step(self, action):
        result = super().step(action)
        term = self.command_manager.get_term('ee_target')
        log = self.extras.setdefault('log', {})
        log['VB13/stage'] = self.cfg.vb13_stage
        log['VB13/hold_ready_fraction'] = term._measured_ready().float().mean()
        log['VB13/target_height_m'] = term.cfg.tool_height_m
        log['VB13/effective_hose_mass_kg'] = self._hose_mass_kg.mean()
        return result
