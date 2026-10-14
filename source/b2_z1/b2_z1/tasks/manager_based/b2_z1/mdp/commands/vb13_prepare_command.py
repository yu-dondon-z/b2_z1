"""VB13 uses measured release in BOTH training and deterministic validation."""
from isaaclab.utils import configclass
from .vb12_schedule_command import VB12ScheduleTrajectoryCommand, VB12ScheduleTrajectoryCommandCfg


class VB13PrepareTrajectoryCommand(VB12ScheduleTrajectoryCommand):
    def __init__(self, cfg, env):
        if cfg.vb12_training_schedule:
            raise ValueError('VB13 requires the measured preparation gate in training and play')
        super().__init__(cfg, env)

    def _update_command(self):
        before = self._startup_ready.clone()
        super()._update_command()
        self._success_event[:] = ~before & self._startup_ready
        check = ~self._deadline_checked & (self._episode_age_s >= 10.)
        self._failure_event[:] = check & ~self._startup_ready
        self._deadline_checked |= check


@configclass
class VB13PrepareTrajectoryCommandCfg(VB12ScheduleTrajectoryCommandCfg):
    class_type: type = VB13PrepareTrajectoryCommand
    vb12_training_schedule: bool = False
