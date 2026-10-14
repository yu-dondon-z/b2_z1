"""VB12 isolated repair: exogenous training phases, unchanged deployment gate."""
import copy
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from .b2_z1_vb11_env_cfg import (B2Z1VB11ReadyAMPEnvCfg, VB11ReadyRewardsCfg,
                               B2Z1VB11ReadyAMPEnvCfg_Play, B2Z1VB11ReadyVisionAMPEnvCfg_Play)
from .mdp.commands.vb12_schedule_command import VB12ScheduleTrajectoryCommandCfg, VB12ScheduleTrajectoryCommand
from .mdp import vb12_task_signals as signals


@configclass
class VB12ScheduleRewardsCfg(VB11ReadyRewardsCfg):
    tcp_dense_error = RewTerm(func=signals.dense_tcp_error, weight=-1.5)
    # Isaac rewards are multiplied by dt=.02: once-only +3 / -5, not
    # tiny +.06 / -.10 signals drowned by normal per-step reward.
    preparation_success = RewTerm(func=signals.preparation_success, weight=150.)
    preparation_failure = RewTerm(func=signals.preparation_failure, weight=-250.)
    physical_failure = RewTerm(func=signals.physical_failure, weight=-500.)


def configure_schedule(cfg, training=False):
    command = VB12ScheduleTrajectoryCommandCfg()
    for key, value in vars(cfg.commands.ee_target).items():
        setattr(command, key, copy.deepcopy(value))
    command.class_type = VB12ScheduleTrajectoryCommand
    command.vb12_training_schedule = training
    cfg.commands.ee_target = command
    if training:
        if command.mode != "synthetic" or not command.synthetic_world_fixed_path:
            raise ValueError("VB12 external schedule requires world-fixed synthetic training")
        # Failure is now explicit reward/telemetry, not endless task denial
        # followed by another zero-speed episode. All physical checks stay.
        cfg.terminations.startup_not_ready = None
        command.vb11_standby_fraction = .15
        command.vb11_hold_range_s = (4., 6.)


@configclass
class B2Z1VB12ScheduleAMPEnvCfg(B2Z1VB11ReadyAMPEnvCfg):
    rewards: VB12ScheduleRewardsCfg = VB12ScheduleRewardsCfg()
    vb9_arm_guide_weight: float = .12
    vb12_arm_noise_cap: float = .10
    vb12_arm_initial_std: float = .08
    vb12_leg_guide_weight: float = .10

    def __post_init__(self):
        super().__post_init__()
        configure_schedule(self, training=True)
        # Half legacy cold starts, half clear starts. No default/normalization
        # change, no fabricated arm experts and no runtime teacher.
        self.events.vb9_pose_reset.params["fraction"] = .50


@configclass
class B2Z1VB12ScheduleAMPEnvCfg_Play(B2Z1VB11ReadyAMPEnvCfg_Play):
    def __post_init__(self):
        super().__post_init__()
        configure_schedule(self)


@configclass
class B2Z1VB12ScheduleVisionAMPEnvCfg_Play(B2Z1VB11ReadyVisionAMPEnvCfg_Play):
    def __post_init__(self):
        super().__post_init__()
        configure_schedule(self)
