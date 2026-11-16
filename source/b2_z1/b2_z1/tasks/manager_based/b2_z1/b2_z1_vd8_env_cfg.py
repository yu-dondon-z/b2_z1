"""VD8: VD5 startup with physical roll-pulse training and contact-onset force."""

from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from .b2_z1_vd5_startup_env_cfg import B2Z1VD5StartupEnvCfg, B2Z1VD5StartupPlayEnvCfg
from .mdp.vd8_events import RollPulse
from .mdp.vd8_rewards import LandingOnsetForce


def _configure_vd8(cfg, *, play=False):
    if cfg.scene.terrain.terrain_type != 'plane' or cfg.sim.dt != .005 or cfg.decimation != 4:
        raise ValueError('VD8 requires 200 Hz physics, 50 Hz actor and flat terrain')
    feet = SceneEntityCfg('contact_forces', body_names=['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'],
                          preserve_order=True)
    cfg.rewards.vd8_touchdown_force = RewTerm(func=LandingOnsetForce, weight=-.2,
        params={'sensor_cfg': feet, 'threshold': 300., 'scale': 200.})
    if not play:
        cfg.events.vd8_roll_pulse = EventTerm(func=RollPulse, mode='interval',
            interval_range_s=(.02, .02), is_global_time=True,
            params={'asset_cfg': SceneEntityCfg('robot', body_names=['base_link']),
                    'min_torque': 4., 'max_torque': 10., 'duration_s': .12})


@configclass
class B2Z1VD8EnvCfg(B2Z1VD5StartupEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_vd8(self)


@configclass
class B2Z1VD8PlayEnvCfg(B2Z1VD5StartupPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_vd8(self, play=True)
