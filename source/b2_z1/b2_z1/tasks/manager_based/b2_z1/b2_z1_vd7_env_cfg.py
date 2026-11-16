"""VD7: short continuation from VD5, actual TCP error and worst-foot impact."""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from .b2_z1_vd5_startup_env_cfg import B2Z1VD5StartupEnvCfg, B2Z1VD5StartupPlayEnvCfg
from .mdp.vd6_rewards import feet_soft_landing
from .mdp.vd7_rewards import TCPErrorHighpass, worst_foot_impact


def _configure_vd7(cfg):
    if cfg.scene.terrain.terrain_type != 'plane':
        raise ValueError('VD7 landing reward requires plane terrain')
    feet = SceneEntityCfg('robot',body_names=['FL_foot','FR_foot','RL_foot','RR_foot'],preserve_order=True)
    contact = SceneEntityCfg('contact_forces',body_names=['FL_foot','FR_foot','RL_foot','RR_foot'],preserve_order=True)
    cfg.rewards.vd7_tcp_highpass = RewTerm(func=TCPErrorHighpass,weight=-1.,
        params={'command_name':'ee_target','cutoff':3.,'deadzone':.002,'scale':.01})
    cfg.rewards.vd7_soft_landing = RewTerm(func=feet_soft_landing,weight=-1.,
        params={'asset_cfg':feet,'sensor_cfg':contact,'band':.10,'speed_limit':.35})
    cfg.rewards.vd7_worst_impact = RewTerm(func=worst_foot_impact,weight=-.2,
        params={'sensor_cfg':contact,'threshold':350.,'scale':200.})


@configclass
class B2Z1VD7EnvCfg(B2Z1VD5StartupEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_vd7(self)


@configclass
class B2Z1VD7PlayEnvCfg(B2Z1VD5StartupPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_vd7(self)
