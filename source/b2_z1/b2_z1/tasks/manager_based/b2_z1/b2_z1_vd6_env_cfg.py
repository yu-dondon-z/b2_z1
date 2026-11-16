"""VD6: targeted continuation of VD5, with unchanged policy IO and commands."""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from .b2_z1_vd5_startup_env_cfg import B2Z1VD5StartupEnvCfg, B2Z1VD5StartupPlayEnvCfg
from .mdp.vd6_rewards import feet_soft_landing, feet_impact_excess, arm_action_continuity


def _configure_vd6(cfg):
    if cfg.scene.terrain.terrain_type != 'plane':
        raise ValueError('VD6 landing height currently requires plane terrain')
    if cfg.actions.JointPositionAction.scale != .25:
        raise ValueError('VD6 continuity reward assumes the original 0.25 rad action scale')
    feet = SceneEntityCfg('robot', body_names=['FL_foot','FR_foot','RL_foot','RR_foot'], preserve_order=True)
    contact = SceneEntityCfg('contact_forces', body_names=['FL_foot','FR_foot','RL_foot','RR_foot'], preserve_order=True)
    arm = SceneEntityCfg('robot', joint_names=[f'joint{i}' for i in range(1,7)])
    cfg.rewards.vd6_soft_landing = RewTerm(func=feet_soft_landing, weight=-1.,
        params={'asset_cfg': feet, 'sensor_cfg': contact, 'band': .10, 'speed_limit': .35})
    cfg.rewards.vd6_impact_excess = RewTerm(func=feet_impact_excess, weight=-.10,
        params={'sensor_cfg': contact, 'threshold': 350., 'scale': 200.})
    cfg.rewards.vd6_arm_slew = RewTerm(func=arm_action_continuity, weight=-.10,
        params={'asset_cfg': arm, 'deadzone': .02, 'scale': .05})
    cfg.rewards.vd6_arm_second_difference = RewTerm(func=arm_action_continuity, weight=-.05,
        params={'asset_cfg': arm, 'second_order': True, 'deadzone': .03, 'scale': .06})


@configclass
class B2Z1VD6EnvCfg(B2Z1VD5StartupEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_vd6(self)


@configclass
class B2Z1VD6PlayEnvCfg(B2Z1VD5StartupPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_vd6(self)
