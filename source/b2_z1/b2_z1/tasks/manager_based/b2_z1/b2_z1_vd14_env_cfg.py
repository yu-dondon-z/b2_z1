"""Posture-only candidate; VD13 assets, drives, actor and AMP stay unchanged."""
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from b2_z1.assets.vd14_contract import posture_profile

from .b2_z1_vd13_env_cfg import B2Z1VD13EnvCfg, B2Z1VD13PlayEnvCfg
from .mdp.vd14_rewards import front_stance_width_penalty


def _configure_vd14(cfg):
    profile = posture_profile()
    cfg.rewards.base_tilt = cfg.rewards.base_tilt.replace(
        weight=profile["base_tilt_weight"],
        params={**cfg.rewards.base_tilt.params,
                "deadzone": profile["base_tilt_deadzone_rad"],
                "scale": profile["base_tilt_scale_rad"]},
    )
    cfg.rewards.vd14_front_width = RewTerm(
        func=front_stance_width_penalty, weight=profile["front_width_weight"],
        params={"asset_cfg": SceneEntityCfg("robot", body_names=profile["front_body_names"],
                                            preserve_order=True),
                "minimum_width": profile["front_min_width_m"],
                "scale": profile["front_width_scale_m"]},
    )
    cfg.vd14_posture_profile = profile


@configclass
class B2Z1VD14EnvCfg(B2Z1VD13EnvCfg):
    vd14_posture_profile: dict = {}

    def __post_init__(self):
        super().__post_init__()
        _configure_vd14(self)


@configclass
class B2Z1VD14PlayEnvCfg(B2Z1VD13PlayEnvCfg):
    vd14_posture_profile: dict = {}

    def __post_init__(self):
        super().__post_init__()
        _configure_vd14(self)
