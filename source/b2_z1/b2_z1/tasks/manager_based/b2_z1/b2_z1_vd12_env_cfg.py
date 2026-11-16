"""Isolated whole-gun exclusion-zone candidate; VD11 is preserved unchanged."""

from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from b2_z1.assets.vd12_contract import geometry_profile

from .b2_z1_vd11_env_cfg import B2Z1VD11EnvCfg, B2Z1VD11PlayEnvCfg
from .mdp.vd12_rewards import GunClearancePenalty, GunClearanceTermination, gun_exclusion_failure


def _configure_vd12(cfg):
    # Unlike contact-based checks this rule remains effective with historical
    # self-collisions disabled. It is a conservative exclusion zone, not fake
    # sensor force. Keep model, drive gains, commands and 70-D actor unchanged.
    cfg.vd12_geometry_profile = geometry_profile(cfg.scene.robot.spawn.asset_path)
    cfg.rewards.vd12_gun_clearance = RewTerm(
        func=GunClearancePenalty, weight=cfg.vd12_geometry_profile["clearance_reward_weight"],
    )
    cfg.rewards.vd12_geometry_failure = RewTerm(
        func=gun_exclusion_failure, weight=cfg.vd12_geometry_profile["failure_reward_weight"],
    )
    cfg.terminations.vd12_gun_exclusion = DoneTerm(func=GunClearanceTermination)


@configclass
class B2Z1VD12EnvCfg(B2Z1VD11EnvCfg):
    vd12_geometry_profile: dict = {}

    def __post_init__(self):
        super().__post_init__()
        _configure_vd12(self)


@configclass
class B2Z1VD12PlayEnvCfg(B2Z1VD11PlayEnvCfg):
    vd12_geometry_profile: dict = {}

    def __post_init__(self):
        super().__post_init__()
        _configure_vd12(self)
