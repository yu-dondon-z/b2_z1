"""VD9: physics-rate TCP jitter and pre-contact foot descent rewards."""

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from .b2_z1_vd5_startup_env_cfg import B2Z1VD5StartupEnvCfg, B2Z1VD5StartupPlayEnvCfg
from .mdp.vd9_rewards import PhysicsRateFootDescent, PhysicsRateTCPHighpass


def _configure_vd9(cfg):
    if cfg.scene.terrain.terrain_type != "plane" or cfg.sim.dt != 0.005 or cfg.decimation != 4:
        raise ValueError("VD9 requires flat terrain, 200 Hz physics, and a 50 Hz actor")
    feet = SceneEntityCfg(
        "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
    )
    contact = SceneEntityCfg(
        "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        preserve_order=True,
    )
    cfg.rewards.vd9_tcp_highpass_200hz = RewTerm(
        func=PhysicsRateTCPHighpass,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"), "command_name": "ee_target",
            "base_name": "base_link", "cutoff": 3.0, "deadzone": 0.001,
            "scale": 0.005, "active_after": 2.0,
        },
    )
    cfg.rewards.vd9_foot_descent_200hz = RewTerm(
        func=PhysicsRateFootDescent,
        weight=-2.0,
        params={
            "asset_cfg": feet, "sensor_cfg": contact, "band": 0.12,
            "speed_limit": 0.35, "scale": 0.50, "active_after": 2.0,
        },
    )


@configclass
class B2Z1VD9EnvCfg(B2Z1VD5StartupEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_vd9(self)


@configclass
class B2Z1VD9PlayEnvCfg(B2Z1VD5StartupPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_vd9(self)
