"""VD11 candidate: world-frame TCP jitter and measured touchdown-force costs.

The VD10 hardware, safe reset, real contacts, command curriculum, observations,
tracking rewards and all drive/physics settings are intentionally inherited.
"""

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from .b2_z1_vd10_env_cfg import B2Z1VD10EnvCfg, B2Z1VD10PlayEnvCfg
from .mdp.vd11_rewards import PhysicsRateFootImpact, PhysicsRateWorldTCPHighpass


def _configure_vd11(cfg):
    if cfg.scene.terrain.terrain_type != "plane" or cfg.sim.dt != 0.005 or cfg.decimation != 4:
        raise ValueError("VD11 requires flat terrain, 200 Hz physics, and a 50 Hz actor")
    feet = SceneEntityCfg(
        "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
    )
    contact = SceneEntityCfg(
        "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        preserve_order=True,
    )
    # A zero weight still constructs the old term and its physics callback.
    # Remove it in VD11 only, rather than accumulating unused held-target costs.
    cfg.rewards.vd9_tcp_highpass_200hz = None
    cfg.rewards.vd11_world_tcp_highpass_200hz = RewTerm(
        func=PhysicsRateWorldTCPHighpass,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # Measured 0.5--3 Hz lateral motion is larger than >3 Hz motion.
            # A two-stage 1 Hz filter includes more gait-band motion while
            # rejecting uniform-speed travel; this is not a hard cutoff.
            "base_name": "base_link", "cutoff": 1.0, "deadzone": 0.001,
            "scale": 0.005, "lateral_weight": 2.0, "active_after": 2.0,
        },
    )
    cfg.rewards.vd11_foot_impact_200hz = RewTerm(
        func=PhysicsRateFootImpact,
        weight=-0.5,
        params={
            "asset_cfg": feet, "sensor_cfg": contact, "contact_threshold": 10.0,
            "airborne_duration": 0.02, "window_s": 0.05,
            # Actual deterministic VD10 p95 landing peaks are 270--404 N;
            # 804 N would miss nearly all of the observed hard landings.
            # 300 N is ~1.5 times equal four-foot static load; it is a soft
            # training target, not a physical load rating or a termination.
            "force_limit": 300.0, "force_scale": 200.0, "active_after": 2.0,
        },
    )


@configclass
class B2Z1VD11EnvCfg(B2Z1VD10EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_vd11(self)


@configclass
class B2Z1VD11PlayEnvCfg(B2Z1VD10PlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_vd11(self)
