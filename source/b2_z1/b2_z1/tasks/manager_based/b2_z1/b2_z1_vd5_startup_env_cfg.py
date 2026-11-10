"""Separate VD5 adaptation task covering the nominal stand-to-policy handoff."""

from isaaclab.utils import configclass

from .b2_z1_vd5_env_cfg import B2Z1VD5EnvCfg, B2Z1VD5PlayEnvCfg
from .mdp.commands.vd5_startup_command import VD5StartupCurveCommand, VD5StartupSeamCommand


def _configure_startup(cfg, *, play=False):
    # The configured 0.70 m spawn leaves the default leg pose unsupported.
    # Probing shows 0.42 m penetrates the calf collision; 0.50 m clears it.
    cfg.vd5_nominal_base_height_m = 0.50
    cfg.scene.robot.init_state.pos = (*cfg.scene.robot.init_state.pos[:2], cfg.vd5_nominal_base_height_m)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.reference_state_initialization = not play
    # Keep most resets on the learned AMP gait; reserve one in five for nominal startup.
    cfg.reference_state_initialization_prob = 0.8
    cfg.commands.base_velocity.class_type = VD5StartupCurveCommand
    cfg.commands.base_velocity.vd5_startup_duration_s = 2.0
    cfg.commands.ee_target.class_type = VD5StartupSeamCommand
    cfg.commands.ee_target.vd5_startup_duration_s = 2.0


@configclass
class B2Z1VD5StartupEnvCfg(B2Z1VD5EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_startup(self)


@configclass
class B2Z1VD5StartupPlayEnvCfg(B2Z1VD5PlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_startup(self, play=True)
