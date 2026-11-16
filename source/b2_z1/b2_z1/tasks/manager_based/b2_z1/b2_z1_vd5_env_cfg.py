"""VD5: world-generated seam targets transformed into the moving base frame."""

from isaaclab.utils import configclass

from . import mdp
from .b2_z1_vd4_env_cfg import B2Z1VD4EnvCfg, B2Z1VD4PlayEnvCfg
from .mdp.commands.vd5_world_seam_command import VD5WorldSeamCommand


def _configure_vd5(cfg, *, play=False):
    # One physical seam per episode. Replacing it midway would invent a new
    # rear target before the TCP has traversed the old part of the seam.
    path_duration = cfg.episode_length_s + 1.0
    cfg.commands.base_velocity.resampling_time_range = (path_duration, path_duration)
    cfg.commands.ee_target.class_type = VD5WorldSeamCommand
    cfg.commands.ee_target.curriculum_enabled = False
    cfg.commands.ee_target.vd5_seam_trailing_distance = 0.70
    cfg.commands.ee_target.vd5_ground_transition_steps = 0 if play else 100_000
    cfg.rewards.end_effector_orientation_error.func = mdp.tcp_gun_axis_command_error
    # VD3 flange terms assume full TCP attitude and the nominal Bezier derivative;
    # neither defines VD5's seam target. Keep joint-speed regularization instead.
    for name in ("flange_position_residual", "flange_linear_velocity_residual",
                 "flange_angular_velocity_residual"):
        getattr(cfg.rewards, name).weight = 0.0


@configclass
class B2Z1VD5EnvCfg(B2Z1VD4EnvCfg):
    """Training task with seam-derived EE commands and retained camera assets."""

    def __post_init__(self):
        super().__post_init__()
        _configure_vd5(self)


@configclass
class B2Z1VD5PlayEnvCfg(B2Z1VD4PlayEnvCfg):
    """Play task retaining camera assets; EE commands do not consume camera data."""

    def __post_init__(self):
        super().__post_init__()
        _configure_vd5(self, play=True)
