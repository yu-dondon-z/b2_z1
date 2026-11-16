"""750mm clamp-to-tip / .95m work distance; unchanged VD12 objectives/drives."""
from isaaclab.utils import configclass
from b2_z1.assets.unitree import B2_Z1_MOUNT75_CFG
from b2_z1.assets.vd12_contract import geometry_profile
from b2_z1.assets.vd13_contract import mount_profile, WORK_DISTANCE_M
from .b2_z1_vd12_env_cfg import B2Z1VD12EnvCfg, B2Z1VD12PlayEnvCfg
from .mdp.commands.vd10_safe_seam_command import VD10SafeSeamCommand


class VD13SafeSeamCommand(VD10SafeSeamCommand):
    vd10_startup_profile = dict(VD10SafeSeamCommand.vd10_startup_profile,
                              mount_version=13, tcp_in_link06_m=[.825, 0., .1],
                              clamp_to_tip_m=.75)


def _configure_vd13(cfg):
    # Preserve VD12 reset joints/defaults and drive settings, replacing only spawn.
    cfg.scene.robot = cfg.scene.robot.replace(spawn=B2_Z1_MOUNT75_CFG.spawn.replace())
    cfg.commands.ee_target.class_type = VD13SafeSeamCommand
    cfg.commands.ee_target.vd5_seam_trailing_distance = WORK_DISTANCE_M
    cfg.vd12_geometry_profile = geometry_profile(cfg.scene.robot.spawn.asset_path)
    cfg.vd13_mount_profile = mount_profile()


@configclass
class B2Z1VD13EnvCfg(B2Z1VD12EnvCfg):
    vd13_mount_profile: dict = {}

    def __post_init__(self):
        super().__post_init__()
        _configure_vd13(self)


@configclass
class B2Z1VD13PlayEnvCfg(B2Z1VD12PlayEnvCfg):
    vd13_mount_profile: dict = {}

    def __post_init__(self):
        super().__post_init__()
        _configure_vd13(self)
