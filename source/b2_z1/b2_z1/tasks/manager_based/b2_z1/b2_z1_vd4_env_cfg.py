"""VD4 environment using the measured B2 front RGB camera geometry.

VD3 defaults stay untouched. This task changes only the front camera model used
by the ideal geometric path command generator; it does not add RGB perception.
"""
from isaaclab.utils import configclass

from .b2_z1_env_cfg import B2Z1EnvCfg, B2Z1PlayEnvCfg
from .mdp.commands.curve_camera import make_curve_camera_cfg


def _apply_front_rgb_calibration(cfg, *, render):
    command = cfg.commands.base_velocity
    # Calibration from this robot, 2026-09-15. T_base_camera is converted from
    # optical (right, down, forward) to base (forward, left, up) coordinates.
    command.camera_position_b = (0.44643878, -0.00569431, -0.00031663)
    command.camera_rpy_b = (-0.01715695, 0.02446081, 0.00610092)
    # 1920x1080 calibration scaled by 1/3. Isaac's pinhole renderer requires
    # square pixels and a centered principal point; this matches FOV closely.
    command.camera_intrinsics = (251.47, 251.47, 320.0, 180.0)
    command.camera_image_size = (640, 360)
    cfg.scene.base_camera = make_curve_camera_cfg(command, render=render)


@configclass
class B2Z1VD4EnvCfg(B2Z1EnvCfg):
    """VD4 training environment; retains the VD3 task and policy contract."""

    def __post_init__(self):
        super().__post_init__()
        _apply_front_rgb_calibration(self, render=False)


@configclass
class B2Z1VD4PlayEnvCfg(B2Z1PlayEnvCfg):
    """VD4 Play configuration with the same calibrated geometric camera."""

    def __post_init__(self):
        super().__post_init__()
        _apply_front_rgb_calibration(self, render=True)
