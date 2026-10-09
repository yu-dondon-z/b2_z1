"""VB5: second-stage effective hose mass curriculum under free payout."""

from isaaclab.utils import configclass

from .b2_z1_vb4_env_cfg import B2Z1VB4PayoutAMPEnvCfg, B2Z1VB4PayoutAMPEnvCfg_Play
from .b2_z1_va15_vision_env_cfg import B2Z1VA15LoadedGunVisionAMPEnvCfg_Play


@configclass
class B2Z1VB5PayoutStage2AMPEnvCfg(B2Z1VB4PayoutAMPEnvCfg):
    """Continue the screened VB4 gait with 1-6 kg effective hose mass."""

    def __post_init__(self):
        super().__post_init__()
        self.hose_mass_range_kg = (1.0, 6.0)


@configclass
class B2Z1VB5PayoutStage2AMPEnvCfg_Play(B2Z1VB4PayoutAMPEnvCfg_Play):
    """Fixed mass (B2_Z1_HOSE_MASS_KG) for equal-condition playback."""

    def __post_init__(self):
        super().__post_init__()
        # Keep TCP/path markers, avoiding remote USD assets for velocity arrows.
        self.commands.base_velocity.debug_vis = False


@configclass
class B2Z1VB5PayoutVisionAMPEnvCfg_Play(B2Z1VA15LoadedGunVisionAMPEnvCfg_Play):
    """Camera-driven crack playback with gun and freely paying-out hose."""

    def __post_init__(self):
        super().__post_init__()
        self.hose_tension_stiffness_n_per_m = 0.0
        self.hose_tension_damping_ns_per_m = 0.0
        self.commands.base_velocity.debug_vis = False
        # VB6 visual playback: road-surface standoff near the measured TCP
        # height (~7.5 mm). Keep the trench-bottom-to-road offset unchanged.
        # Training and synthetic-play command distributions are not modified.
        self.commands.ee_target.tool_height_m = 0.01
        self.commands.ee_target.file_target_height_filter_in_world_frame = True
        # This flat simulation road is at world z=0. Avoid perceived trench
        # depth noise moving the 1 cm standoff above/below the actual surface.
        self.commands.ee_target.file_road_surface_z_w_m = 0.0
