"""VB4: gun and hanging-hose weight with an assumed freely paying-out hose."""

from isaaclab.utils import configclass

from .b2_z1_vb1_env_cfg import B2Z1VB1LoadedGunAMPEnvCfg, B2Z1VB1LoadedGunAMPEnvCfg_Play


@configclass
class B2Z1VB4PayoutAMPEnvCfg(B2Z1VB1LoadedGunAMPEnvCfg):
    """Stage-1 load without fixed-truck tether pull.

    This is a *payout hypothesis*, not a measured hose model. Gun collision,
    gun inertia, and 1-3 kg effective unsupported hose mass remain active.
    """

    def __post_init__(self):
        super().__post_init__()
        self.hose_tension_stiffness_n_per_m = 0.0
        self.hose_tension_damping_ns_per_m = 0.0
        self.events.base_external_force_torque = None


@configclass
class B2Z1VB4PayoutAMPEnvCfg_Play(B2Z1VB1LoadedGunAMPEnvCfg_Play):
    """Identical free-payout hypothesis for fixed-load playback."""

    def __post_init__(self):
        super().__post_init__()
        self.hose_tension_stiffness_n_per_m = 0.0
        self.hose_tension_damping_ns_per_m = 0.0
