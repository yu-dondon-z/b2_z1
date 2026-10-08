"""VB2: isolate a bounded truck-side hose pull from the VB1 weight model."""

from isaaclab.utils import configclass

from .b2_z1_vb1_env_cfg import B2Z1VB1LoadedGunAMPEnvCfg, B2Z1VB1LoadedGunAMPEnvCfg_Play
from .b2_z1_va15_vision_env_cfg import B2Z1VA15LoadedGunVisionAMPEnvCfg_Play


@configclass
class B2Z1VB2BoundedHoseAMPEnvCfg(B2Z1VB1LoadedGunAMPEnvCfg):
    """Stage-1 continuation with 1-3 kg suspended mass and capped tether pull.

    60 N is a conservative *simulation hypothesis*, not a measured truck/hose
    specification. The uncapped cable-model tension is logged for comparison.
    """

    hose_tension_max_n: float = 60.0

    def __post_init__(self):
        super().__post_init__()
        # The inherited reset event writes a zero local-frame base wrench.
        # It does not perturb the robot, but flips Isaac Lab's articulation-wide
        # wrench frame away from the gun's global-frame load every reset.
        self.events.base_external_force_torque = None
        if self.hose_tension_max_n <= 0.0:
            raise ValueError("VB2 hose_tension_max_n must be positive")


@configclass
class B2Z1VB2BoundedHoseAMPEnvCfg_Play(B2Z1VB1LoadedGunAMPEnvCfg_Play):
    """Use the same bounded tether for fixed-load synthetic evaluation."""

    hose_tension_max_n: float = 60.0


@configclass
class B2Z1VB2BoundedHoseVisionAMPEnvCfg_Play(B2Z1VA15LoadedGunVisionAMPEnvCfg_Play):
    """Vision playback with the VB2 bounded tether model."""

    hose_tension_max_n: float = 60.0
