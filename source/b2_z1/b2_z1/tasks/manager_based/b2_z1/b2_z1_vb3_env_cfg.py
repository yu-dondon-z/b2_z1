"""VB3: rebalance body path alignment against tool tracking under load."""

from isaaclab.utils import configclass

from .b2_z1_vb2_env_cfg import (
    B2Z1VB2BoundedHoseAMPEnvCfg,
    B2Z1VB2BoundedHoseAMPEnvCfg_Play,
)


@configclass
class B2Z1VB3PathBalanceAMPEnvCfg(B2Z1VB2BoundedHoseAMPEnvCfg):
    """Keep the VB2 load model but resist base lateral and heading drift."""

    def __post_init__(self):
        super().__post_init__()
        # VB2 improved TCP p95 while base lateral p95 roughly doubled.
        # Use a modest change to preserve its upright trot and arm behaviour.
        self.rewards.base_centerline_error.weight = -0.25
        self.rewards.base_heading_error.weight = -0.15
        self.rewards.crack_centerline_tracking.weight = 0.85
        self.rewards.crack_heading_tracking.weight = 0.40


@configclass
class B2Z1VB3PathBalanceAMPEnvCfg_Play(B2Z1VB2BoundedHoseAMPEnvCfg_Play):
    """Unchanged fixed-load comparison scene for VB3 checkpoints."""
