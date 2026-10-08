"""VB1: VA15 curve policy adapted to a physical gun and randomized hose load."""

from __future__ import annotations

import os
import math

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from b2_z1.assets.unitree import B2_Z1_GUN_CFG

from .b2_z1_amp_env_cfg import B2Z1SceneCfg
from .b2_z1_va_env_cfg import B2Z1VA15CurriculumAMPEnvCfg, B2Z1VA15CurriculumAMPEnvCfg_Play


@configclass
class VB1LoadedGunSceneCfg(B2Z1SceneCfg):
    robot: ArticulationCfg = B2_Z1_GUN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class B2Z1VB1LoadedGunAMPEnvCfg(B2Z1VA15CurriculumAMPEnvCfg):
    """Training: sample one load per environment at reset, not each physics tick."""

    scene: VB1LoadedGunSceneCfg = VB1LoadedGunSceneCfg(num_envs=512, env_spacing=4.0)
    hose_effective_mass_kg: float = 3.0
    hose_mass_range_kg: tuple[float, float] = (1.0, 3.0)
    hose_attachment_gun_frame_m: tuple[float, float, float] = (0.0, -0.03, 0.48)
    hose_truck_anchor_env_m: tuple[float, float, float] = (0.0, -2.0, 2.8)
    hose_anchor_jitter_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hose_slack_length_m: float = 4.0
    hose_slack_jitter_m: float = 0.0
    hose_tension_stiffness_n_per_m: float = 45.0
    hose_tension_damping_ns_per_m: float = 8.0
    hose_linear_drag_ns_per_m: float = 4.0
    hose_acceleration_filter_s: float = 0.08
    hose_max_acceleration_mps2: float = 8.0
    hose_max_force_n: float = 250.0

    def __post_init__(self):
        super().__post_init__()
        stage = int(os.environ.get("B2_Z1_VB1_STAGE", "1"))
        if stage == 1:
            self.hose_mass_range_kg = (1.0, 3.0)
            self.hose_anchor_jitter_m = (0.0, 0.0, 0.0)
            self.hose_slack_jitter_m = 0.0
        elif stage == 2:
            self.hose_mass_range_kg = (1.0, 6.0)
            self.hose_anchor_jitter_m = (0.25, 0.25, 0.20)
            self.hose_slack_jitter_m = 0.25
        elif stage == 3:
            self.hose_mass_range_kg = (1.0, 10.0)
            self.hose_anchor_jitter_m = (0.50, 0.50, 0.40)
            self.hose_slack_jitter_m = 0.50
        else:
            raise ValueError("B2_Z1_VB1_STAGE must be 1, 2 or 3")
        self.hose_mass_range_kg = tuple(self.hose_mass_range_kg)
        low, high = self.hose_mass_range_kg
        if not 1.0 <= low <= high <= 10.0:
            raise ValueError("VB1 hose mass range must lie within [1, 10] kg")


@configclass
class B2Z1VB1LoadedGunAMPEnvCfg_Play(B2Z1VA15CurriculumAMPEnvCfg_Play):
    """Fixed-load synthetic-curve evaluation without a camera dependency."""

    scene: VB1LoadedGunSceneCfg = VB1LoadedGunSceneCfg(num_envs=1, env_spacing=4.0)
    hose_effective_mass_kg: float = 3.0
    hose_attachment_gun_frame_m: tuple[float, float, float] = (0.0, -0.03, 0.48)
    hose_truck_anchor_env_m: tuple[float, float, float] = (0.0, -2.0, 2.8)
    hose_slack_length_m: float = 4.0
    hose_tension_stiffness_n_per_m: float = 45.0
    hose_tension_damping_ns_per_m: float = 8.0
    hose_linear_drag_ns_per_m: float = 4.0
    hose_acceleration_filter_s: float = 0.08
    hose_max_acceleration_mps2: float = 8.0
    hose_max_force_n: float = 250.0

    def __post_init__(self):
        super().__post_init__()
        self.hose_effective_mass_kg = float(os.environ.get("B2_Z1_HOSE_MASS_KG", self.hose_effective_mass_kg))
        if not math.isfinite(self.hose_effective_mass_kg) or not 1.0 <= self.hose_effective_mass_kg <= 10.0:
            raise ValueError("B2_Z1_HOSE_MASS_KG must be within [1, 10] kg")
