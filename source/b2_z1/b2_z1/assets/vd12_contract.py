"""Single simulator-independent VD12 exclusion-zone configuration source."""

import hashlib
from pathlib import Path


def geometry_profile(urdf_path):
    """Return a fresh JSON-compatible contract, pinned to model and math source."""
    return {
        "version": 1,
        "urdf_sha256": hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest(),
        "geometry_method": "vd12_segment_obb_sphere_capsule_lower_bound_v1",
        "geometry_source_sha256": hashlib.sha256(
            Path(__file__).with_name("vd12_geometry.py").read_bytes()
        ).hexdigest(),
        "numerical_guard_m": 1.0e-6,
        "minimum_quaternion_norm": 1.0e-8,
        "soft_margin_m": 0.03,
        "hard_threshold_m": 0.0,
        "cost_cap": 9.0,
        "physics_dt_s": 0.005,
        "decimation": 4,
        "termination_period_s": 0.02,
        "whole_parent_body_excluded": "link06",
        "scope": "URDF authored robot primitives, fixed-child folding; discrete 200Hz exclusion",
        "clearance_reward_weight": -1.0,
        # RewardManager multiplies by 20 ms: this contributes -2 on failure.
        "failure_reward_weight": -100.0,
    }
