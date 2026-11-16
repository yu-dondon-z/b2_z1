"""Isolated posture objective on the unchanged VD13 mount and controller."""
from . import vd13_contract

EXPERIMENT = "b2_z1_vd14_posture_amp"


def posture_profile():
    # 95% of reference inclinations are <=3.73 degrees; 97% of reference
    # front-foot lateral separations are >=.42m. These are soft costs, not
    # terminations or a mandate to reproduce a single reference trajectory.
    return {
        "version": 1,
        "base_tilt_weight": -8.0,
        "base_tilt_deadzone_rad": .07,
        "base_tilt_scale_rad": .15,
        "front_width_weight": -2.0,
        "front_min_width_m": .42,
        "front_width_scale_m": .10,
        "front_body_names": ["FL_foot", "FR_foot"],
        "width_frame": "signed FL.y-FR.y in full base frame; no contact gating",
    }


def validate_environment_profile(cfg):
    vd13_contract.validate_environment_profile(cfg)
    expected = posture_profile()
    tilt, width = cfg.rewards.base_tilt, cfg.rewards.vd14_front_width
    if (cfg.vd14_posture_profile != expected
            or tilt.weight != expected["base_tilt_weight"]
            or tilt.params["deadzone"] != expected["base_tilt_deadzone_rad"]
            or tilt.params["scale"] != expected["base_tilt_scale_rad"]
            or width.weight != expected["front_width_weight"]
            or width.params["minimum_width"] != expected["front_min_width_m"]
            or width.params["scale"] != expected["front_width_scale_m"]
            or width.params["asset_cfg"].body_names != expected["front_body_names"]
            or not width.params["asset_cfg"].preserve_order
            or width.func.__name__ != "front_stance_width_penalty"
            or tilt.func.__name__ != "base_tilt_penalty"):
        raise ValueError("VD14 actual posture rewards differ from their saved profile")


def validate_checkpoint(checkpoint, posture, mount, geometry, *, migration,
                        load_optimizer, restore_curriculum):
    """Check all provenance before any runner state can be mutated."""
    if posture != posture_profile() or not restore_curriculum:
        raise ValueError("VD14 requires its declared posture profile and full curriculum restoration")
    source = vd13_contract.EXPERIMENT if migration else EXPERIMENT
    if checkpoint.get("training_experiment_name") != source:
        raise ValueError(f"VD14 load requires {source}")
    if migration:
        if load_optimizer:
            raise ValueError("VD13 to VD14 transfers policy/AMP/curriculum, not the optimizer")
    elif not load_optimizer or checkpoint.get("vd14_posture_profile") != posture:
        raise ValueError("VD14 resume requires matching posture rewards and optimizer")
    # The geometry/curriculum/AMP completeness rules are identical to a full
    # VD13 checkpoint. Validate a shallow metadata view; never rewrite source.
    view = dict(checkpoint, training_experiment_name=vd13_contract.EXPERIMENT)
    vd13_contract.validate_checkpoint(view, mount, geometry, migration=False,
                                     load_optimizer=True, restore_curriculum=True)
