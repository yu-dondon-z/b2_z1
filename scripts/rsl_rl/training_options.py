"""Small, simulator-independent helpers for training launch and checkpoint loading."""

from pathlib import Path


CURRICULUM_EXPERIMENTS = {
    "b2_z1_vd2_curve_amp", "b2_z1_vd3_curve_amp", "b2_z1_vd4_front_calibrated_amp",
    "b2_z1_vd5_world_seam_amp",
    "b2_z1_vd5_startup_amp",
    "b2_z1_vd6_smooth_amp",
    "b2_z1_vd7_tcp_amp",
    "b2_z1_vd8_roll_landing_amp",
    "b2_z1_vd9_physics_rate_amp",
    "b2_z1_vd10_hollow_gun_amp",
    "b2_z1_vd11_motion_quality_amp",
    "b2_z1_vd12_geometry_guard_amp",
    "b2_z1_vd13_mount75_amp",
    "b2_z1_vd14_posture_amp",
}


def load_training_checkpoint(runner, env, experiment_name, resume_path=None,
                             warm_start=None, legacy_curriculum=None, startup_from=None,
                             vd6_from=None, vd7_from=None, vd8_from=None, vd9_from=None,
                             vd9_warm_start=None, vd10_from=None, vd11_from=None, vd12_from=None, vd13_from=None,
                             vd14_from=None):
    other_modes = (resume_path, warm_start, startup_from, vd6_from, vd7_from, vd8_from,
                   vd9_from, vd9_warm_start, vd10_from, vd11_from)
    if vd14_from is not None and (any(other_modes) or vd12_from is not None or vd13_from is not None):
        raise ValueError("VD14 load mode is exclusive, including explicit fresh start")
    if vd14_from is not None and experiment_name != "b2_z1_vd14_posture_amp":
        raise ValueError("VD14 source requires the VD14 experiment")
    if experiment_name == "b2_z1_vd14_posture_amp" and (
            any(other_modes[1:]) or vd12_from is not None or vd13_from is not None
            or legacy_curriculum is not None):
        raise ValueError("VD14 accepts only --vd14-from, same-version resume, or fresh training")
    if vd13_from is not None and (any(other_modes) or vd12_from is not None):
        raise ValueError('VD13 load mode is exclusive, including explicit fresh start')
    if vd13_from is not None and experiment_name != 'b2_z1_vd13_mount75_amp':
        raise ValueError('VD13 source requires the VD13 experiment')
    if experiment_name == 'b2_z1_vd13_mount75_amp' and (
            any(other_modes[1:]) or vd12_from is not None or legacy_curriculum is not None):
        raise ValueError('VD13 accepts only VD12 warm start, same-version resume, or fresh training')
    if sum(bool(value) for value in (resume_path, warm_start, startup_from, vd6_from, vd7_from,
                                     vd8_from, vd9_from, vd9_warm_start, vd10_from, vd11_from,
                                     vd12_from)) > 1:
        raise ValueError("Checkpoint load modes are mutually exclusive")
    if vd12_from is not None:
        if any((resume_path, warm_start, startup_from, vd6_from, vd7_from, vd8_from, vd9_from,
                vd9_warm_start, vd10_from, vd11_from)):
            raise ValueError("Checkpoint load modes are mutually exclusive, including explicit VD12 fresh start")
        if experiment_name != "b2_z1_vd12_geometry_guard_amp":
            raise ValueError("VD12 warm start requires experiment_name=b2_z1_vd12_geometry_guard_amp")
    if experiment_name == "b2_z1_vd12_geometry_guard_amp":
        if any((warm_start, startup_from, vd6_from, vd7_from, vd8_from, vd9_from,
                vd9_warm_start, vd10_from, vd11_from)) or legacy_curriculum is not None:
            raise ValueError("VD12 accepts only --vd12-from, a same-version resume, or fresh training")
    if vd14_from:
        # Same mount/commands: retain the learned task curriculum, but begin a
        # new reward experiment without importing Adam moments or iteration.
        runner.load(vd14_from, load_optimizer=False, restore_curriculum=True, migrate_vd14=True)
        runner.current_learning_iteration = 0
        env.reset()
    elif vd13_from:
        runner.load(vd13_from, load_optimizer=False, restore_curriculum=False, migrate_vd13=True)
        runner.current_learning_iteration = 0
        env.reset()
    elif vd12_from:
        # The measured-tube actor and AMP transfer, not the old optimizer,
        # iteration or ground-target curriculum. Re-learn the high-to-ground
        # transition with the new conservative whole-gun exclusion rule.
        runner.load(vd12_from, load_optimizer=False, restore_curriculum=False, migrate_vd12=True)
        runner.current_learning_iteration = 0
        env.reset()
    elif vd11_from:
        if experiment_name != 'b2_z1_vd11_motion_quality_amp':
            raise ValueError('VD11 warm start requires experiment_name=b2_z1_vd11_motion_quality_amp')
        # Preserve the measured-tube geometry/startup profile and ground level,
        # but never import VD10's optimizer or iteration into the new objective.
        runner.load(vd11_from, load_optimizer=False, restore_curriculum=True, migrate_vd11=True)
        runner.current_learning_iteration = 0
        env.reset()
    elif vd10_from:
        if experiment_name != 'b2_z1_vd10_hollow_gun_amp':
            raise ValueError('VD10 warm start requires experiment_name=b2_z1_vd10_hollow_gun_amp')
        runner.load(vd10_from, load_optimizer=False, restore_curriculum=False)
        runner.current_learning_iteration = 0
        env.reset()
    elif vd9_warm_start:
        if experiment_name != 'b2_z1_vd9_physics_rate_amp':
            raise ValueError('VD9 warm start requires experiment_name=b2_z1_vd9_physics_rate_amp')
        runner.load(vd9_warm_start, load_optimizer=False, restore_curriculum=False)
        runner.current_learning_iteration = 0
        env.reset()
    elif vd9_from:
        if experiment_name != 'b2_z1_vd9_physics_rate_amp':
            raise ValueError('VD9 warm start requires experiment_name=b2_z1_vd9_physics_rate_amp')
        runner.load(vd9_from, load_optimizer=False, restore_curriculum=True, migrate_vd9=True)
        runner.current_learning_iteration = 0
        env.reset()
    elif vd8_from:
        if experiment_name != 'b2_z1_vd8_roll_landing_amp':
            raise ValueError('VD8 migration requires experiment_name=b2_z1_vd8_roll_landing_amp')
        runner.load(vd8_from, restore_curriculum=True, migrate_vd8=True)
        env.reset()
    elif vd7_from:
        if experiment_name != 'b2_z1_vd7_tcp_amp':
            raise ValueError('VD7 migration requires experiment_name=b2_z1_vd7_tcp_amp')
        runner.load(vd7_from, restore_curriculum=True, migrate_vd7=True)
        env.reset()
    elif vd6_from:
        if experiment_name != 'b2_z1_vd6_smooth_amp':
            raise ValueError('VD6 migration requires experiment_name=b2_z1_vd6_smooth_amp')
        runner.load(vd6_from, restore_curriculum=True, migrate_vd6=True)
        env.reset()
    elif startup_from:
        if experiment_name != "b2_z1_vd5_startup_amp":
            raise ValueError("Startup migration requires experiment_name=b2_z1_vd5_startup_amp")
        runner.load(startup_from, restore_curriculum=True, migrate_vd5_startup=True)
        env.reset()
    elif resume_path:
        if experiment_name in CURRICULUM_EXPERIMENTS:
            runner.load(resume_path, restore_curriculum=True, legacy_curriculum=legacy_curriculum)
            env.reset()
        else:
            runner.load(resume_path)
    elif warm_start:
        # Curriculum is deliberately left at the new task's configured initial level.
        runner.load(warm_start, load_optimizer=False, restore_curriculum=False)
        runner.current_learning_iteration = 0
        env.reset()


def persistent_log_root(path):
    """Reject Linux RAM filesystems before writing durable training artifacts."""
    root = Path(path).expanduser().resolve()
    mount_type, mount_length = None, -1
    try:
        lines = Path("/proc/mounts").read_text().splitlines()
    except FileNotFoundError:
        lines = []
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        mount = Path(fields[1].replace("\\040", " ").replace("\\011", "\t")).resolve()
        if (root == mount or mount in root.parents) and len(str(mount)) > mount_length:
            mount_type, mount_length = fields[2], len(str(mount))
    if mount_type in {"tmpfs", "ramfs", "devtmpfs"}:
        raise ValueError(f"Training log root must use persistent storage, got {mount_type}: {root}")
    return root
