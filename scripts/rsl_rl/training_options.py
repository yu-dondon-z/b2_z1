"""Small, simulator-independent helpers for training launch and checkpoint loading."""

from pathlib import Path


CURRICULUM_EXPERIMENTS = {
    "b2_z1_vd2_curve_amp", "b2_z1_vd3_curve_amp", "b2_z1_vd4_front_calibrated_amp",
    "b2_z1_vd5_world_seam_amp",
    "b2_z1_vd5_startup_amp",
}


def load_training_checkpoint(runner, env, experiment_name, resume_path=None,
                             warm_start=None, legacy_curriculum=None, startup_from=None):
    if sum(bool(value) for value in (resume_path, warm_start, startup_from)) > 1:
        raise ValueError("Checkpoint load modes are mutually exclusive")
    if startup_from:
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
