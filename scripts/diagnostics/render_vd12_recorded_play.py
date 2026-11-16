#!/usr/bin/env python3
"""CPU-only schematic playback of recorded Isaac states; never runs dynamics."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

LABEL = "Isaac actual states; simplified offline reconstruction, not live Isaac rendering or MuJoCo dynamics"
VIDEO_USE = "Pose visualization only; decimated video must not be used to establish high-frequency jitter or safety. Use the original 200 Hz logs for metrics."


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pose(position, rotation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def quaternion_matrix(wxyz):
    w, x, y, z = np.asarray(wxyz, float)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def parse_joints(path):
    result = []
    for joint in ET.parse(path).getroot().findall("joint"):
        origin = joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        axis = joint.find("axis")
        axis_xyz = np.fromstring(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" ")
        result.append(dict(name=joint.get("name"), kind=joint.get("type"),
                           parent=joint.find("parent").get("link"), child=joint.find("child").get("link"),
                           origin=pose(xyz, Rotation.from_euler("xyz", rpy).as_matrix()), axis=axis_xyz))
    return result


def forward_kinematics(joints, joint_names, q, base_position, base_quat):
    if len(q) != len(joint_names) or len(set(joint_names)) != len(joint_names):
        raise ValueError("Logged joint names/positions must form a one-to-one mapping")
    angles = dict(zip(joint_names, q))
    moving = {j["name"] for j in joints if j["kind"] != "fixed"}
    if moving != set(joint_names):
        raise ValueError(f"Missing or extra logged joints: {moving ^ set(joint_names)}")
    transforms = {"base_link": pose(base_position, quaternion_matrix(base_quat))}
    pending = list(joints)
    while pending:
        old_count = len(pending)
        for joint in pending[:]:
            if joint["parent"] not in transforms:
                continue
            motion = np.eye(4)
            if joint["kind"] in ("revolute", "continuous"):
                axis = joint["axis"] / np.linalg.norm(joint["axis"])
                motion[:3, :3] = Rotation.from_rotvec(axis * angles[joint["name"]]).as_matrix()
            elif joint["kind"] != "fixed":
                raise ValueError(f"Unsupported joint type {joint['kind']}")
            transforms[joint["child"]] = transforms[joint["parent"]] @ joint["origin"] @ motion
            pending.remove(joint)
        if len(pending) == old_count:
            raise ValueError("URDF graph is not rooted at base_link")
    return transforms


def primitive_lines(primitive):
    """Return local-space wireframe polylines for authored collision shapes."""
    kind = primitive["primitive"]
    if kind == "box":
        half = np.asarray(primitive["half_size_m"])
        vertices = np.asarray(list(itertools.product([-1, 1], repeat=3))) * half
        return [vertices[[i, j]] for i in range(8) for j in range(i + 1, 8)
                if np.count_nonzero(vertices[i] != vertices[j]) == 1]
    radius = primitive["radius_m"]
    angle = np.linspace(0, 2 * np.pi, 25)
    circle = np.c_[radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)]
    if kind == "sphere":
        return [circle, circle[:, [0, 2, 1]], circle[:, [2, 0, 1]]]
    if kind == "cylinder":
        half_length = primitive["half_length_m"]
        rings = [circle + [0, 0, z] for z in (-half_length, half_length)]
        return rings + [np.array([[radius * np.cos(a), radius * np.sin(a), -half_length],
                                  [radius * np.cos(a), radius * np.sin(a), half_length]])
                        for a in np.linspace(0, 2 * np.pi, 5)[:-1]]
    raise ValueError(f"Unsupported primitive: {kind}")


def transformed_lines(primitive, transforms):
    transform = transforms[primitive["carrier_body"]] @ pose(
        primitive["local_position_m"], quaternion_matrix(primitive["local_quaternion_wxyz"]))
    return [line @ transform[:3, :3].T + transform[:3, 3] for line in primitive_lines(primitive)]


def validate_summary(manifest, summary, fps):
    """Only complete, first-episode recordings are eligible for this renderer."""
    if summary.get("status") != "completed_sampled_play" or summary.get("completed_requested_duration") is not True:
        raise ValueError("Requires a completed_requested_duration summary; rejected/partial plays are not accepted")
    if summary.get("initial_episode_only") is not True:
        raise ValueError("Requires a complete initial episode without resets")
    if any(ids for name, ids in summary.get("terminal", {}).get("causes", {}).items() if name != "time_out"):
        raise ValueError("Source summary contains a failure/geometry termination")
    if type(manifest.get("iteration")) is not int or manifest["iteration"] < 0:
        raise ValueError("Manifest must contain a nonnegative checkpoint iteration")
    dt = manifest["physics_dt_s"]
    if not np.isclose(dt, .005, rtol=0, atol=1e-12):
        raise ValueError("Renderer requires the original 200 Hz recording")
    if type(fps) is not int or not 1 <= fps <= 50 or 200 % fps:
        raise ValueError("Video fps must be an integer divisor of 200 Hz between 1 and 50")
    duration = manifest["requested_duration_s"]
    for name in ("duration_s", "requested_duration_s"):
        if not np.isfinite(summary.get(name, np.nan)) or not np.isclose(summary[name], duration, rtol=0, atol=dt / 10):
            raise ValueError("Summary duration does not match the full requested recording")
    return 200 // fps


def select_complete_records(rows, manifest, summary, stride):
    """Validate every original 200 Hz row before selecting exact-timebase frames."""
    records = []
    count = 0
    num_envs = manifest["num_envs"]
    decimation = round(manifest["control_dt_s"] / manifest["physics_dt_s"])
    for count, row in enumerate(rows, 1):
        if row.get("physics_step") != count or not np.isclose(row.get("time_s", np.nan), count * .005, rtol=0, atol=1e-7):
            raise ValueError(f"Non-contiguous physics data or timebase at row {count}")
        actor_step = np.asarray(row.get("episode_actor_step", []))
        if actor_step.shape != (num_envs,) or np.any(actor_step != (count - 1) // decimation):
            raise ValueError(f"Episode reset/discontinuity at row {count}")
        for name, shape in (("joint_position_rad", (num_envs, 18)),
                            ("base_position_w", (num_envs, 3)), ("base_quat_wxyz", (num_envs, 4)),
                            ("tcp_position_w", (num_envs, 3)), ("tcp_command_world_50hz_zoh", (num_envs, 3))):
            values = np.asarray(row.get(name, []), dtype=float)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"Missing/nonfinite {name} at row {count}")
        if np.any(np.linalg.norm(row["base_quat_wxyz"], axis=1) < 1e-8):
            raise ValueError(f"Invalid root quaternion at row {count}")
        clearance = np.asarray(row.get("minimum_clearance_lower_bound_m", []), dtype=float)
        valid = np.asarray(row.get("geometry_input_valid", []))
        latched = np.asarray(row.get("exclusion_latched", []))
        if (clearance.shape != (num_envs,) or not np.isfinite(clearance).all() or np.any(clearance < 0)
                or valid.shape != (num_envs,) or not np.all(valid == True)
                or latched.shape != (num_envs,) or not np.all(latched == False)):
            raise ValueError(f"Invalid/excluded geometry at row {count}; only complete safe-source recordings are supported")
        if count % stride == 0:
            records.append(row)
    if count == 0:
        raise ValueError("Physics log is empty")
    if count != summary.get("physics_steps") or not np.isclose(count * .005, manifest["requested_duration_s"], rtol=0, atol=1e-7):
        raise ValueError("Physics log is incomplete or disagrees with the completed summary")
    if len(records) < 6:
        raise ValueError("Recording is too short to provide six distinct stage frames at the requested fps")
    return records, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("play_dir", type=Path)
    parser.add_argument("--env", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import imageio.v2 as imageio

    source = args.play_dir.resolve()
    manifest_path = source / "manifest.json"
    log_path = source / "physics_200hz.jsonl.gz"
    summary_path = source / "summary.json"
    manifest = json.loads(manifest_path.read_text())
    if not 0 <= args.env < manifest["num_envs"]:
        raise ValueError("Invalid environment index")
    if not summary_path.is_file():
        raise ValueError("A completed summary.json is required; partial/interrupted plays are not accepted")
    summary = json.loads(summary_path.read_text())
    step_stride = validate_summary(manifest, summary, args.fps)
    registry = manifest["geometry_registry"]
    urdf = Path(registry["source_urdf"])
    if digest(urdf) != registry["source_urdf_sha256"]:
        raise ValueError("URDF hash differs from recorded geometry registry")
    joints = parse_joints(urdf)
    with gzip.open(log_path, "rt") as f:
        records, source_samples = select_complete_records((json.loads(line) for line in f), manifest, summary, step_stride)
    output = Path(tempfile.mkdtemp(prefix="vd12_recorded_play_", dir=source.parent))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4), dpi=100)
    fig.subplots_adjust(top=.85, bottom=.16, wspace=.23)
    iteration = manifest["iteration"]
    fig.suptitle(f"VD12 model_{iteration} | Recorded Isaac Lab Play | Actual 18-joint trajectory", fontsize=14)
    fig.text(.5, .035, LABEL, ha="center", fontsize=8)
    artists = []
    for ax, dims, name in zip(axes, [(0, 2), (0, 1)], ["Side view: base heading / height", "Top view: base heading / lateral"]):
        ax.set_title(name)
        ax.set_xlabel("Base-heading forward (m)")
        ax.set_ylabel("World height (m)" if dims == (0, 2) else "Base-heading left (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=.2)
        if dims == (0, 2):
            ax.set_xlim(-1.5, 1.0)
            ax.set_ylim(-.08, 1.5)
            ax.axhline(0, color="saddlebrown", lw=2)
        else:
            ax.set_xlim(-1.5, 1.0)
            ax.set_ylim(-.85, .85)
        body = LineCollection([], colors="#6e859b", linewidths=.8, alpha=.65)
        skeleton = LineCollection([], colors="#254460", linewidths=2)
        arm = LineCollection([], colors="#d47a0b", linewidths=4)
        gun = LineCollection([], colors="#d52936", linewidths=1.5)
        for item in (body, skeleton, arm, gun):
            ax.add_collection(item)
        tip, = ax.plot([], [], "o", color="#d52936", ms=6, label="Gun tip")
        target, = ax.plot([], [], "+", color="#188b48", ms=10, label="50 Hz target (ZOH)")
        ax.legend(loc="upper right", fontsize=8)
        artists.append((dims, body, skeleton, arm, gun, tip, target))
    caption = fig.text(.5, .90, "", ha="center", fontsize=11)
    errors = []
    snapshots = []
    snapshot_indices = set(np.linspace(0, len(records) - 1, 6, dtype=int))
    movie = output / f"vd12_model{iteration}_recorded_play.mp4"
    with imageio.get_writer(movie, fps=args.fps, codec="libx264", quality=7) as writer:
        for frame_index, row in enumerate(records):
            env = args.env
            transforms = forward_kinematics(joints, manifest["joint_names"], row["joint_position_rad"][env],
                                            row["base_position_w"][env], row["base_quat_wxyz"][env])
            tcp = transforms["link06"] @ np.array([.975, 0, .1, 1])
            logged_tcp = np.asarray(row["tcp_position_w"][env])
            errors.append(float(np.linalg.norm(tcp[:3] - logged_tcp)))
            root = transforms["base_link"]
            yaw = np.arctan2(root[1, 0], root[0, 0])
            view_rot = Rotation.from_euler("z", -yaw).as_matrix()
            origin = root[:3, 3] * [1, 1, 0]

            def view(points):
                return (np.asarray(points) - origin) @ view_rot.T

            body_lines = [view(line) for primitive in registry["pairs"] for line in transformed_lines(primitive, transforms)]
            gun_lines = [view(line) for line in transformed_lines(registry["tube"], transforms)]
            skeleton_lines, arm_lines = [], []
            for joint in joints:
                if joint["kind"] == "fixed":
                    continue
                line = view([transforms[joint["parent"]][:3, 3], transforms[joint["child"]][:3, 3]])
                (arm_lines if joint["name"].startswith("joint") else skeleton_lines).append(line)
            tip_xy = view(logged_tcp)
            target_xy = view(row["tcp_command_world_50hz_zoh"][env])
            for dims, body, skeleton, arm, gun, tip, target in artists:
                body.set_segments([line[:, dims] for line in body_lines])
                skeleton.set_segments([line[:, dims] for line in skeleton_lines])
                arm.set_segments([line[:, dims] for line in arm_lines])
                gun.set_segments([line[:, dims] for line in gun_lines])
                tip.set_data([tip_xy[dims[0]]], [tip_xy[dims[1]]])
                target.set_data([target_xy[dims[0]]], [target_xy[dims[1]]])
            caption.set_text(f"t={row['time_s']:.3f} s | conservative clearance lower bound="
                             f"{1000*row['minimum_clearance_lower_bound_m'][env]:.1f} mm | env {env}")
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            writer.append_data(frame)
            if frame_index in snapshot_indices:
                path = output / f"frame_{frame_index:04d}_{row['time_s']:.3f}s.png"
                imageio.imwrite(path, frame)
                snapshots.append(path)
    plt.close(fig)
    tiles = [imageio.imread(path) for path in snapshots]
    imageio.imwrite(output / "stages_contact_sheet.png", np.concatenate([
        np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:6], axis=1)], axis=0))
    result = dict(label=LABEL, source_play_dir=str(source), environment=args.env,
                  inputs_sha256={str(p): digest(p) for p in (manifest_path, summary_path, log_path, urdf)},
                  checkpoint_sha256=manifest["checkpoint_sha256"], source_samples=source_samples,
                  checkpoint_iteration=iteration, completed_requested_duration=True,
                  source_validated_no_gaps_no_reset_no_exclusion=True,
                  video_frames=len(records), fps=args.fps, actual_first_time_s=records[0]["time_s"],
                  actual_last_time_s=records[-1]["time_s"], all_18_joints_from_logs=True,
                  video_use=VIDEO_USE, high_frequency_jitter_proven_by_video=False,
                  root_position_and_quaternion_from_logs=True, dynamics_stepped=False,
                  fk_tcp_offset_link06_m=[.975, 0, .1], fk_tcp_error_max_m=max(errors),
                  fk_tcp_error_mean_m=float(np.mean(errors)),
                  note="Rigid URDF FK may differ from logged PhysX states by roundoff and constraint tolerances; not a safety certificate",
                  movie=str(movie), contact_sheet=str(output / "stages_contact_sheet.png"))
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
