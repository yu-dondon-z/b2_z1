#!/usr/bin/env python3
"""Read-only VD11 Isaac Play with saved curriculum and independent geometry guard.

No optimizer, exporter or robot IO is constructed. The training task (not its
modified Play task) preserves reset distributions, randomization and real
terminations. Geometry is evaluation-only and never added to actor observations.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TASK = "Unitree-b2z1-VD11-Velocity"
DEFAULT_CHECKPOINT = ROOT / "logs/rsl_rl/b2_z1_vd11_motion_quality_amp/2026-11-13_17-56-49_vd11_motion_quality_long_20261113/model_5999.pt"
JOINT_NAMES = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "joint1", "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    "joint2", "joint3", "joint4", "joint5", "joint6",
)
POLICY_TERMS = ("velocity_commands", "target_commands", "base_ang_vel", "projected_gravity",
                "joint_pos_rel", "joint_vel_rel", "actions")


class GeometryStop(Exception):
    """Stop before environment termination/autoreset can erase the bad pose."""


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_actor_contract(checkpoint, agent):
    if checkpoint.get("training_experiment_name") != "b2_z1_vd11_motion_quality_amp":
        raise ValueError("Only the VD11 measured-gun experiment is accepted")
    if checkpoint.get("iter") != 5999 or isinstance(checkpoint.get("iter"), bool):
        raise ValueError("This review requires VD11 model_5999")
    if "obs_norm_state_dict" in checkpoint or agent.get("empirical_normalization") is not False:
        raise ValueError("Expected explicitly disabled empirical observation normalization")
    if "clip_actions" not in agent or agent["clip_actions"] is not None:
        raise ValueError("Expected the saved unclipped actor output contract")
    policy = agent.get("policy", {})
    if policy.get("activation") != "elu" or policy.get("actor_hidden_dims") != [512, 256, 128]:
        raise ValueError("Unexpected actor architecture")
    if checkpoint.get("vd5_curriculum_state", {}).get("ground_blend") != 1.0:
        raise ValueError("Expected final ground curriculum")


def validate_runtime_contract(names, terms, observation):
    if tuple(names) != JOINT_NAMES:
        raise ValueError("Runtime joint order differs from the trained 18-value action order")
    if tuple(terms) != POLICY_TERMS:
        raise ValueError("Runtime policy observation terms/order differ from training")
    if observation.ndim != 2 or observation.shape[-1] != 70:
        raise ValueError("Expected exactly 70 actor observations, without extra truth fields")


def validate_joint_selection_contract(action_names, action_ids, position_ids, velocity_ids):
    if tuple(action_names) != JOINT_NAMES:
        raise ValueError("Resolved action joint names differ from training")

    def indices(value):
        return list(range(18))[value] if isinstance(value, slice) else list(value)

    expected = list(range(18))
    for label, values in (("action", action_ids), ("position observation", position_ids),
                          ("velocity observation", velocity_ids)):
        if indices(values) != expected:
            raise ValueError(f"Resolved {label} joint IDs differ from training")


def side_view(base_position, base_quaternion):
    """Yaw-relative camera: a randomized reset must not hide the long tube end-on."""
    w, x, y, z = base_quaternion
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    base = np.asarray(base_position)
    return base + rotation @ np.array((-1.4, -3.0, 1.1)), base + rotation @ np.array((-0.35, 0.0, 0.05))


def load_saved_env(path):
    """Read trusted local YAML structure without constructing tagged Python objects."""
    import yaml

    class SnapshotLoader(yaml.SafeLoader):
        pass

    def tagged(loader, suffix, node):
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        return loader.construct_scalar(node)

    SnapshotLoader.add_multi_constructor("tag:yaml.org,2002:python/", tagged)
    return yaml.load(path.read_text(), Loader=SnapshotLoader)


def compare_training_settings(cfg, saved):
    """Fail on critical model/physics/IO drift; rendering and env count may differ."""
    current = cfg.to_dict()
    paths = ["sim.dt", "decimation", "episode_length_s", "reference_state_initialization",
             "reference_state_initialization_prob", "observations.policy.enable_corruption",
             "scene.robot.spawn.articulation_props.enabled_self_collisions",
             "scene.robot.init_state.joint_pos", "scene.robot.init_state.pos",
             "actions.JointPositionAction.scale", "actions.JointPositionAction.use_default_offset",
             "actions.JointPositionAction.joint_names", "actions.JointPositionAction.preserve_order",
             "actions.JointPositionAction.clip", "actions.JointPositionAction.offset"]
    for term in POLICY_TERMS:
        for field in ("scale", "clip", "params"):
            paths.append(f"observations.policy.{term}.{field}")
    for motor in current["scene"]["robot"]["actuators"]:
        for field in ("joint_names_expr", "stiffness", "damping", "effort_limit_sim", "velocity_limit_sim"):
            paths.append(f"scene.robot.actuators.{motor}.{field}")

    def lookup(mapping, path):
        for part in path.split("."):
            mapping = mapping[part]
        # YAML tuples become lists; compare their structural values.
        return json.loads(json.dumps(mapping))

    checks = {}
    for path in paths:
        actual, expected = lookup(current, path), lookup(saved, path)
        if actual != expected:
            raise ValueError(f"Current config differs from saved training config: {path}: {actual} != {expected}")
        checks[path] = actual
    return checks


def parser_with_launcher():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--reset-mode", choices=("mixed", "nominal"), default="mixed")
    parser.add_argument("--no-video", action="store_true", help="Keep PNGs/measurements but skip MP4")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/vd11_checked_play_20261114")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or not 0 < args.duration <= 60 or not 1 <= args.num_envs <= 8:
        parser.error("duration must be (0,60] seconds; num_envs must be 1..8")
    args.enable_cameras = True  # Only the external viewport; no image is an actor input.
    if args.viewer:
        args.headless = False
    return AppLauncher, args


def run(args, simulation_app):
    import gymnasium as gym
    import imageio.v2 as imageio
    import torch
    import yaml

    import b2_z1.tasks  # noqa: F401
    from b2_z1.assets.vd10_runtime_contract import audit_vd10_runtime_asset
    from b2_z1.assets.vd11_geometry_audit import GunRobotGeometryAudit
    from b2_z1.rsl_rl.utils.vd5_curriculum_state import capture_vd5_curriculum_state, restore_vd5_curriculum_state
    from b2_z1.tasks.manager_based.b2_z1.mdp.tcp_frame import tcp_pose_w
    from isaaclab_tasks.utils import parse_env_cfg

    checkpoint_path = args.checkpoint.resolve(strict=True)
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    saved_agent = yaml.safe_load((checkpoint_path.parent / "params/agent.yaml").read_text())
    validate_actor_contract(checkpoint, saved_agent)
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    settings = compare_training_settings(cfg, load_saved_env(checkpoint_path.parent / "params/env.yaml"))
    cfg.viewer.resolution = (960, 540)
    actor = torch.nn.Sequential(torch.nn.Linear(70, 512), torch.nn.ELU(), torch.nn.Linear(512, 256),
        torch.nn.ELU(), torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 18)).to(cfg.sim.device).eval()
    actor.load_state_dict({k.removeprefix("actor."): v for k, v in checkpoint["model_state_dict"].items()
                           if k.startswith("actor.")}, strict=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=time.strftime("play_%Y%m%d_%H%M%S_"), dir=args.output_root))
    print(f"[VD11 checked Play] output={output}", flush=True)
    env = gym.make(TASK, cfg=cfg, render_mode="rgb_array")
    raw, writer = env.unwrapped, None
    physics_file = gzip.open(output / "physics_200hz.jsonl.gz", "wt", encoding="utf-8")
    actor_file = gzip.open(output / "actor_50hz.jsonl.gz", "wt", encoding="utf-8")
    geometry_file = gzip.open(output / "geometry_200hz.jsonl.gz", "wt", encoding="utf-8")
    try:
        robot = raw.scene["robot"]
        restore_vd5_curriculum_state(raw, checkpoint["vd5_curriculum_state"])
        if args.reset_mode == "nominal":
            # Explicit differential diagnostic, AFTER saved-profile validation.
            raw.cfg.reference_state_initialization_prob = 0.0
        observations, _ = env.reset(seed=args.seed)
        validate_runtime_contract(robot.joint_names, raw.observation_manager.active_terms["policy"], observations["policy"])
        import inspect

        resolved_observation_ids = {}
        for name, term_cfg in zip(raw.observation_manager.active_terms["policy"],
                                  raw.observation_manager._group_obs_term_cfgs["policy"]):
            if name not in ("joint_pos_rel", "joint_vel_rel"):
                continue
            asset_cfg = term_cfg.params.get("asset_cfg")
            if asset_cfg is None:
                asset_cfg = inspect.signature(term_cfg.func).parameters["asset_cfg"].default
            if asset_cfg.name != "robot":
                raise ValueError("Joint observations unexpectedly reference another asset")
            resolved_observation_ids[name] = asset_cfg.joint_ids
        action_term = raw.action_manager.get_term("JointPositionAction")
        validate_joint_selection_contract(action_term._joint_names, action_term._joint_ids,
            resolved_observation_ids["joint_pos_rel"], resolved_observation_ids["joint_vel_rel"])
        asset = audit_vd10_runtime_asset(robot)
        save(output / "asset_audit.json", asset)
        audit = GunRobotGeometryAudit(cfg.scene.robot.spawn.asset_path)
        limits = robot.data.joint_pos_limits.detach().cpu().numpy()
        sensor = raw.scene.sensors["contact_forces"]
        tool_id = sensor.body_names.index("link06")
        base_id = robot.body_names.index("base_link")
        counts = Counter(safe=0, uncertain=0, invalid=0)
        per_env_counts = [Counter(safe=0, uncertain=0, invalid=0) for _ in range(raw.num_envs)]
        frames, terminal, physics_steps = [], None, 0
        snapshots = iter((0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0, 30.0))
        next_snapshot = next(snapshots)
        save(output / "manifest.json", dict(
            task=TASK, checkpoint=str(checkpoint_path), checkpoint_sha256=sha256(checkpoint_path),
            actor_mode="deterministic mean", actor_observations=70, action_count=18,
            joint_names=list(robot.joint_names), policy_terms=list(POLICY_TERMS),
            resolved_action_and_joint_observation_ids_verified=list(range(18)),
            action_clip_note="No RSL wrapper clip; unchanged JointPositionAction target clip [-100,100] remains active",
            joint_position_limits_rad=limits.tolist(), physics_dt_s=raw.physics_dt, control_dt_s=raw.step_dt,
            training_settings_verified=settings, num_envs=raw.num_envs, seed=args.seed,
            saved_curriculum=checkpoint["vd5_curriculum_state"], restored_curriculum=capture_vd5_curriculum_state(raw),
            reset_mode=args.reset_mode, effective_reference_fraction=raw.cfg.reference_state_initialization_prob,
            overrides=["reference fraction=0 (explicit nominal differential)"] if args.reset_mode == "nominal" else [],
            geometry_audit=audit.metadata, geometry_guard="200Hz actual q; stop on confirmed invalid BEFORE autoreset",
            contact_and_training_terminations="unchanged", randomization="unchanged training configuration",
            observation_noise=cfg.observations.policy.enable_corruption,
            commands_source="unchanged training ideal geometric teacher; not actual image perception",
            actor_extra_world_fields=[], diagnostic_truth_never_fed_to_actor=True,
            no_training=True, no_export=True, no_robot_io=True, requested_duration_s=args.duration,
            torque_note="No actual drive torque claimed. This Play records actual q/qd and contact, not PD-estimated torque."))

        def frame(label, time_s, env_id=0):
            base = robot.data.body_link_pos_w[env_id, base_id].detach().cpu().numpy()
            quaternion = robot.data.body_link_quat_w[env_id, base_id].detach().cpu().numpy()
            eye, target = side_view(base, quaternion)
            raw.sim.set_camera_view(eye=eye.tolist(), target=target.tolist())
            rgb = env.render()
            # Render-product warmup only, never advances physics.
            for _ in range(8):
                if rgb is not None and np.any(rgb):
                    break
                rgb = env.render()
            if rgb is None or not np.any(rgb):
                raise RuntimeError("Viewport did not produce an image; refusing a black Play artifact")
            if label is not None:
                filename = f"{len(frames):02d}_{label}_env{env_id}_{time_s:.3f}s.png"
                imageio.imwrite(output / filename, rgb)
                frames.append(dict(path=filename, time_s=time_s, env_id=env_id,
                    joint_position_rad=robot.data.joint_pos[env_id].detach().cpu().tolist(),
                    no_state_edits=True))
            return rgb

        def record_physics():
            nonlocal physics_steps, terminal
            physics_steps += 1
            q = robot.data.joint_pos.detach().cpu().numpy()
            reports = audit.audit_batch(robot.joint_names, q)
            t = physics_steps * raw.physics_dt
            for i, report in enumerate(reports):
                counts[report["status"]] += 1
                per_env_counts[i][report["status"]] += 1
            geometry_file.write(json.dumps(dict(time_s=t, env_reports=reports), allow_nan=False) + "\n")
            tcp, _ = tcp_pose_w(robot)
            goal = raw.command_manager.get_term("ee_target")
            row = dict(time_s=t, joint_position_rad=q.tolist(),
                joint_velocity_radps=robot.data.joint_vel.detach().cpu().tolist(),
                joint_target_rad=robot.data.joint_pos_target.detach().cpu().tolist(),
                episode_actor_step=raw.episode_length_buf.detach().cpu().tolist(),
                base_position_w=robot.data.root_pos_w.detach().cpu().tolist(),
                base_quat_wxyz=robot.data.root_quat_w.detach().cpu().tolist(),
                tcp_position_w=tcp.detach().cpu().tolist(), tcp_command_world_50hz_zoh=goal._get_target_pos_w().detach().cpu().tolist(),
                tool_contact_force_w=sensor.data.net_forces_w[:, tool_id].detach().cpu().tolist(),
                all_contact_force_w=sensor.data.net_forces_w.detach().cpu().tolist(),
                evaluation_only=True)
            physics_file.write(json.dumps(row, allow_nan=False) + "\n")
            invalid = [i for i, report in enumerate(reports) if report["confirmed_invalid"]]
            if invalid:
                terminal = dict(reason="invalid_geometry", time_s=t, env_ids=invalid,
                    evidence=[reports[i] for i in invalid],
                    note="Independent geometry stop before termination/autoreset; NOT a contact-force event")
                raise GeometryStop()

        initial = audit.audit_batch(robot.joint_names, robot.data.joint_pos.detach().cpu().numpy())
        save(output / "initial_geometry.json", initial)
        invalid_initial = [i for i, report in enumerate(initial) if report["confirmed_invalid"]]
        if invalid_initial:
            terminal = dict(reason="invalid_geometry_at_reset", time_s=0.0, env_ids=invalid_initial,
                            evidence=[initial[i] for i in invalid_initial])
        first_rgb = frame("reset", 0.0)
        if not args.no_video:
            writer = imageio.get_writer(output / "play.mp4", fps=50, codec="libx264", macro_block_size=2)
            writer.append_data(first_rgb)
        raw._physics_step_callbacks.append(record_physics)
        wall_start = time.perf_counter()
        step = 0
        try:
            while terminal is None and step * raw.step_dt < args.duration and simulation_app.is_running():
                with torch.inference_mode():
                    actions = actor(observations["policy"])
                    if tuple(actions.shape) != (raw.num_envs, 18) or not torch.isfinite(actions).all():
                        raise ValueError("Nonfinite or wrong-shaped actor output")
                    actor_file.write(json.dumps(dict(time_s=step * raw.step_dt,
                        observations=observations["policy"].detach().cpu().tolist(),
                        actions=actions.detach().cpu().tolist()), allow_nan=False) + "\n")
                    observations, _, terminated, truncated, _ = env.step(actions)
                step += 1
                t = physics_steps * raw.physics_dt
                label = "requested" if t + 1e-9 >= next_snapshot else None
                rgb = frame(label, t)
                if label:
                    next_snapshot = next(snapshots, float("inf"))
                if writer is not None:
                    writer.append_data(rgb)
                if terminated.any() or truncated.any():
                    print(f"[VD11 checked Play] training termination at {t:.3f}s; original resets retained", flush=True)
                if args.real_time:
                    remaining = wall_start + t - time.perf_counter()
                    if remaining > 0:
                        time.sleep(min(remaining, raw.step_dt))
        except GeometryStop:
            pass
        t = physics_steps * raw.physics_dt
        if terminal:
            for i in terminal["env_ids"]:
                rgb = frame("invalid_geometry", t, i)
                if writer is not None:
                    writer.append_data(rgb)
        else:
            frame("end", t)
        save(output / "state_frames.json", frames)
        status = "invalid" if terminal else "uncertain" if counts["uncertain"] else "safe_sampled_geometry"
        summary = dict(status=status, terminal=terminal, duration_s=t, physics_steps=physics_steps,
            completed_requested_duration=terminal is None and t + 1e-9 >= args.duration,
            geometry_counts=dict(counts), geometry_counts_by_env=[dict(c) for c in per_env_counts],
            deployment_validation=False, output=str(output),
            no_reward_or_model_changes=True, no_training=True, no_robot_io=True)
        save(output / "summary.json", summary)
        result_file = os.environ.get("VD11_PLAY_RESULT_FILE")
        if result_file:
            # The launcher wrapper reads this after Kit's fast shutdown, which
            # can otherwise replace Python's nonzero result with exit status 0.
            save(Path(result_file), dict(exit_code=2 if terminal else 0, output=str(output), status=status))
        print(json.dumps(dict(output=str(output), status=status, duration_s=t,
            invalid_env_ids=terminal["env_ids"] if terminal else [],
            completed_requested_duration=summary["completed_requested_duration"]), indent=2), flush=True)
        return 2 if terminal else 0
    finally:
        if writer is not None:
            writer.close()
        physics_file.close()
        actor_file.close()
        geometry_file.close()
        env.close()


def main():
    AppLauncher, args = parser_with_launcher()
    # Keep Kit's normal fast shutdown: disabling it hangs this installed Kit
    # during plugin cleanup. The shell wrapper preserves the result separately.
    launcher = AppLauncher(args)
    try:
        return run(args, launcher.app)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
