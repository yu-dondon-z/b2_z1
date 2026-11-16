#!/usr/bin/env python3
"""Read-only VD12 Isaac Play, with real checkpoint curriculum and guard.

No optimizer, export or robot IO. Keep the Train task's physics, observations,
events and native terminations. Stop the evaluation at its first unsafe sample
before autoreset can hide it; this early stop is not a physical contact event.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from play_vd11_checked import (
    POLICY_TERMS, compare_training_settings, load_saved_env, save, sha256,
    side_view, validate_joint_selection_contract, validate_runtime_contract,
)

ROOT = Path(__file__).resolve().parents[2]
TASK = "Unitree-b2z1-VD12-Velocity"
DEFAULT_CHECKPOINT = ROOT / "logs/rsl_rl/b2_z1_vd12_geometry_guard_amp/2026-11-14_15-07-38_vd12_geometry_guard_long_20261114/model_2000.pt"
FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


class EvaluationStop(Exception):
    """Preserve the actual bad pose before the original autoreset boundary."""


def validate_checkpoint(checkpoint, agent, profile, saved_env, *, version=12):
    if version not in (12, 13, 14):
        raise ValueError('Unsupported Play version')
    experiment = {12: 'b2_z1_vd12_geometry_guard_amp', 13: 'b2_z1_vd13_mount75_amp',
                  14: 'b2_z1_vd14_posture_amp'}[version]
    if checkpoint.get("training_experiment_name") != experiment:
        raise ValueError(f"VD{version} Play requires an actual VD{version} checkpoint")
    if agent.get("experiment_name") != experiment:
        raise ValueError(f"Saved agent experiment is not VD{version}")
    iteration = checkpoint.get("iter")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Invalid VD12 checkpoint iteration")
    if (not isinstance(profile, dict) or profile.get("version") != 1
            or checkpoint.get("vd12_geometry_profile") != profile
            or saved_env.get("vd12_geometry_profile") != profile):
        raise ValueError("Saved/current VD12 geometry profiles differ")
    if "obs_norm_state_dict" in checkpoint or agent.get("empirical_normalization") is not False:
        raise ValueError("Expected disabled observation normalization")
    if "clip_actions" not in agent or agent["clip_actions"] is not None:
        raise ValueError("Expected saved unclipped actor contract")
    policy = agent.get("policy", {})
    if policy.get("activation") != "elu" or policy.get("actor_hidden_dims") != [512, 256, 128]:
        raise ValueError("Unexpected VD12 actor architecture")
    blend = checkpoint.get("vd5_curriculum_state", {}).get("ground_blend")
    if isinstance(blend, bool) or not isinstance(blend, (int, float)) or not math.isfinite(blend) or not 0 <= blend <= 1:
        raise ValueError("Missing or invalid saved ground curriculum")
    if version == 13:
        from vd13_play_contract import validate_mount_checkpoint
        validate_mount_checkpoint(checkpoint, agent, saved_env, profile)
    elif version == 14:
        from vd14_play_contract import validate_posture_checkpoint
        validate_posture_checkpoint(checkpoint, agent, saved_env, profile)


def run(args, simulation_app, output):
    import gymnasium as gym
    import imageio.v2 as imageio
    import torch
    import yaml

    import b2_z1.tasks  # noqa: F401
    version = getattr(args, 'evaluation_version', 12)
    task = f'Unitree-b2z1-VD{version}-Velocity'
    if version in (13, 14):
        from b2_z1.assets.vd13_runtime_contract import audit_vd13_runtime_asset as audit_vd10_runtime_asset
    else:
        from b2_z1.assets.vd10_runtime_contract import audit_vd10_runtime_asset
    from b2_z1.assets.vd11_geometry_audit import GunRobotGeometryAudit
    from b2_z1.rsl_rl.utils.vd5_curriculum_state import capture_vd5_curriculum_state, restore_vd5_curriculum_state
    from b2_z1.tasks.manager_based.b2_z1.mdp.tcp_frame import quat_apply, tcp_pose_w
    from isaaclab_tasks.utils import parse_env_cfg

    checkpoint_path = args.checkpoint.resolve(strict=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_agent = yaml.safe_load((checkpoint_path.parent / "params/agent.yaml").read_text())
    saved_env = load_saved_env(checkpoint_path.parent / "params/env.yaml")
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    validate_checkpoint(checkpoint, saved_agent, cfg.vd12_geometry_profile, saved_env, version=version)
    if version == 14:
        from b2_z1.assets.vd14_contract import validate_environment_profile
        validate_environment_profile(cfg)
    elif version == 13:
        from b2_z1.assets.vd13_contract import validate_environment_profile
        validate_environment_profile(cfg)
    settings = compare_training_settings(cfg, saved_env)
    cfg.viewer.resolution = (800, 450)
    actor = torch.nn.Sequential(torch.nn.Linear(70, 512), torch.nn.ELU(), torch.nn.Linear(512, 256),
        torch.nn.ELU(), torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 18)).to(cfg.sim.device).eval()
    actor.load_state_dict({k.removeprefix("actor."): v for k, v in checkpoint["model_state_dict"].items()
                           if k.startswith("actor.")}, strict=True)
    if not all(bool(torch.isfinite(parameter).all()) for parameter in actor.parameters()):
        raise ValueError("Nonfinite saved actor parameters")
    env = gym.make(task, cfg=cfg, render_mode=None if args.no_render else "rgb_array")
    raw, writer = env.unwrapped, None
    try:
        robot, guard = raw.scene["robot"], raw._vd12_gun_guard
        restore_vd5_curriculum_state(raw, checkpoint["vd5_curriculum_state"])
        restored = capture_vd5_curriculum_state(raw)
        if args.reset_mode == "nominal":
            raw.cfg.reference_state_initialization_prob = 0.0  # Declared differential, not training.
        observations, _ = env.reset(seed=args.seed)
        validate_runtime_contract(robot.joint_names, raw.observation_manager.active_terms["policy"], observations["policy"])
        import inspect

        resolved = {}
        for name, term in zip(raw.observation_manager.active_terms["policy"],
                              raw.observation_manager._group_obs_term_cfgs["policy"]):
            if name in ("joint_pos_rel", "joint_vel_rel"):
                asset = term.params.get("asset_cfg")
                if asset is None:
                    asset = inspect.signature(term.func).parameters["asset_cfg"].default
                if asset.name != "robot":
                    raise ValueError("Joint observation uses another asset")
                resolved[name] = asset.joint_ids
        action = raw.action_manager.get_term("JointPositionAction")
        validate_joint_selection_contract(action._joint_names, action._joint_ids,
            resolved["joint_pos_rel"], resolved["joint_vel_rel"])
        save(output / "asset_audit.json", audit_vd10_runtime_asset(robot))
        independent = GunRobotGeometryAudit(cfg.scene.robot.spawn.asset_path)
        initial_audit = independent.audit_batch(robot.joint_names, robot.data.joint_pos.detach().cpu().numpy())
        save(output / "initial_geometry.json", initial_audit)
        sensor = raw.scene.sensors["contact_forces"]
        foot_ids, _ = robot.find_bodies(list(FOOT_NAMES), preserve_order=True)
        hip_ids, _ = robot.find_bodies(['FL_hip', 'FR_hip', 'RL_hip', 'RR_hip'], preserve_order=True)
        foot_sensor_ids = [sensor.body_names.index(name) for name in FOOT_NAMES]
        tool_id = sensor.body_names.index("link06")
        base_id = robot.body_names.index("base_link")
        save(output / "manifest.json", dict(
            task=task, checkpoint=str(checkpoint_path), checkpoint_sha256=sha256(checkpoint_path),
            mount_profile=getattr(cfg, 'vd13_mount_profile', None),
            posture_profile=getattr(cfg, 'vd14_posture_profile', None),
            work_distance_m=cfg.commands.ee_target.vd5_seam_trailing_distance,
            iteration=checkpoint["iter"], geometry_profile=cfg.vd12_geometry_profile,
            geometry_registry=guard.geometry.metadata, body_names=list(robot.body_names),
            joint_names=list(robot.joint_names), joint_position_limits_rad=robot.data.joint_pos_limits.detach().cpu().tolist(),
            foot_names=list(FOOT_NAMES), actor_mode="deterministic mean", actor_observations=70,
            action_count=18, policy_terms=list(POLICY_TERMS), actor_extra_world_fields=[],
            joint_order_verified=True, training_settings_verified=settings,
            saved_curriculum=checkpoint["vd5_curriculum_state"], restored_curriculum=restored,
            reset_mode=args.reset_mode, effective_reference_fraction=raw.cfg.reference_state_initialization_prob,
            reset_override="nominal differential: RSI probability 0" if args.reset_mode == "nominal" else None,
            curriculum_frozen=False, target_or_drive_override=False, real_terminations_retained=True,
            physics_dt_s=raw.physics_dt, control_dt_s=raw.step_dt, num_envs=raw.num_envs, seed=args.seed,
            contact_sensor_update_period_s=sensor.cfg.update_period,
            command_source="unchanged training geometry teacher, not real camera perception",
            geometry_semantics="conservative exclusion, not measured contact or penetration depth",
            evaluation_stop="first exclusion sample before native autoreset; OR first native episode end",
            torque_note="No actual drive torque claimed; no PD estimate substituted for measured torque",
            video_enabled=not (args.no_render or args.no_video), video_fps=25,
            requested_duration_s=args.duration, optimizer_created=False, no_training=True,
            no_export=True, no_robot_io=True,
        ))
        terminal, physics_steps, actor_steps = None, 0, 0
        frames, last_rgb = [], None
        min_clearance = float("inf")
        next_snapshot = 0.5
        snapshot_times = iter((1., 1.5, 2., 5., 10., 20., 30., 60.))

        def frame(label, t, env_id=0, *, append_video=False):
            nonlocal last_rgb
            if args.no_render:
                return
            eye, target = side_view(robot.data.body_link_pos_w[env_id, base_id].detach().cpu().numpy(),
                                    robot.data.body_link_quat_w[env_id, base_id].detach().cpu().numpy())
            raw.sim.set_camera_view(eye=eye.tolist(), target=target.tolist())
            rgb = env.render()
            for _ in range(8):
                if rgb is not None and np.any(rgb):
                    break
                rgb = env.render()
            if rgb is None or not np.any(rgb):
                raise RuntimeError("No real viewport image was produced")
            last_rgb = rgb
            if writer is not None and append_video:
                writer.append_data(rgb)
            if label:
                name = f"{len(frames):02d}_{label}_env{env_id}_{t:.3f}s.png"
                imageio.imwrite(output / name, rgb)
                frames.append(dict(path=name, time_s=t, env_id=env_id, no_state_edits=True))

        if not args.no_render and not args.no_video:
            writer = imageio.get_writer(output / "play.mp4", fps=25, codec="libx264", macro_block_size=2)
        frame("reset", 0., append_video=True)
        initial = guard.geometry(robot.data.body_pos_w, robot.data.body_quat_w)
        initial_bad = (~initial["input_valid"]) | (initial["minimum_clearance_lower_bound_m"] < 0)
        if bool(initial_bad.any()):
            terminal = dict(reason="exclusion_at_reset", time_s=0.,
                env_ids=initial_bad.nonzero(as_tuple=True)[0].cpu().tolist(),
                native_termination_observed=False)

        with ExitStack() as stack:
            physics_file = stack.enter_context(gzip.open(output / "physics_200hz.jsonl.gz", "wt", encoding="utf-8"))
            actor_file = stack.enter_context(gzip.open(output / "actor_50hz.jsonl.gz", "wt", encoding="utf-8"))

            def record_physics():
                nonlocal physics_steps, min_clearance, terminal, next_snapshot
                physics_steps += 1
                t = physics_steps * raw.physics_dt
                result = guard.last_result
                minimum = result["minimum_clearance_lower_bound_m"]
                valid = result["input_valid"]
                min_clearance = min(min_clearance, float(minimum.min().item()))
                tcp, orientation = tcp_pose_w(robot)
                goal = raw.command_manager.get_term("ee_target")
                vector = tcp.new_tensor((1., 0., 0.)).expand_as(tcp)
                axis_error = torch.acos((quat_apply(orientation, vector)
                    * quat_apply(goal._get_target_quat_w(), vector)).sum(-1).clamp(-1., 1.))

                def values(tensor):
                    return tensor.detach().cpu().tolist()

                row = dict(time_s=t, physics_step=raw._sim_step_counter,
                    joint_position_rad=values(robot.data.joint_pos),
                    joint_velocity_radps=values(robot.data.joint_vel),
                    joint_target_rad=values(robot.data.joint_pos_target),
                    episode_actor_step=values(raw.episode_length_buf),
                    base_position_w=values(robot.data.root_pos_w), base_quat_wxyz=values(robot.data.root_quat_w),
                    foot_position_w=values(robot.data.body_link_pos_w[:, foot_ids]),
                    hip_position_w=values(robot.data.body_link_pos_w[:, hip_ids]),
                    tcp_position_w=values(tcp), tcp_command_world_50hz_zoh=values(goal._get_target_pos_w()),
                    gun_axis_error_rad=values(axis_error), tool_contact_force_w=values(sensor.data.net_forces_w[:, tool_id]),
                    foot_contact_force_w=values(sensor.data.net_forces_w[:, foot_sensor_ids]),
                    foot_velocity_w=values(robot.data.body_link_lin_vel_w[:, foot_ids]),
                    foot_surface_height_m=values(robot.data.body_link_pos_w[:, foot_ids, 2] - raw.scene.env_origins[:, 2:3] - .032),
                    minimum_clearance_lower_bound_m=values(minimum), geometry_input_valid=values(valid),
                    minimum_pair_index=values(result["minimum_pair_index"]), exclusion_latched=values(guard.window.violated))
                physics_file.write(json.dumps(row, allow_nan=False) + "\n")
                bad = (~valid) | guard.window.violated
                if bool(bad.any()):
                    ids = bad.nonzero(as_tuple=True)[0].cpu().tolist()
                    q = robot.data.joint_pos.detach().cpu().numpy()
                    save(output / "terminal_independent_geometry.json", independent.audit_batch(robot.joint_names, q))
                    terminal = dict(reason="conservative_exclusion", time_s=t, env_ids=ids,
                        minimum_clearance_lower_bound_m=values(minimum),
                        minimum_pair_names=[guard.geometry.pair_names[int(result["minimum_pair_index"][i])]
                            if bool(valid[i]) else None for i in ids],
                        native_termination_observed=False,
                        note="Evaluation stopped before native termination/autoreset; no contact or penetration claim")
                    for env_id in ids:
                        frame("exclusion", t, env_id)
                    raise EvaluationStop()
                if physics_steps % 4 == 0:
                    label = "requested" if t + 1e-9 >= next_snapshot else None
                    if label:
                        next_snapshot = next(snapshot_times, float("inf"))
                    if label or physics_steps % 8 == 0 or args.viewer:
                        frame(label, t, append_video=physics_steps % 8 == 0)

            raw._physics_step_callbacks.append(record_physics)
            start = time.perf_counter()
            try:
                while terminal is None and physics_steps * raw.physics_dt + 1e-9 < args.duration and simulation_app.is_running():
                    with torch.inference_mode():
                        actions = actor(observations["policy"])
                        if tuple(actions.shape) != (raw.num_envs, 18) or not bool(torch.isfinite(actions).all()):
                            raise ValueError("Nonfinite or invalid actor output")
                        actor_file.write(json.dumps(dict(time_s=actor_steps * raw.step_dt,
                            observations=observations["policy"].detach().cpu().tolist(),
                            actions=actions.detach().cpu().tolist()), allow_nan=False) + "\n")
                        observations, _, terminated, truncated, _ = env.step(actions)
                    actor_steps += 1
                    if bool((terminated | truncated).any()):
                        ids = (terminated | truncated).nonzero(as_tuple=True)[0].cpu().tolist()
                        terminal = dict(reason="native_episode_end", time_s=physics_steps * raw.physics_dt,
                            env_ids=ids, native_termination_observed=True,
                            causes={name: (raw.termination_manager.get_term(name) & (terminated | truncated))
                                    .nonzero(as_tuple=True)[0].cpu().tolist()
                                    for name in raw.termination_manager.active_terms},
                            note="Stopped at the first episode end; metrics use only pre-reset physics samples")
                    if args.real_time:
                        remaining = start + physics_steps * raw.physics_dt - time.perf_counter()
                        if remaining > 0:
                            time.sleep(min(remaining, raw.step_dt))
            except EvaluationStop:
                pass
            finally:
                raw._physics_step_callbacks.remove(record_physics)
        elapsed = physics_steps * raw.physics_dt
        completed = elapsed + 1e-9 >= args.duration
        timeout_only = terminal is not None and terminal["reason"] == "native_episode_end" and all(
            not ids for name, ids in terminal["causes"].items() if name != "time_out")
        failed = terminal is not None and not (timeout_only and completed)
        if terminal is None:
            frame("end", elapsed)
            save(output / "final_independent_geometry.json", independent.audit_batch(
                robot.joint_names, robot.data.joint_pos.detach().cpu().numpy()))
        status = "rejected" if failed else "completed_sampled_play" if completed else "interrupted"
        summary = dict(status=status, terminal=terminal, duration_s=elapsed, requested_duration_s=args.duration,
            physics_steps=physics_steps, actor_steps_completed=actor_steps,
            completed_requested_duration=completed, minimum_clearance_lower_bound_m=min_clearance if math.isfinite(min_clearance) else None,
            geometry_profile=cfg.vd12_geometry_profile, checkpoint=str(checkpoint_path), checkpoint_iteration=checkpoint["iter"],
            reset_mode=args.reset_mode, output=str(output), initial_episode_only=True,
            deployment_accepted=False, no_training=True, no_robot_io=True)
        save(output / "state_frames.json", frames)
        save(output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return 2 if failed else 0 if completed else 3
    finally:
        if writer is not None:
            writer.close()
        env.close()


def main(*, version=12):
    from isaaclab.app import AppLauncher

    if version == 14:
        from vd14_play_contract import DEFAULT_CHECKPOINT as default_checkpoint
    elif version == 13:
        from vd13_play_contract import DEFAULT_CHECKPOINT as default_checkpoint
    elif version == 12:
        default_checkpoint = DEFAULT_CHECKPOINT
    else:
        raise ValueError('Unsupported Play version')
    parser = argparse.ArgumentParser(description=(__doc__ if version == 12 else
        f'VD{version}: actual Isaac Play, preserved Train settings and geometry teacher. Not image-driven deployment.'))
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--duration", type=float, default=30.)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset-mode", choices=("mixed", "nominal"), default="nominal")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / (f"outputs/vd{version}_checked_play" if version in (13, 14) else "outputs/vd12_checked_play_20261114"))
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True, evaluation_version=version)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or not 0 < args.duration <= 60 or not 1 <= args.num_envs <= 8:
        parser.error("duration must be (0,60] seconds and num_envs 1..8")
    if args.viewer and args.no_render:
        parser.error("--viewer and --no-render are mutually exclusive")
    args.enable_cameras = not args.no_render
    if args.viewer:
        args.headless = False
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=time.strftime("play_%Y%m%d_%H%M%S_"), dir=args.output_root))
    print(f"[VD{version} checked Play] output={output}", flush=True)
    code = 1
    launcher = AppLauncher(args)
    try:
        code = run(args, launcher.app, output)
    except BaseException as exc:
        save(output / "error.json", dict(type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        result = dict(exit_code=code, output=str(output), deployment_accepted=False)
        save(output / "process_result.json", result)
        result_variable = f'VD{version}_PLAY_RESULT_FILE'
        if os.environ.get(result_variable):
            save(Path(os.environ[result_variable]), result)
        launcher.app.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
