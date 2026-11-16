#!/usr/bin/env python3
"""No-learning VD12 guard integration probe with the preserved VD11 actor.

This tests rejection of an already known unsafe policy, not policy improvement.
It uses real physics, real original contacts and original episode reset logic.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "logs/rsl_rl/b2_z1_vd11_motion_quality_amp/2026-11-13_17-56-49_vd11_motion_quality_long_20261113/model_5999.pt"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def classify_enforcement(first_bad, termination, decimation, ground_blend):
    """Never classify an observed but unenforced exclusion as a safe probe."""
    verified = first_bad is not None and termination is not None
    if verified:
        verified = (
            0 <= termination["physics_step"] - first_bad["physics_step"] < decimation
            and bool(first_bad["affected_env_ids"])
            and set(first_bad["affected_env_ids"]).issubset(termination["env_ids"])
        )
    if verified:
        return "guard_enforcement_verified", True, 0
    if first_bad is not None or termination is not None:
        return "guard_enforcement_failed", False, 2
    return "no_exclusion_observed_in_short_probe", False, 0 if ground_blend == 0 else 2


def run(args, output):
    import gymnasium as gym
    import torch
    import yaml

    import b2_z1.tasks  # noqa: F401
    from b2_z1.assets.vd10_runtime_contract import audit_vd10_runtime_asset
    from b2_z1.rsl_rl.utils.vd5_curriculum_state import restore_vd5_curriculum_state
    from isaaclab_tasks.utils import parse_env_cfg
    from play_vd11_checked import (
        compare_training_settings, load_saved_env, sha256, validate_actor_contract,
        validate_runtime_contract,
    )

    checkpoint_path = args.checkpoint.resolve(strict=True)
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    source_agent = yaml.safe_load((checkpoint_path.parent / "params/agent.yaml").read_text())
    validate_actor_contract(checkpoint, source_agent)
    cfg = parse_env_cfg("Unitree-b2z1-VD12-Velocity", device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    settings = compare_training_settings(cfg, load_saved_env(checkpoint_path.parent / "params/env.yaml"))
    actor = torch.nn.Sequential(torch.nn.Linear(70, 512), torch.nn.ELU(), torch.nn.Linear(512, 256),
        torch.nn.ELU(), torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 18)).to(cfg.sim.device).eval()
    actor.load_state_dict({k.removeprefix("actor."): v for k, v in checkpoint["model_state_dict"].items()
                           if k.startswith("actor.")}, strict=True)
    env = gym.make("Unitree-b2z1-VD12-Velocity", cfg=cfg)
    raw = env.unwrapped
    physics_path = output / "physics_200hz.jsonl.gz"
    try:
        restore_vd5_curriculum_state(raw, checkpoint["vd5_curriculum_state"])
        goal = raw.command_manager.get_term("ee_target")
        goal._vd5_ground_blend = args.ground_blend
        goal._vd5_ground_transition_steps = 0  # Explicit diagnostic hold, not training.
        if args.reset_mode == "nominal":
            raw.cfg.reference_state_initialization_prob = 0.0
        observations, _ = env.reset(seed=args.seed)
        robot, guard = raw.scene["robot"], raw._vd12_gun_guard
        validate_runtime_contract(robot.joint_names, raw.observation_manager.active_terms["policy"],
                                  observations["policy"])
        with torch.inference_mode():
            initial = guard.geometry(robot.data.body_pos_w, robot.data.body_quat_w)
        initial_min = initial["minimum_clearance_lower_bound_m"].cpu().tolist()
        save(output / "asset_audit.json", audit_vd10_runtime_asset(robot))
        save(output / "manifest.json", dict(
            task="Unitree-b2z1-VD12-Velocity", actor_source_experiment="VD11, not a trained VD12 actor",
            checkpoint=str(checkpoint_path), checkpoint_sha256=sha256(checkpoint_path),
            geometry_profile=cfg.vd12_geometry_profile, geometry_registry=guard.geometry.metadata,
            body_names=list(robot.body_names), joint_names=list(robot.joint_names),
            source_physics_and_actor_settings_verified=settings,
            actor_observations=70, action_count=18, actor_extra_world_fields=[],
            num_envs=raw.num_envs, seed=args.seed, reset_mode=args.reset_mode,
            reference_initialization_prob=raw.cfg.reference_state_initialization_prob,
            diagnostic_ground_blend=args.ground_blend,
            training_ground_curriculum_changed=False,
            initial_minimum_clearance_lower_bound_m=initial_min,
            geometry_semantics="conservative exclusion zone, NOT measured self-contact or penetration",
            optimizer_created=False, actor_updated=False, exported=False, robot_io=False,
            contact_sensors_and_drive_gains="unchanged", physics_period_s=raw.physics_dt,
            episode_termination_period_s=raw.step_dt, substep_violation_latched=True,
        ))
        if not bool(initial["input_valid"].all()) or min(initial_min) < 0:
            raise RuntimeError("Reset entered the geometry exclusion zone; refusing the probe")
        first_bad = None
        termination = None
        sample_count = 0
        all_minimum = min(initial_min)
        tool_id = raw.scene["contact_forces"].body_names.index("link06")
        with gzip.open(physics_path, "wt", encoding="utf-8") as stream:
            def record():
                nonlocal first_bad, sample_count, all_minimum
                result = guard.last_result
                minima = result["minimum_clearance_lower_bound_m"].detach().cpu().tolist()
                bad = guard.window.violated.nonzero(as_tuple=True)[0].detach().cpu().tolist()
                sample_count += 1
                all_minimum = min(all_minimum, min(minima))
                entry = dict(
                    physics_step=raw._sim_step_counter, time_s=sample_count * raw.physics_dt,
                    episode_actor_step=raw.episode_length_buf.detach().cpu().tolist(),
                    minimum_clearance_lower_bound_m=minima,
                    minimum_pair_index=result["minimum_pair_index"].detach().cpu().tolist(),
                    exclusion_latched=guard.window.violated.detach().cpu().tolist(),
                    first_bad_physics_step=guard.window.first_bad_physics_step.detach().cpu().tolist(),
                    actual_q_rad=robot.data.joint_pos.detach().cpu().tolist(),
                    actual_qd_rad_s=robot.data.joint_vel.detach().cpu().tolist(),
                    joint_position_target_rad=robot.data.joint_pos_target.detach().cpu().tolist(),
                    tool_contact_force_world_n=raw.scene["contact_forces"].data.net_forces_w[:, tool_id]
                    .detach().cpu().tolist(),
                )
                if bad and first_bad is None:
                    first_bad = dict(entry, affected_env_ids=bad)
                stream.write(json.dumps(entry, allow_nan=False) + "\n")

            raw._physics_step_callbacks.append(record)
            actual_steps = 0
            for actor_step in range(math.ceil(args.duration / raw.step_dt)):
                with torch.inference_mode():
                    observations, _, terminated, truncated, _ = env.step(actor(observations["policy"]))
                actual_steps = actor_step + 1
                # get_term() is sticky; attribute ONLY actual current dones.
                rejected = raw.termination_manager.get_term("vd12_gun_exclusion") & (terminated | truncated)
                if bool(rejected.any()):
                    termination = dict(
                        actor_step=actual_steps, physics_step=raw._sim_step_counter,
                        time_s=actual_steps * raw.step_dt,
                        env_ids=rejected.nonzero(as_tuple=True)[0].detach().cpu().tolist(),
                    )
                    break
            raw._physics_step_callbacks.remove(record)
        status, verified, exit_code = classify_enforcement(
            first_bad, termination, raw.cfg.decimation, args.ground_blend,
        )
        summary = dict(
            status=status,
            guard_enforcement_verified=verified, deployment_accepted=False,
            policy_improvement_tested=False, optimizer_called=False,
            reset_initial_minimum_clearance_lower_bound_m=initial_min,
            minimum_clearance_lower_bound_m=all_minimum,
            first_exclusion_sample=first_bad, actual_episode_termination=termination,
            physics_sample_count=sample_count, actor_steps=actual_steps,
            ground_blend=args.ground_blend, reset_mode=args.reset_mode,
            note="Expected rejection of old unsafe VD11 actor; not evidence of learned VD12 success.",
        )
        save(output / "summary.json", summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "first_exclusion_sample"}, indent=2), flush=True)
        return exit_code
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ground-blend", type=int, choices=(0, 1), default=1)
    parser.add_argument("--reset-mode", choices=("mixed", "nominal"), default="mixed")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/vd12_guard_probe_20261114")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 16 or not math.isfinite(args.duration) or not 0 < args.duration <= 10:
        parser.error("Probe is limited to 1..16 environments and (0,10] seconds")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=time.strftime("probe_%Y%m%d_%H%M%S_"), dir=args.output_root))
    print(f"[VD12 guard probe] output={output}", flush=True)
    launcher = AppLauncher(args)
    code = 1
    try:
        code = run(args, output)
    except BaseException as exc:
        save(output / "error.json", {"type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        result = dict(exit_code=code, output=str(output), deployment_accepted=False)
        save(output / "process_result.json", result)
        if os.environ.get("VD12_PROBE_RESULT_FILE"):
            save(Path(os.environ["VD12_PROBE_RESULT_FILE"]), result)
        launcher.app.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
