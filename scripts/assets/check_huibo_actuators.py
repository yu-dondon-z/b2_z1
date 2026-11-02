#!/usr/bin/env python3
"""Exercise each Huibo simulation joint through the configured action term.

This is a fixed-base, zero-gravity physics diagnostic, without a policy,
camera rendering, randomization or hardware connection. Success requires
measured joint motion, rather than the position-target tensor changing.
Run with the Isaac Lab Python environment, for example::

    python scripts/assets/check_huibo_actuators.py \
        --output outputs/huibo_actuators_NEW_DIRECTORY --device cuda:0

The output directory must not exist. The report records partial results
before raising if an assertion fails. Only the synthetic Huibo task is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, help="New directory for report.json")
parser.add_argument("--hold-steps", type=int, default=50, help="Policy steps per move and return phase")
parser.add_argument("--warmup-steps", type=int, default=50)
parser.add_argument("--delta-rad", type=float, default=0.08)
parser.add_argument("--tolerance-rad", type=float, default=0.02)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, enable_cameras=False)
args_cli = parser.parse_args()
if not 40 <= args_cli.hold_steps <= 60:
    parser.error("--hold-steps must be between 40 and 60")
if args_cli.warmup_steps < 20:
    parser.error("--warmup-steps must be at least 20")
if not 0.0 < args_cli.delta_rad <= 0.10:
    parser.error("--delta-rad must be in (0, 0.10]")
if not 0.0 < args_cli.tolerance_rad < args_cli.delta_rad:
    parser.error("--tolerance-rad must be positive and smaller than --delta-rad")
if args_cli.enable_cameras:
    parser.error("This diagnostic does not enable RTX cameras")
output = Path(args_cli.output).expanduser().resolve()
output.mkdir(parents=True, exist_ok=False)
launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import torch
from pxr import Usd, UsdPhysics

from isaaclab_tasks.utils import parse_env_cfg
import b2_z1.tasks  # noqa: F401
from b2_z1.assets.b2_huibo import HUIBO_ARM_JOINT_NAMES


def _list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise AssertionError(f"Non-finite {name}")


def _physics_metadata(env, robot):
    view = robot.root_physx_view
    fields = {
        "joint_position_limits_rad": view.get_dof_limits(),
        "joint_effort_limits_Nm": view.get_dof_max_forces(),
        "joint_velocity_limits_rad_s": view.get_dof_max_velocities(),
        "joint_stiffness_Nm_rad": view.get_dof_stiffnesses(),
        "joint_damping_Nm_s_rad": view.get_dof_dampings(),
        "body_masses_kg": view.get_masses(),
    }
    for name, value in fields.items():
        _finite(value, name)
    result = {name: _list(value[0]) for name, value in fields.items()}
    result["body_order"] = list(robot.body_names)
    result["body_mass_by_name_kg"] = dict(zip(robot.body_names, result["body_masses_kg"]))
    result["actuators"] = {}
    for name, actuator in robot.actuators.items():
        result["actuators"][name] = {
            "class": type(actuator).__name__,
            "joint_names": list(actuator.joint_names),
            "joint_indices": (
                list(range(robot.num_joints))[actuator.joint_indices]
                if isinstance(actuator.joint_indices, slice)
                else _list(actuator.joint_indices)
            ),
            "effort_limit_sim_Nm": _list(actuator.effort_limit_sim),
            "velocity_limit_sim_rad_s": _list(actuator.velocity_limit_sim),
            "stiffness": _list(actuator.stiffness),
            "damping": _list(actuator.damping),
        }
    robot_prim = env.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot")
    enabled = []
    # URDF mesh colliders are instanced USD assets. Default PrimRange skips
    # instance proxies and would report only the non-instanced tool shape.
    for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()):
                enabled.append(str(prim.GetPath()))
    result["enabled_collision_prims"] = enabled
    result["arm_links_with_enabled_collision"] = [
        name for name in ("right_arm_base_link", *(f"right_link{i}" for i in range(1, 8)))
        if any(f"/{name}/" in path for path in enabled)
    ]
    if len(result["arm_links_with_enabled_collision"]) != 8:
        raise AssertionError("Imported Huibo links are missing enabled mesh collision shapes")
    result["contact_sensor_activation_configured"] = bool(robot.cfg.spawn.activate_contact_sensors)
    result["self_collision_enabled_in_config"] = bool(
        robot.cfg.spawn.articulation_props.enabled_self_collisions
    )
    return result


def main():
    task = "Unitree-b2huibo-AMP-DualVision"
    report = {
        "task": task,
        "passed": False,
        "scope": "Simulated joint response through actual action mapping; no policy or hardware",
        "settings": {
            "gravity_m_s2": [0.0, 0.0, 0.0],
            "fixed_base": True,
            "randomization_events": [],
            "reward_terms": [],
            "termination_terms": [],
            "reference_state_initialization": False,
            "headless": bool(args_cli.headless),
            "cameras_enabled": False,
            "delta_rad": args_cli.delta_rad,
            "tolerance_rad": args_cli.tolerance_rad,
            "hold_steps": args_cli.hold_steps,
            "warmup_steps": args_cli.warmup_steps,
        },
        "force_field_meanings": {
            "physx_projected_joint_force_Nm": (
                "PhysX incoming joint force projected on the revolute axis; "
                "a simulated physical reaction, not a hardware torque measurement"
            ),
            "actuator_estimated_torque_Nm": (
                "Isaac Lab actuator estimate; implicit PD is solved inside PhysX, "
                "so this field is not labelled as a measured drive torque"
            ),
        },
        "axes": [],
    }
    env = None
    try:
        cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1)
        cfg.sim.gravity = (0.0, 0.0, 0.0)
        cfg.scene.robot.spawn.fix_base = True
        cfg.reference_state_initialization = False
        cfg.reference_state_initialization_prob = 0.0
        cfg.events = {}
        cfg.rewards = {}
        cfg.terminations = {}
        cfg.curriculum = {}
        cfg.episode_length_s = 100000.0
        cfg.commands.ee_target.debug_vis = False
        cfg.commands.ee_target.show_rear_camera_frustum = False
        cfg.commands.base_velocity.debug_vis = False
        cfg.observations.policy.enable_corruption = False
        env = gym.make(task, cfg=cfg).unwrapped
        env.reset()
        robot = env.scene["robot"]
        term = env.action_manager.get_term("JointPositionAction")
        action_joint_names = list(term._joint_names)
        joint_names = list(robot.joint_names)
        assert robot.num_joints == 19, f"Expected 19 joints, got {robot.num_joints}"
        assert env.action_manager.total_action_dim == 19
        assert len(action_joint_names) == 19 and set(action_joint_names) == set(joint_names)
        scale = torch.as_tensor(term._scale, device=env.device).expand(1, 19)
        _finite(scale, "action scale")
        assert bool(torch.allclose(scale, torch.full_like(scale, 0.25))), "Unexpected action scale"
        report.update(
            articulation_joint_order=joint_names,
            action_joint_order=action_joint_names,
            action_term=type(term).__name__,
            action_scale=_list(scale[0]),
            action_filter_alpha=float(term._alpha),
            physics_dt_s=float(env.physics_dt),
            policy_dt_s=float(env.step_dt),
        )
        report["physics"] = _physics_metadata(env, robot)
        nominal = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(nominal, torch.zeros_like(nominal))
        robot.set_joint_position_target(nominal)
        env.scene.write_data_to_sim()
        env.sim.forward()
        zero_action = torch.zeros((1, 19), device=env.device)
        force_getter = robot.root_physx_view.get_dof_projected_joint_forces
        force_available = None

        def sample():
            nonlocal force_available
            q = robot.data.joint_pos[0].detach().clone()
            velocity = robot.data.joint_vel[0].detach().clone()
            torque = robot.data.applied_torque[0].detach().clone()
            for name, value in (("joint position", q), ("joint velocity", velocity), ("actuator torque estimate", torque)):
                _finite(value, name)
            projected = None
            if force_available is not False:
                try:
                    projected = force_getter()[0].detach().clone()
                except Exception as exc:
                    force_available = False
                    report["physx_projected_force_unavailable"] = str(exc)
                else:
                    force_available = True
                    _finite(projected, "PhysX projected joint force")
            return {
                "q_rad": q,
                "velocity_rad_s": velocity,
                "actuator_estimated_torque_Nm": torque,
                "physx_projected_joint_force_Nm": projected,
            }

        def step_and_sample(action):
            _, _, terminated, truncated, _ = env.step(action)
            assert not bool(terminated.any() or truncated.any()), "Unexpected environment reset"
            return sample()

        with torch.inference_mode():
            for _ in range(args_cli.warmup_steps):
                step_and_sample(zero_action)
            baseline = robot.data.joint_pos[0].detach().clone()
            baseline_error = baseline - nominal[0]
            report["nominal_q_rad"] = _list(nominal[0])
            report["settled_baseline_q_rad"] = _list(baseline)
            report["baseline_error_rad"] = _list(baseline_error)
            assert float(baseline_error.abs().max()) < args_cli.tolerance_rad, "Zero-action baseline did not settle"
            position_limits = robot.root_physx_view.get_dof_limits()[0].to(env.device)
            for arm_name in HUIBO_ARM_JOINT_NAMES:
                joint_id = joint_names.index(arm_name)
                action_id = action_joint_names.index(arm_name)
                target = float(nominal[0, joint_id]) + args_cli.delta_rad
                assert float(position_limits[joint_id, 0]) < target < float(position_limits[joint_id, 1]), (
                    f"Requested target exceeds imported limit: {arm_name}"
                )
                action = zero_action.clone()
                action[0, action_id] = args_cli.delta_rad / scale[0, action_id]
                move = [step_and_sample(action) for _ in range(args_cli.hold_steps)]
                moved_q = torch.stack([item["q_rad"] for item in move])
                delta = moved_q - baseline
                other_ids = [index for index in range(19) if index != joint_id]
                error = float(moved_q[-1, joint_id]) - target
                other_drift = float(delta[:, other_ids].abs().max())
                returned = [step_and_sample(zero_action) for _ in range(args_cli.hold_steps)]
                return_q = torch.stack([item["q_rad"] for item in returned])
                return_error = float((return_q[-1] - nominal[0]).abs().max())
                all_samples = move + returned
                peak_velocity = float(torch.stack([item["velocity_rad_s"] for item in all_samples]).abs().max())
                estimated_torques = torch.stack([item["actuator_estimated_torque_Nm"] for item in all_samples])
                axis = {
                    "joint_name": arm_name,
                    "articulation_joint_index": joint_id,
                    "action_index": action_id,
                    "raw_action": float(action[0, action_id]),
                    "target_q_rad": target,
                    "requested_delta_rad": args_cli.delta_rad,
                    "final_q_rad": float(moved_q[-1, joint_id]),
                    "final_delta_from_baseline_rad": float(delta[-1, joint_id]),
                    "peak_delta_from_baseline_rad": float(delta[:, joint_id].max()),
                    "final_target_error_rad": error,
                    "peak_other_joint_drift_rad": other_drift,
                    "final_delta_all_joints_rad": _list(delta[-1]),
                    "peak_absolute_delta_all_joints_rad": _list(delta.abs().amax(dim=0)),
                    "return_max_error_all_joints_rad": return_error,
                    "peak_absolute_velocity_all_joints_rad_s": peak_velocity,
                    "peak_absolute_estimated_torque_all_joints_Nm": _list(estimated_torques.abs().amax(dim=0)),
                    "passed": (
                        abs(error) <= args_cli.tolerance_rad
                        and other_drift <= args_cli.tolerance_rad
                        and return_error <= args_cli.tolerance_rad
                    ),
                    "samples": [
                        {
                            "phase": "move" if index < args_cli.hold_steps else "return",
                            "phase_step": index % args_cli.hold_steps + 1,
                            **{key: _list(value) for key, value in item.items()},
                        }
                        for index, item in enumerate(all_samples)
                    ],
                }
                projected_samples = [item["physx_projected_joint_force_Nm"] for item in all_samples]
                if all(value is not None for value in projected_samples):
                    axis["peak_absolute_physx_projected_joint_force_all_joints_Nm"] = _list(
                        torch.stack(projected_samples).abs().amax(dim=0)
                    )
                report["axes"].append(axis)
                print(
                    f"{arm_name}: measured delta={axis['final_delta_from_baseline_rad']:.6f} rad, "
                    f"target error={error:.6f}, other drift={other_drift:.6f}, "
                    f"return error={return_error:.6f}, passed={axis['passed']}",
                    flush=True,
                )
            report["passed"] = len(report["axes"]) == 7 and all(axis["passed"] for axis in report["axes"])
            assert report["passed"], "One or more measured joint-response checks failed; inspect report.json"
    except BaseException as exc:
        report["passed"] = False
        report["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Report: {output / 'report.json'}", flush=True)
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Kit's normal fast close exits the process before Python prints an
        # unhandled exception. Preserve the traceback and failure exit here.
        traceback.print_exc()
        raise
    else:
        simulation_app.close()
