#!/usr/bin/env python3
"""CPU-only full-gun IK witnesses for the existing, final-ground 2 s startup.

This is NOT the older ground-curriculum/linear-pitch check.  It reproduces
VD5StartupSeamCommand's quintic position blend and shortest quaternion nlerp,
then VD10SafeSeamCommand's *current-attitude* end-rim correction.  No command,
model, policy, simulator, collision mask, or training configuration is changed.

Two nominal straight-line cases are checked: a standing base and an ideal base
following its existing 50 Hz, zero-order-held startup velocity command.  Legs
retain their authored nominal reset positions; no moving-gait claim is made.
Every accepted IK witness checks full TCP attitude, hard joint limits, the
whole finite tube against all authored URDF colliders, and its ground surface.
Finite sampling and nearby joint solutions do NOT certify continuous motion.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import tempfile

sys.dont_write_bytecode = True

import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/b2_z1/b2_z1"
TASK = SOURCE / "tasks/manager_based/b2_z1"
URDF = SOURCE / "assets/b2_z1/b2_z1_vd10_hollow_gun.urdf"
FK_SOURCE = ROOT / "scripts/rsl_rl/analyze_vd5_reachability.py"
OLD_CHECK_SOURCE = ROOT / "scripts/diagnostics/check_vd10_startup_geometry.py"
AUDIT_SOURCE = SOURCE / "assets/vd11_geometry_audit.py"
ARM_NAMES = tuple(f"joint{i}" for i in range(1, 7))
CONTROL_DT = 0.02
POSITION_TOL = 1e-6
ROTATION_TOL = 1e-5


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numeric_expression(node):
    """Read numeric source constants without importing Isaac or using eval."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (float, int)):
        return node.value
    if isinstance(node, ast.Attribute) and ast.unparse(node) == "math.pi":
        return math.pi
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = numeric_expression(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = numeric_expression(node.left), numeric_expression(node.right)
        operations = {ast.Add: lambda: left + right, ast.Sub: lambda: left - right,
                      ast.Mult: lambda: left * right, ast.Div: lambda: left / right}
        if type(node.op) in operations:
            return operations[type(node.op)]()
    raise ValueError("Unsupported numeric source expression: " + ast.unparse(node))


def source_assignment(path, target):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and any(ast.unparse(item) == target for item in node.targets):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, TypeError):
                return numeric_expression(node.value)
    raise ValueError(f"Missing source assignment {target} in {path}")


def source_call_keyword(path, call_suffix, keyword):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith(call_suffix):
            for item in node.keywords:
                if item.arg == keyword:
                    return ast.literal_eval(item.value)
    raise ValueError(f"Missing {call_suffix}({keyword}=...) in {path}")


def source_profile():
    hollow = TASK / "b2_z1_vd10_env_cfg.py"
    startup = TASK / "b2_z1_vd5_startup_env_cfg.py"
    command = TASK / "mdp/commands/vd10_safe_seam_command.py"
    tcp = TASK / "mdp/tcp_frame.py"
    reset = None
    for node in ast.walk(ast.parse(hollow.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "robot.init_state.joint_pos.update":
            reset = ast.literal_eval(node.args[0])
    if reset is None:
        raise ValueError("Cannot find existing VD10 reset joint pose")
    duration = source_assignment(startup, "cfg.commands.ee_target.vd5_startup_duration_s")
    if duration != source_assignment(startup, "cfg.commands.base_velocity.vd5_startup_duration_s"):
        raise ValueError("Base and tool startup clocks differ; this nominal model is insufficient")
    return dict(reset_arm_q_rad=[reset[name] for name in ARM_NAMES], duration_s=duration,
                base_height_m=source_assignment(startup, "cfg.vd5_nominal_base_height_m"),
                trailing_distance_m=source_assignment(hollow, "goal.vd5_seam_trailing_distance"),
                speed_m_s=source_call_keyword(TASK / "b2_z1_env_cfg.py", "CurveVelocityCommandCfg", "forward_speed"),
                seam_pitch_rad=source_assignment(command, "SEAM_PITCH_RAD"),
                working_clearance_m=source_assignment(command, "WORKING_SURFACE_CLEARANCE_M"),
                tcp_in_link06_m=source_assignment(tcp, "HOLLOW_GUN_TCP_POS_IN_LINK06"),
                tcp_quat_wxyz=source_assignment(tcp, "HOLLOW_GUN_TCP_QUAT_IN_LINK06"))


def quintic_fraction(time_s, duration_s):
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Startup duration must be finite and positive")
    u = float(np.clip(time_s / duration_s, 0., 1.))
    return u**3 * (10. + u * (-15. + 6. * u))


def nlerp(first_xyzw, second_xyzw, fraction):
    first, second = np.asarray(first_xyzw, dtype=float), np.asarray(second_xyzw, dtype=float)
    first, second = first / np.linalg.norm(first), second / np.linalg.norm(second)
    if first @ second < 0:
        second = -second
    result = first + fraction * (second - first)
    return result / np.linalg.norm(result)


def zoh_base_x(time_s, duration_s, speed_m_s):
    """Ideal straight base follows the source 50 Hz held ramp; not dynamics."""
    if time_s < 0:
        raise ValueError("Time must be nonnegative")
    count = int(math.floor(time_s / CONTROL_DT + 1e-10))
    distance = sum(quintic_fraction(k * CONTROL_DT, duration_s) for k in range(count)) * CONTROL_DT
    distance += max(0., time_s - count * CONTROL_DT) * quintic_fraction(count * CONTROL_DT, duration_s)
    return speed_m_s * distance


class SeamPathModel:
    def __init__(self):
        self.profile = source_profile()
        self.fk = load_module("vd12_original_urdf_fk", FK_SOURCE)
        self.fk.URDF = URDF  # in-memory selection only; original source is untouched
        self.chain, _, self.bounds = self.fk.load_arm_chain()
        self.tcp = np.eye(4)
        self.tcp[:3, 3] = self.profile["tcp_in_link06_m"]
        self.tcp[:3, :3] = Rotation.from_quat(np.asarray(self.profile["tcp_quat_wxyz"])[[1, 2, 3, 0]]).as_matrix()
        self.geometry = load_module("vd12_independent_urdf_geometry", AUDIT_SOURCE)
        self.audit = self.geometry.GunRobotGeometryAudit(URDF)
        # Reuse only this old check's analytic branch as an IK seed.  Its old
        # curriculum target generator and clearance implementation are not used.
        self.seed_model = load_module("vd12_analytic_seed_only", OLD_CHECK_SOURCE).Model()
        defaults = source_call_keyword(SOURCE / "assets/unitree.py", "ArticulationCfg.InitialStateCfg", "joint_pos")
        self.nominal_joints = {}
        for name in self.audit.joint_names:
            if name in ARM_NAMES:
                continue
            matches = [value for pattern, value in defaults.items() if re.fullmatch(pattern, name)]
            if len(matches) != 1:
                raise ValueError("Ambiguous/missing nominal joint default: " + name)
            self.nominal_joints[name] = matches[0]
        self.start_q = np.asarray(self.profile["reset_arm_q_rad"])
        self.start_pose = self.forward(self.start_q)
        self.start_quat = Rotation.from_matrix(self.start_pose[:3, :3]).as_quat()
        self.end_quat = Rotation.from_euler("xyz", (0., self.profile["seam_pitch_rad"], math.pi)).as_quat()
        self.radius = self.audit.metadata["tube"]["radius_m"]
        self.length = self.audit.metadata["tube"]["length_m"]

    def forward(self, q):
        return self.fk.forward_kinematics(q, self.chain, self.tcp)

    def floor_support(self, axis_z):
        return self.length * max(float(axis_z), 0.) + self.radius * math.sqrt(max(0., 1. - axis_z**2))

    def target(self, time_s, speed_m_s=0.):
        """Existing target with ground_blend=1, including 50 Hz command hold."""
        duration = self.profile["duration_s"]
        command_t = min(duration, math.floor(time_s / CONTROL_DT + 1e-10) * CONTROL_DT)
        fraction = quintic_fraction(command_t, duration)
        quaternion = nlerp(self.start_quat, self.end_quat, fraction)
        rotation = Rotation.from_quat(quaternion).as_matrix()
        current_base = np.array((zoh_base_x(time_s, duration, speed_m_s), 0., self.profile["base_height_m"]))
        command_base = np.array((zoh_base_x(command_t, duration, speed_m_s), 0., self.profile["base_height_m"]))
        start_world = self.start_pose[:3, 3] + np.array((0., 0., self.profile["base_height_m"]))
        seam_world = np.array((command_base[0] - self.profile["trailing_distance_m"], 0., 0.))
        position_world = start_world + fraction * (seam_world - start_world)
        # Source applies this AFTER the parent's position blend: using CURRENT
        # interpolated orientation, not a linear blend to the compensated end.
        rim_height = self.floor_support(rotation[2, 0]) + self.profile["working_clearance_m"]
        position_world[2] += fraction * rim_height
        return dict(time_s=float(time_s), command_time_s=command_t, startup_fraction=fraction,
                    position_root_m=position_world - current_base, position_world_m=position_world,
                    quaternion_xyzw=quaternion, rotation=rotation, base_position_world_m=current_base,
                    current_attitude_end_rim_height_m=rim_height)

    def evidence(self, q, target):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected six finite arm joint positions")
        actual = self.forward(q)
        margin = np.minimum(q - self.bounds[:, 0], self.bounds[:, 1] - q)
        position_error = float(np.linalg.norm(actual[:3, 3] - target["position_root_m"]))
        rotation_error = float(Rotation.from_matrix(actual[:3, :3].T @ target["rotation"]).magnitude())
        positions = dict(self.nominal_joints)
        positions.update(zip(ARM_NAMES, q))
        ordered = [positions[name] for name in self.audit.joint_names]
        geometry = self.audit.audit(self.audit.joint_names, ordered)
        independent_r, independent_p = self.audit._fk(positions)["link06"]
        independent_tcp = independent_p + independent_r @ self.tcp[:3, 3]
        independent_rotation = independent_r @ self.tcp[:3, :3]
        independent_position_error = float(np.linalg.norm(independent_tcp - actual[:3, 3]))
        independent_rotation_error = float(np.linalg.norm(independent_rotation - actual[:3, :3]))
        floor_clearance = float(actual[2, 3] + target["base_position_world_m"][2] - self.floor_support(actual[2, 0]))
        pose_feasible = (position_error <= POSITION_TOL and rotation_error <= ROTATION_TOL
                         and float(margin.min()) >= -1e-9)
        accepted = (pose_feasible and geometry["status"] == "safe"
                    and floor_clearance >= self.profile["working_clearance_m"] - 1e-8
                    and independent_position_error < 1e-9 and independent_rotation_error < 1e-9)
        return dict(accepted=bool(accepted), full_pose_within_hard_limits=bool(pose_feasible),
                    joint_angles_rad=q.tolist(), joint_limit_margin_rad=margin.tolist(),
                    position_error_m=position_error, full_orientation_error_rad=rotation_error,
                    finite_tube_floor_clearance_m=floor_clearance,
                    independent_fk_position_difference_m=independent_position_error,
                    independent_fk_rotation_matrix_difference=independent_rotation_error,
                    geometry=geometry)

    def solve(self, target, previous_q, *, seed_count=8, max_nfev=100):
        if seed_count < 1 or max_nfev < 1:
            raise ValueError("Require positive IK seed and evaluation counts")
        lower, upper = self.bounds[:, 0], self.bounds[:, 1]
        starts = [np.asarray(previous_q)]
        # This closed-form branch happens to match full attitude in the two
        # nominal cases.  Numerical IK and full-rotation residual independently
        # reject it if that ceases to be true; axis-only success is insufficient.
        rotation = target["rotation"]
        pitch = math.atan2(-rotation[2, 0], -rotation[0, 0])
        try:
            starts.append(self.seed_model.solve(target["position_root_m"], pitch)[0])
        except ValueError:
            pass
        starts.extend((self.start_q, (lower + upper) / 2.))
        rng = np.random.default_rng(120741)
        starts.extend(rng.uniform(lower, upper) for _ in range(max(0, seed_count - len(starts))))
        attempts = []

        def residual(q):
            actual = self.forward(q)
            return np.r_[actual[:3, 3] - target["position_root_m"],
                         .2 * Rotation.from_matrix(actual[:3, :3].T @ rotation).as_rotvec()]

        for start in starts[:seed_count]:
            result = least_squares(residual, np.clip(start, lower + 1e-12, upper - 1e-12),
                                   bounds=(lower, upper), max_nfev=max_nfev,
                                   ftol=1e-11, xtol=1e-11, gtol=1e-11)
            candidate = self.evidence(result.x, target)
            candidate["optimizer"] = dict(success=bool(result.success), evaluations=result.nfev,
                                          seed_index=len(attempts))
            attempts.append(candidate)
            if candidate["accepted"]:
                return candidate, attempts
        return None, attempts


def validate_path(model, *, points=101, speed_m_s=0., seed_count=8):
    if points < 2 or seed_count < 1:
        raise ValueError("Require at least two path points and a positive IK seed count")
    records, previous_q = [], model.start_q.copy()
    for index, time_s in enumerate(np.linspace(0., model.profile["duration_s"], points)):
        target = model.target(float(time_s), speed_m_s)
        selected, attempts = model.solve(target, previous_q, seed_count=seed_count)
        record = {key: value.tolist() if isinstance(value, np.ndarray) else value
                  for key, value in target.items() if key != "rotation"}
        record.update(sample_index=index, ik_attempt_count=len(attempts), selected_solution=selected,
                      failed_candidates=[] if selected else attempts)
        records.append(record)
        if selected is None:
            return dict(sampled_path_passed=False, requested_sample_count=points,
                        attempted_sample_count=len(records), stop_reason="No certified safe full-pose IK witness found; search failure is not an unreachability proof",
                        unreachability_proved=False, points=records)
        previous_q = np.asarray(selected["joint_angles_rad"])
    solutions = [row["selected_solution"] for row in records]
    joint_path = np.array([row["joint_angles_rad"] for row in solutions])
    steps = np.abs(np.diff(joint_path, axis=0))
    return dict(sampled_path_passed=True, requested_sample_count=points, attempted_sample_count=len(records),
                unreachability_proved=False, continuous_motion_certified=False,
                min_hard_limit_margin_rad=min(min(row["joint_limit_margin_rad"]) for row in solutions),
                min_robot_clearance_lower_bound_m=min(row["geometry"]["minimum_clearance_lower_bound_m"] for row in solutions),
                min_finite_tube_floor_clearance_m=min(row["finite_tube_floor_clearance_m"] for row in solutions),
                max_position_error_m=max(row["position_error_m"] for row in solutions),
                max_full_orientation_error_rad=max(row["full_orientation_error_rad"] for row in solutions),
                max_independent_fk_position_difference_m=max(row["independent_fk_position_difference_m"] for row in solutions),
                geometry_status_counts=dict(Counter(row["geometry"]["status"] for row in solutions)),
                adjacent_joint_continuity=dict(max_absolute_step_rad=float(steps.max()),
                    max_step_per_joint_rad=steps.max(axis=0).tolist(),
                    max_finite_difference_speed_rad_s=(steps.max(axis=0) / (model.profile["duration_s"] / (points - 1))).tolist(),
                    note="Finite differences are kinematic diagnostics, not a dynamics, velocity-envelope, or swept-collision certificate"),
                points=records)


def reference_trace_check(model, trace_path):
    """Check recorded target heights at full startup, without reinterpreting q."""
    rows = json.loads(trace_path.read_text(encoding="utf-8"))
    reset_path = trace_path.parent / "vd10_reset_diagnostics.json"
    reset = json.loads(reset_path.read_text(encoding="utf-8"))
    if len(reset["sample_arm_joint_rad"]) != 6:
        raise ValueError("Recorded reset arm schema differs")
    reset_difference = float(np.max(np.abs(np.asarray(reset["sample_arm_joint_rad"]) - model.start_q)))
    errors = []
    for index, row in enumerate(rows):
        if index == 0 or row["physics_step"] % 4 != 1 or row["sample_episode_actor_step"] * CONTROL_DT < model.profile["duration_s"]:
            continue
        if row["ground_blend"] != 1.:
            raise ValueError("Reference trace is not at full ground curriculum")
        # Command update precedes this actor interval. Use the previous
        # interval's last recorded base quaternion, not the current substep.
        base = np.asarray(rows[index - 1]["sample_root_quat_wxyz"])
        rotation = Rotation.from_quat(base[[1, 2, 3, 0]]).as_matrix()
        target_axis = rotation @ Rotation.from_quat(model.end_quat).as_matrix()[:, 0]
        expected = model.floor_support(target_axis[2]) + model.profile["working_clearance_m"]
        errors.append(abs(expected - row["sample_tcp_target_world_m"][2]))
    selected = []
    for step in (1, 248, 301, 404, 5996):
        if step <= len(rows):
            row = rows[step - 1]
            selected.append(dict(physics_step=step, episode_actor_step=row["sample_episode_actor_step"],
                                 target_world_z_m=row["sample_tcp_target_world_m"][2],
                                 actual_world_z_m=row["sample_tcp_z_m"], actual_arm_q_rad=row["sample_arm_joint_rad"],
                                 actual_joint_target_rad=row["sample_arm_joint_target_rad"]))
    return dict(source_trace=str(trace_path.resolve()), trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                sample_env_id=0, actual_arm_joints_recorded=6, source_physics_samples=len(rows),
                reset_q_matches_source=reset_difference < 1e-6, max_reset_q_difference_rad=reset_difference,
                settled_target_height_checked_count=len(errors),
                max_settled_target_height_error_m=max(errors) if errors else None,
                settled_height_matches_source=bool(errors and max(errors) < 1e-6), selected_actual_frames=selected,
                limitation="Recorded AMP base height/attitude differ from the nominal paths. Initial world TCP/target quaternion is not fully logged, so no exact reconstruction claim is made for the recorded transient orientation or all 32 environments.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=101)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--reference-trace", type=Path)
    args = parser.parse_args()
    if args.points < 101 or args.seeds < 1:
        parser.error("Require at least 101 nominal path points and one IK seed")
    model = SeamPathModel()
    inputs = [URDF, FK_SOURCE, OLD_CHECK_SOURCE, AUDIT_SOURCE, Path(__file__),
              SOURCE / "assets/unitree.py", TASK / "b2_z1_env_cfg.py", TASK / "b2_z1_vd10_env_cfg.py",
              TASK / "b2_z1_vd5_startup_env_cfg.py", TASK / "mdp/tcp_frame.py",
              TASK / "mdp/commands/vd5_startup_command.py", TASK / "mdp/commands/vd5_world_seam_command.py",
              TASK / "mdp/commands/vd10_safe_seam_command.py"]
    report = dict(schema_version=1, evaluation_only=True, physics_advanced=False, actor_or_training_changed=False,
                  python=sys.executable, numpy_version=np.__version__, scipy_version=scipy.__version__,
                  inputs=[dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in inputs],
                  profile=model.profile, actual_initial_tcp_root_m=model.start_pose[:3, 3].tolist(),
                  hard_joint_bounds_rad=model.bounds.tolist(), arm_joint_names=ARM_NAMES,
                  nominal_leg_q_rad=model.nominal_joints, independent_geometry_metadata=model.audit.metadata,
                  method="Full ground curriculum=1. Quintic episode startup + shortest quaternion nlerp + current-attitude rim correction, with 50 Hz command hold. Full 6D bounded scipy IK, analytic seed/continuation/multistart fallback; independent all-collider URDF full-gun audit and exact cylinder ground support.",
                  paths={}, limitations=["Only straight nominal base motion with fixed nominal leg joints; no AMP gait, curved seam, perturbation, compliance or dynamic verification.",
                      "Every accepted point has a full-pose joint witness and whole-tube sampled separation certificate; neither sparse samples nor small adjacent q changes prove continuous collision-free motion.",
                      "Unmodelled visual/physical surfaces are outside authored URDF collision scope.",
                      "These witnesses are NOT actor outputs and do not establish policy competence or training convergence.",
                      "No safe witness found is an inconclusive search result, never a proof that all possible branches are unreachable."])
    if args.reference_trace:
        report["existing_actual_trace_crosscheck"] = reference_trace_check(model, args.reference_trace)
    for label, speed in (("standing_base", 0.), ("ideal_commanded_straight_base", model.profile["speed_m_s"])):
        report["paths"][label] = validate_path(model, points=args.points, speed_m_s=speed, seed_count=args.seeds)
        item = report["paths"][label]
        print(f"{label}: sampled_path_passed={item['sampled_path_passed']}, points={item['attempted_sample_count']}", flush=True)
        if not item["sampled_path_passed"]:
            break  # do not silently adjust the task or continue to a deployment
    report["all_requested_sampled_paths_passed"] = len(report["paths"]) == 2 and all(row["sampled_path_passed"] for row in report["paths"].values())
    if args.reference_trace:
        crosscheck = report["existing_actual_trace_crosscheck"]
        report["all_requested_sampled_paths_passed"] &= crosscheck["reset_q_matches_source"] and crosscheck["settled_height_matches_source"]
    output_root = ROOT / "outputs"
    output_root.mkdir(exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="vd12_seam_path_", dir=output_root))
    output = output_dir / "full_ground_startup_path.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("report=" + str(output))
    return 0 if report["all_requested_sampled_paths_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
