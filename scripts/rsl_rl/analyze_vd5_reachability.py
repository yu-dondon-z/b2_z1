#!/usr/bin/env python3
"""Check VD5's rear Z1 TCP targets against URDF joint limits without Isaac Sim."""

from __future__ import annotations

import argparse
import ast
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "source/b2_z1/b2_z1/assets/b2_z1/b2_z1_vd3_zero_gun.urdf"
TCP_SOURCE = ROOT / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/tcp_frame.py"


def _rpy_matrix(rpy):
    return Rotation.from_euler("xyz", rpy).as_matrix()


def _transform(xyz, rpy):
    result = np.eye(4)
    result[:3, :3] = _rpy_matrix(rpy)
    result[:3, 3] = xyz
    return result


def _axis_rotation(axis, angle):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(np.asarray(axis, dtype=float) * angle).as_matrix()
    return result


def _source_constant(name):
    module = ast.parse(TCP_SOURCE.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return np.asarray(ast.literal_eval(node.value), dtype=float)
    raise ValueError(f"Could not find {name} in {TCP_SOURCE}")


def load_arm_chain():
    root = ET.parse(URDF).getroot()
    joints = {}
    for joint in root.findall("joint"):
        child = joint.find("child").get("link")
        joints[child] = joint

    chain = []
    child = "link06"
    while child != "base_link":
        joint = joints[child]
        parent = joint.find("parent").get("link")
        origin = joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        axis_node = joint.find("axis")
        axis = np.fromstring(axis_node.get("xyz", "0 0 1"), sep=" ") if axis_node is not None else None
        limit = joint.find("limit")
        bounds = (float(limit.get("lower")), float(limit.get("upper"))) if limit is not None else None
        chain.append((joint.get("type"), xyz, rpy, axis, bounds, joint.get("name")))
        child = parent
    chain.reverse()
    if [item[5] for item in chain if item[0] == "revolute"] != [f"joint{i}" for i in range(1, 7)]:
        raise ValueError("Unexpected Z1 arm chain in URDF")
    tcp_pos = _source_constant("TCP_POS_IN_LINK06")
    tcp_quat_wxyz = _source_constant("TCP_QUAT_IN_LINK06")
    tcp_rot = Rotation.from_quat(tcp_quat_wxyz[[1, 2, 3, 0]]).as_matrix()
    tcp = np.eye(4)
    tcp[:3, :3] = tcp_rot
    tcp[:3, 3] = tcp_pos
    bounds = np.asarray([item[4] for item in chain if item[0] == "revolute"], dtype=float)
    return chain, tcp, bounds


def forward_kinematics(q, chain, tcp):
    transform = np.eye(4)
    index = 0
    for joint_type, xyz, rpy, axis, _, _ in chain:
        transform = transform @ _transform(xyz, rpy)
        if joint_type == "revolute":
            transform = transform @ _axis_rotation(axis, q[index])
            index += 1
    return transform @ tcp


def target_rotation():
    # Match Isaac Lab's quat_from_euler_xyz composition exactly.
    return Rotation.from_euler("xyz", (0.0, math.pi / 2.0 - 0.45, math.pi)).as_matrix()


def solve_target(target, desired_rotation, chain, tcp, bounds, *, position_only, seed_count):
    lower, upper = bounds[:, 0], bounds[:, 1]
    center = 0.5 * (lower + upper)
    rng = np.random.default_rng(741)
    starts = [center, np.clip(np.array([0.0, -0.5, 1.0, -1.0, 0.0, 0.0]), lower, upper)]
    starts += [rng.uniform(lower, upper) for _ in range(max(0, seed_count - len(starts)))]
    best = None

    def residual(q):
        actual = forward_kinematics(q, chain, tcp)
        position_error = actual[:3, 3] - target
        if position_only:
            return position_error
        rotation_error = Rotation.from_matrix(actual[:3, :3].T @ desired_rotation).as_rotvec()
        return np.concatenate((position_error, 0.04 * rotation_error))

    for start in starts:
        result = least_squares(residual, start, bounds=(lower, upper), max_nfev=300,
                               ftol=1e-10, xtol=1e-10, gtol=1e-10)
        actual = forward_kinematics(result.x, chain, tcp)
        pos_error = float(np.linalg.norm(actual[:3, 3] - target))
        rot_error = float(Rotation.from_matrix(actual[:3, :3].T @ desired_rotation).magnitude())
        score = pos_error if position_only else pos_error + 0.04 * rot_error
        if best is None or score < best[0]:
            best = (score, result.x, pos_error, rot_error)
    _, q, pos_error, rot_error = best
    actual_rotation = forward_kinematics(q, chain, tcp)[:3, :3]
    # The gun axis is +X in the TCP frame; roll around that axis is not an
    # aiming error for a rotationally symmetric nozzle.
    gun_axis_error = math.acos(float(np.clip(
        actual_rotation[:, 0] @ desired_rotation[:, 0], -1.0, 1.0
    )))
    margin = np.minimum(q - lower, upper - q)
    return {
        "position_error_m": pos_error,
        "orientation_error_deg": math.degrees(rot_error),
        "gun_axis_error_deg": math.degrees(gun_axis_error),
        "joint_angles_rad": q.tolist(),
        "joint_limit_margin_rad": margin.tolist(),
    }


def solve_axis_target(target, desired_rotation, chain, tcp, bounds, *, tolerance_deg, seed_count):
    """Maximize joint-limit margin with exact position and a gun-axis cone constraint.

    This is a local, multistart kinematic feasibility check, not a collision or
    dynamic stability test. An unsuccessful search is not an unreachability proof.
    """
    lower, upper = bounds[:, 0], bounds[:, 1]
    desired_axis = desired_rotation[:, 0]
    threshold = math.cos(math.radians(tolerance_deg))
    rng = np.random.default_rng(741)
    starts = [.5 * (lower + upper)]
    starts += [rng.uniform(lower, upper) for _ in range(seed_count - 1)]

    def position_error(q):
        return forward_kinematics(q, chain, tcp)[:3, 3] - target

    def cone(z):
        return forward_kinematics(z[:6], chain, tcp)[:3, 0] @ desired_axis - threshold

    constraints = [
        {"type": "eq", "fun": lambda z: position_error(z[:6])},
        {"type": "ineq", "fun": cone},
        {"type": "ineq", "fun": lambda z: z[:6] - lower - z[6]},
        {"type": "ineq", "fun": lambda z: upper - z[:6] - z[6]},
    ]
    candidates = []
    for start in starts:
        position = least_squares(position_error, start, bounds=(lower, upper), max_nfev=200)
        result = minimize(lambda z: -z[6], np.r_[position.x, 0.], method="SLSQP",
            bounds=list(zip(lower, upper)) + [(0., float(np.min(upper-lower)/2))],
            constraints=constraints, options={"maxiter": 400, "ftol": 1.e-10})
        q = result.x[:6]
        actual = forward_kinematics(q, chain, tcp)
        error = float(np.linalg.norm(actual[:3, 3] - target))
        axis_error = math.degrees(math.acos(float(np.clip(actual[:3, 0] @ desired_axis, -1., 1.))))
        margins = np.minimum(q-lower, upper-q)
        feasible = error <= .001 and axis_error <= tolerance_deg + 1.e-4 and margins.min() >= -1.e-6
        candidates.append({"feasible": bool(feasible), "position_error_m": error,
            "gun_axis_error_deg": axis_error, "joint_angles_rad": q.tolist(),
            "joint_limit_margin_rad": margins.tolist(), "optimizer_success": bool(result.success)})
    return min(candidates, key=lambda row: (not row["feasible"],
        -min(row["joint_limit_margin_rad"]) if row["feasible"] else row["position_error_m"]
        + .04 * math.radians(row["gun_axis_error_deg"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=-1.0)
    parser.add_argument("--y-offsets", default="-0.10,0,0.10")
    parser.add_argument("--heights", default="-0.20,-0.30,-0.40,-0.50,-0.60,-0.70,-0.80")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--mode", choices=("pose", "axis"), default="pose")
    parser.add_argument("--axis-tolerance-deg", type=float, default=15.)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seeds < 1 or not 0. < args.axis_tolerance_deg < 180.:
        raise ValueError("Require positive seeds and axis tolerance in (0, 180) degrees.")
    chain, tcp, bounds = load_arm_chain()
    desired_rotation = target_rotation()
    report = {"urdf": str(URDF), "tcp_frame_source": str(TCP_SOURCE), "mode": args.mode,
              "axis_tolerance_deg": args.axis_tolerance_deg, "seeds": args.seeds,
              "collision_checked": False, "dynamics_checked": False, "targets": []}
    for y in map(float, args.y_offsets.split(",")):
        for z in map(float, args.heights.split(",")):
            target = np.array([args.x, y, z])
            if args.mode == "axis":
                solution = solve_axis_target(target, desired_rotation, chain, tcp, bounds,
                    tolerance_deg=args.axis_tolerance_deg, seed_count=args.seeds)
                report["targets"].append({"target_base_m": target.tolist(), "axis_constrained": solution})
                continue
            report["targets"].append({
                "target_base_m": target.tolist(),
                "position_only": solve_target(target, desired_rotation, chain, tcp, bounds,
                                               position_only=True, seed_count=args.seeds),
                "fixed_pose": solve_target(target, desired_rotation, chain, tcp, bounds,
                                           position_only=False, seed_count=args.seeds),
            })
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["targets"]:
        if args.mode == "axis":
            result = row["axis_constrained"]
            print(f"target={row['target_base_m']} feasible={result['feasible']} "
                  f"position={result['position_error_m']*1000:.3f} mm "
                  f"axis={result['gun_axis_error_deg']:.2f} deg "
                  f"margin={min(result['joint_limit_margin_rad']):.3f} rad")
            continue
        p = row["position_only"]
        f = row["fixed_pose"]
        print(f"y={row['target_base_m'][1]:+.2f} z={row['target_base_m'][2]:+.2f} "
              f"pos-only={p['position_error_m']*1000:6.1f} mm "
              f"(pose={p['orientation_error_deg']:5.1f} deg, gun-axis={p['gun_axis_error_deg']:5.1f} deg) "
              f"fixed-pose={f['position_error_m']*1000:6.1f} mm, "
              f"pose={f['orientation_error_deg']:5.1f} deg, gun-axis={f['gun_axis_error_deg']:5.1f} deg, "
              f"min-joint-margin={min(f['joint_limit_margin_rad']):.3f} rad")


if __name__ == "__main__":
    main()
