#!/usr/bin/env python3
"""Reproduce the VD10 startup/seam geometry check on CPU, without Isaac Sim.

Run from the workspace root with the env_isaaclab Python interpreter. This
checks the nominal straight seam and the recorded AMP reference poses; it is
not a dynamics, curved-path, controller, or continuous-collision certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True

import numpy as np
import scipy
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "source/b2_z1/b2_z1/assets/b2_z1/b2_z1_vd10_hollow_gun.urdf"
FK_SOURCE = ROOT / "scripts/rsl_rl/analyze_vd5_reachability.py"
OUTPUT = ROOT / "outputs/vd10_startup_20261113/geometry.json"
TCP = np.array((0.975, 0.0, 0.1))
START = np.array((-0.95, 0.0, 0.35))
START_PITCH = 0.30
SEAM_PITCH = math.pi / 2 - 0.45
RADIUS, LENGTH, SURFACE_CLEARANCE = 0.028, 1.1, 0.005
ARM_NAMES = tuple(f"joint{i}" for i in range(1, 7))
LEG_NAMES = ("FL", "FR", "RL", "RR")


def load_geometry():
    spec = importlib.util.spec_from_file_location("vd10_existing_fk", FK_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Change only this imported module's in-memory asset selection. Its source
    # remains unchanged; discard its VD3 TCP and supply the measured VD10 TCP.
    module.URDF = URDF
    chain, _, bounds = module.load_arm_chain()
    tcp = np.eye(4)
    tcp[:3, 3] = TCP
    return module, chain, tcp, bounds


class Model:
    def __init__(self):
        self.fk_module, self.chain, self.tcp, self.bounds = load_geometry()
        root = ET.parse(URDF).getroot()
        self.links = {link.get("name"): link for link in root.findall("link")}
        self.children = {}
        for joint in root.findall("joint"):
            self.children.setdefault(joint.find("parent").get("link"), []).append(joint)
        self.origins = {}
        for node in root.iter():
            if node.tag in ("joint", "collision"):
                origin = node.find("origin")
                xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                self.origins[node] = self.fk_module._transform(xyz, rpy)
        offsets = {item[5]: item[1] for item in self.chain}
        self.shoulder_height = sum(item[1][2] for item in self.chain[:3])
        self.upper_length = -offsets["joint3"][0]
        self.lower_length = np.hypot(offsets["joint4"][0], offsets["joint4"][2])
        self.lower_angle = np.arctan2(offsets["joint4"][2], -offsets["joint4"][0])
        self.wrist_length = offsets["joint5"][0] + offsets["joint6"][0] + TCP[0]

    def solve(self, target, pitch):
        """One exact axis-constrained branch: q1=q5=0, free tube-axis roll q6.

        The directed gun axis in base coordinates is (-cos(p), 0, -sin(p)).
        The other elbow/roll branches are not searched. FK verifies this branch.
        """
        target = np.atleast_2d(target)
        lateral = target[:, 1] / TCP[2]
        if np.any(np.abs(lateral) > 1 + 1e-12):
            raise ValueError("Target exceeds this IK branch's lateral reach")
        roll = np.arcsin(np.clip(lateral, -1, 1))
        offset = TCP[2] * np.cos(roll)
        x = target[:, 0] + self.wrist_length * np.cos(pitch) + offset * np.sin(pitch)
        z = target[:, 2] - self.shoulder_height + self.wrist_length * np.sin(pitch) - offset * np.cos(pitch)
        cosine = (x*x + z*z - self.upper_length**2 - self.lower_length**2) / (2*self.upper_length*self.lower_length)
        if np.any(np.abs(cosine) > 1 + 1e-12):
            raise ValueError("Target exceeds this IK branch's planar reach")
        beta = np.arccos(np.clip(cosine, -1, 1))
        q = np.zeros((len(target), 6))
        q[:, 1] = np.arctan2(z, x) - np.arctan2(self.lower_length*np.sin(beta), self.upper_length+self.lower_length*np.cos(beta))
        q[:, 2] = beta - self.lower_angle
        q[:, 3] = pitch - q[:, 1] - q[:, 2]
        q[:, 5] = roll
        return q

    def transforms(self, q, frames):
        count = len(q)
        angles = dict(zip(ARM_NAMES, q.T))
        for i, leg in enumerate(LEG_NAMES):
            for j, name in enumerate(("hip", "thigh", "calf")):
                angles[f"{leg}_{name}_joint"] = frames[:, 7 + 3*i + j]
        transforms = {}

        def visit(name, matrix):
            transforms[name] = matrix
            for joint in self.children.get(name, []):
                child = matrix @ self.origins[joint]
                if joint.get("type") == "revolute":
                    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                    rotation = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
                    rotation[:, :3, :3] = Rotation.from_rotvec(angles[joint.get("name")][:, None] * axis).as_matrix()
                    child = child @ rotation
                visit(joint.find("child").get("link"), child)

        visit("base_link", np.broadcast_to(np.eye(4), (count, 4, 4)))
        return transforms

    def clearances(self, transforms):
        tip = transform_points(transforms["link06"], TCP)
        rear = transform_points(transforms["link06"], TCP - np.array((LENGTH, 0, 0)))
        names = ("all_robot", "tail", "all_legs", "hind_legs", "arm")
        results = {name: np.full(len(tip), np.inf) for name in names}
        for name, link in self.links.items():
            if name == "link06":
                continue
            for collision in link.findall("collision"):
                matrix = transforms[name] @ self.origins[collision]
                shape = next(iter(collision.find("geometry")))
                if shape.tag == "box":
                    distance = segment_box_distance(rear, tip, matrix, np.fromstring(shape.get("size"), sep=" ")/2) - RADIUS
                elif shape.tag == "cylinder":
                    half_axis = matrix[:, :3, 2] * (float(shape.get("length"))/2)
                    distance = segment_distance(rear, tip, matrix[:, :3, 3]-half_axis, matrix[:, :3, 3]+half_axis) - RADIUS - float(shape.get("radius"))
                elif shape.tag == "sphere":
                    distance = segment_distance(rear, tip, matrix[:, :3, 3], matrix[:, :3, 3]) - RADIUS - float(shape.get("radius"))
                else:
                    raise ValueError(f"Unsupported collision geometry: {shape.tag}")
                groups = ["all_robot"]
                if name == "tail_link":
                    groups.append("tail")
                if name.startswith(tuple(leg + "_" for leg in LEG_NAMES)):
                    groups.append("all_legs")
                if name.startswith(("RL_", "RR_")):
                    groups.append("hind_legs")
                if name.startswith("link"):
                    groups.append("arm")
                for group in groups:
                    results[group] = np.minimum(results[group], distance)
        return results, tip, rear


def transform_points(matrix, point):
    return np.einsum("nij,j->ni", matrix[:, :3, :3], point) + matrix[:, :3, 3]


def segment_distance(a, b, c, d):
    """Exact segment distance, including degenerate point segments."""
    u, v, w = b-a, d-c, a-c
    aa, bb, cc, dd, ee = [np.sum(x*y, axis=1) for x, y in ((u,u), (u,v), (v,v), (u,w), (v,w))]
    determinant = aa*cc - bb*bb
    safe_cc = np.where(cc > 0, cc, 1)
    safe_det = np.where(determinant > 1e-14, determinant, 1)
    s, t = (bb*ee-cc*dd)/safe_det, (aa*ee-bb*dd)/safe_det
    interior = (determinant > 1e-14) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    zeros, ones = np.zeros(len(a)), np.ones(len(a))
    pairs = [(zeros, np.clip(ee/safe_cc, 0, 1)), (ones, np.clip((ee+bb)/safe_cc, 0, 1)),
             (np.clip(-dd/aa, 0, 1), zeros), (np.clip((bb-dd)/aa, 0, 1), ones), (s, t)]
    best = np.full(len(a), np.inf)
    for index, (s, t) in enumerate(pairs):
        distance = np.linalg.norm(w+s[:, None]*u-t[:, None]*v, axis=1)
        best = np.minimum(best, np.where(interior, distance, np.inf) if index == 4 else distance)
    return best


def segment_box_distance(a, b, matrix, half):
    """Exact segment-to-OBB distance via piecewise quadratic minimization."""
    a = np.einsum("nji,nj->ni", matrix[:, :3, :3], a-matrix[:, :3, 3])
    b = np.einsum("nji,nj->ni", matrix[:, :3, :3], b-matrix[:, :3, 3])
    delta = b-a
    safe = np.where(abs(delta) > 1e-15, delta, 1)
    crossings = np.concatenate(((-half-a)/safe, (half-a)/safe), axis=1)
    crossings = np.where(np.tile(abs(delta) > 1e-15, (1, 2)), crossings, 0)
    knots = np.sort(np.clip(np.c_[np.zeros(len(a)), np.ones(len(a)), crossings], 0, 1), axis=1)
    best = np.full(len(a), np.inf)
    for k in range(knots.shape[1]-1):
        low, high = knots[:, k], knots[:, k+1]
        point = a + ((low+high)/2)[:, None]*delta
        offset = np.where(point < -half, a+half, np.where(point > half, a-half, 0))
        slope = np.where(abs(point) > half, delta, 0)
        denominator = np.sum(slope*slope, axis=1)
        fraction = -np.sum(offset*slope, axis=1)/np.where(denominator > 0, denominator, 1)
        fraction = np.clip(fraction, low, high)
        distance = np.linalg.norm(np.maximum(abs(a+fraction[:, None]*delta)-half, 0), axis=1)
        best = np.minimum(best, distance)
    return best


def nominal_frames():
    frame = np.zeros((1, 19))
    frame[0, 2], frame[0, 6] = 0.5, 1.0  # root quaternion is xyzw
    for i, leg in enumerate(LEG_NAMES):
        frame[0, 7+3*i:10+3*i] = (.1 if leg.endswith("L") else -.1, .8 if leg.startswith("F") else 1., -1.5)
    return frame


def frame_rotations(frames):
    rotation = Rotation.from_quat(frames[:, 3:7])
    return rotation.as_matrix(), rotation.as_euler("xyz")[:, 2]


def centre_height(axis_z):
    return LENGTH*np.maximum(axis_z, 0) + RADIUS*np.sqrt(np.maximum(0, 1-axis_z**2)) + SURFACE_CLEARANCE


def evaluate_path(model, frames, path_points, trailing_distance, compensated):
    rotation, yaw = frame_rotations(frames)
    seam_world = np.c_[-trailing_distance*np.cos(yaw), -trailing_distance*np.sin(yaw), -frames[:, 2]]
    seam_base = np.einsum("nji,nj->ni", rotation, seam_world)
    result = {"frame_count": len(frames), "path_point_count": path_points,
              "min_clearance_lower_bound_m": {}, "witnesses": {},
              "max_position_error_m": 0., "max_directed_axis_error_rad": 0.,
              "max_existing_fk_difference": 0., "min_joint_margin_rad": [math.inf]*6,
              "min_soft_joint_margin_rad": [math.inf]*6}
    soft = model.bounds.mean(1)[:, None] + .9*(model.bounds-model.bounds.mean(1)[:, None])
    for blend in np.linspace(0, 1, path_points):
        pitch = START_PITCH + blend*(SEAM_PITCH-START_PITCH)
        desired_axis = np.array((-np.cos(pitch), 0., -np.sin(pitch)))
        target = (1-blend)*START + blend*seam_base
        if compensated:
            world_axis_z = rotation[:, 2, :] @ desired_axis
            # Settled episode-startup fraction = 1: this is the current ground
            # curriculum's base-frame interpolation plus world-Z rim correction.
            target += rotation[:, 2, :] * (blend*centre_height(world_axis_z))[:, None]
        q = model.solve(target, pitch)
        transforms = model.transforms(q, frames)
        distances, tip, rear = model.clearances(transforms)
        axis = transforms["link06"][:, :3, 0]
        axis_z = np.einsum("ni,ni->n", rotation[:, 2, :], axis)
        tip_z = np.einsum("ni,ni->n", rotation[:, 2, :], tip) + frames[:, 2]
        rear_z = np.einsum("ni,ni->n", rotation[:, 2, :], rear) + frames[:, 2]
        distances["floor_exact_cylinder"] = np.minimum(tip_z, rear_z)-RADIUS*np.sqrt(np.maximum(0, 1-axis_z**2))
        for name, values in distances.items():
            index = int(np.argmin(values))
            if values[index] < result["min_clearance_lower_bound_m"].get(name, math.inf):
                result["min_clearance_lower_bound_m"][name] = float(values[index])
                result["witnesses"][name] = {"blend": float(blend), "frame_index": index,
                    "joint_rad": q[index].tolist(), "target_base_m": target[index].tolist()}
        margins = np.minimum(q-model.bounds[:, 0], model.bounds[:, 1]-q)
        soft_margins = np.minimum(q-soft[:, 0], soft[:, 1]-q)
        result["min_joint_margin_rad"] = np.minimum(result["min_joint_margin_rad"], margins.min(0)).tolist()
        result["min_soft_joint_margin_rad"] = np.minimum(result["min_soft_joint_margin_rad"], soft_margins.min(0)).tolist()
        result["max_position_error_m"] = max(result["max_position_error_m"], float(np.linalg.norm(tip-target, axis=1).max()))
        angle = np.arctan2(np.linalg.norm(np.cross(axis, desired_axis), axis=1), axis @ desired_axis)
        result["max_directed_axis_error_rad"] = max(result["max_directed_axis_error_rad"], float(angle.max()))
        for index in {0, len(frames)-1, int(np.argmin(margins.min(1)))}:
            reference = model.fk_module.forward_kinematics(q[index], model.chain, model.tcp)
            difference = max(np.linalg.norm(reference[:3, 3]-tip[index]), np.linalg.norm(reference[:3, 0]-axis[index]))
            result["max_existing_fk_difference"] = max(result["max_existing_fk_difference"], float(difference))
        if blend == 1:
            result["endpoint_joint_rad"] = q[0].tolist() if len(frames) == 1 else None
            result["endpoint_center_world_z_range_m"] = [float(tip_z.min()), float(tip_z.max())]
    result["min_hard_limit_margin_rad"] = min(result["min_joint_margin_rad"])
    result["min_90_percent_soft_limit_margin_rad"] = min(result["min_soft_joint_margin_rad"])
    return result


def old_tail_witness(model):
    target = np.array((-.7, 0., -.5))
    q = model.solve(target, SEAM_PITCH)
    transforms = model.transforms(q, nominal_frames())
    _, tip, rear = model.clearances(transforms)
    axis = (tip[0]-rear[0])/LENGTH
    collision = model.links["tail_link"].find("collision")
    half = np.fromstring(collision.find("geometry/box").get("size"), sep=" ")/2
    matrix = transforms["tail_link"][0] @ model.origins[collision]
    witnesses = []
    # Include face/edge midpoints: this box is wider than the tube, so its
    # eight 3D vertices can be outside even when the rear upper edge overlaps.
    for signs in itertools.product((-1, 0, 1), repeat=3):
        if signs == (0, 0, 0):
            continue
        point = matrix[:3, :3] @ (half*np.asarray(signs)) + matrix[:3, 3]
        axial = float((point-rear[0]) @ axis)
        radial = float(np.linalg.norm(point-rear[0]-axial*axis))
        if 0 < axial < LENGTH and radial < RADIUS:
            witnesses.append({"tail_surface_point_base_m": point.tolist(), "axial_position_m": axial,
                              "radial_distance_m": radial, "radial_overlap_m": RADIUS-radial})
    return {"finite_cylinder_contains_tail_box_surface_point": bool(witnesses), "witness": witnesses[0] if witnesses else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amp-path-points", type=int, default=21,
                        help="Uniform curriculum blend samples per original AMP frame (default: 21)")
    args = parser.parse_args()
    if args.amp_path_points < 2:
        parser.error("--amp-path-points must be at least 2")
    model = Model()
    motion_files = sorted((ROOT / "datasets/b2_motion").glob("*.txt"))
    records, counts = [], []
    for path in motion_files:
        data = json.loads(path.read_text())
        if data["MotionWeight"] > 0:
            records.append(np.asarray(data["Frames"], dtype=float))
            counts.append({"path": str(path.relative_to(ROOT)), "frames": len(records[-1]),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    frames = np.concatenate(records)
    start_q = model.solve(START, START_PITCH)[0]
    report = {
        "scope": "CPU static sampled geometry; not a PhysX/dynamics or continuous-collision validation",
        "python": sys.executable, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "inputs": {"urdf": str(URDF), "urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(),
                   "existing_fk": str(FK_SOURCE), "existing_fk_sha256": hashlib.sha256(FK_SOURCE.read_bytes()).hexdigest(),
                   "amp_files": counts, "amp_frame_count": len(frames),
                   "amp_base_height_range_m": [float(frames[:, 2].min()), float(frames[:, 2].max())]},
        "profile": {"start_tcp_base_m": START.tolist(), "start_pitch_rad": START_PITCH,
                    "start_joint_rad": start_q.tolist(), "tcp_in_link06_m": TCP.tolist(),
                    "tube_length_m": LENGTH, "tube_outer_radius_m": RADIUS, "tube_mass_kg": 2.93,
                    "seam_pitch_rad": SEAM_PITCH, "old_trailing_distance_m": .7, "new_trailing_distance_m": .8,
                    "working_surface_clearance_m": SURFACE_CLEARANCE,
                    "horizontal_endpoint_center_z_m": float(centre_height(-np.sin(SEAM_PITCH)))},
        "method": {"ik": "One axis-constrained branch, q1=q5=0, q6 free; full TCP roll is not constrained",
                   "self_clearance": "Segment-to-OBB or segment-to-segment distance, with all cylinders enclosed in capsules; positive values certify envelope separation, negative values alone do not prove cylinder penetration",
                   "floor_clearance": "Exact finite-cylinder support, min(endpoint_z)-r*sqrt(1-axis_z**2)",
                   "path": "Linear blend of base-relative start and straight world seam, linear base-relative pitch, plus blend times world-Z end-face compensation; episode startup fraction held at 1",
                   "sampling": "101 horizontal path points; original AMP frames only, no inter-frame interpolation, no curved seams or command-position jitter",
                   "independent_fk": "Existing scalar URDF FK checked at first/last/minimum-joint-margin frame of every path sample"},
        "joint_names": list(ARM_NAMES), "hard_joint_bounds_rad": model.bounds.tolist(),
    }
    for name, dataset, points, distance, compensated in (
        ("old_horizontal_path", nominal_frames(), 101, .7, False),
        ("new_horizontal_path", nominal_frames(), 101, .8, True),
        ("new_amp_path", frames, args.amp_path_points, .8, True),
    ):
        print(f"Checking {name}: {len(dataset)} frame(s) x {points} path points", flush=True)
        report[name] = evaluate_path(model, dataset, points, distance, compensated)
    report["old_endpoint_true_tail_overlap"] = old_tail_witness(model)
    checks = []
    for name in ("new_horizontal_path", "new_amp_path"):
        item = report[name]
        checks.extend((item["min_hard_limit_margin_rad"] > 0,
                       min(item["min_clearance_lower_bound_m"].values()) >= SURFACE_CLEARANCE-1e-9,
                       item["max_position_error_m"] < 1e-9,
                       item["max_directed_axis_error_rad"] < 1e-9,
                       item["max_existing_fk_difference"] < 1e-9))
    checks.append(report["old_endpoint_true_tail_overlap"]["finite_cylinder_contains_tail_box_surface_point"])
    report["checks_passed"] = bool(all(checks))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"checks_passed={report['checks_passed']}; report={OUTPUT}")
    if not report["checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
