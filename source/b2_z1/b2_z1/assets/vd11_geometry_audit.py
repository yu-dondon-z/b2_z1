"""Evaluation-only, CPU FK audit of the measured gun against URDF colliders.

This module deliberately imports neither Isaac nor MuJoCo.  Pass encoder joint
positions, not simulator world poses.  It never changes a model, a collision
mask, an action or an observation.  The same source URDF can therefore be used
by Isaac Play and the isolated MuJoCo viewer.

Three-valued result semantics are important:

* ``safe``: disjointness of the authored collision primitives is proved by
  exact distances or conservative enclosing capsules;
* ``invalid``: a positive-length part of the gun's centreline lies strictly
  inside another solid collision primitive, proving an intersection;
* ``uncertain``: the enclosing capsules overlap but the sufficient intersection
  test does not prove a collision (or an unsupported collider is present).

Negative ``clearance_lower_bound_m`` is NEVER a penetration depth.  A ``safe``
result covers only authored URDF collision shapes, not unmodelled physical
geometry or collision-free motion between samples.  The gun is audited as its
solid *outer collision cylinder*, not as the hollow visual mesh.  No narrow-
phase engine API (including ``mj_geomDistance``) is used.

Public API::

    audit = GunRobotGeometryAudit(urdf_path)
    report = audit.audit(joint_names, actual_q)  # one (J,) CPU vector
    reports = audit.audit_batch(joint_names, actual_q_batch)  # (B, J)

Reports and ``audit.metadata`` are JSON-serializable.  A caller may stop an
evaluation on ``report['confirmed_invalid']``; an uncertain result must not be
presented as a confirmed collision.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _vector(text, default=(0.0, 0.0, 0.0)):
    value = np.asarray(default if text is None else text.split(), dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("URDF vector must contain three finite values")
    return value


def _rotation_rpy(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                     (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                     (-sp, cp * sr, cp * cr)))


def _origin(element):
    origin = element.find("origin")
    if origin is None:
        return np.eye(3), np.zeros(3)
    return _rotation_rpy(_vector(origin.get("rpy"))), _vector(origin.get("xyz"))


def _axis_rotation(axis, angle):
    x, y, z = axis
    cross = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def _clip_linear_interval(start, delta, lower, upper, interval=(0.0, 1.0)):
    """Parameter interval where a segment coordinate is in [lower, upper]."""
    lo, hi = interval
    if lower > upper or lo > hi:
        return None
    if abs(delta) < 1e-15:
        return (lo, hi) if lower <= start <= upper else None
    a, b = (lower - start) / delta, (upper - start) / delta
    lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    return (lo, hi) if lo <= hi else None


def _segment_box_interval(a, b, half_size, tolerance):
    interval = (0.0, 1.0)
    for k in range(3):
        interval = _clip_linear_interval(a[k], b[k] - a[k],
                                         -half_size[k] + tolerance,
                                         half_size[k] - tolerance, interval)
        if interval is None:
            return None
    return interval


def _quadratic_interval(a, b, c):
    """Parameter interval satisfying a*t*t+b*t+c <= 0, with a >= 0."""
    if a < 1e-28:
        if abs(b) < 1e-15:
            return (0.0, 1.0) if c <= 0.0 else None
        bound = -c / b
        lo, hi = (0.0, min(1.0, bound)) if b > 0.0 else (max(0.0, bound), 1.0)
    else:
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        # Both roots are needed to clip against a finite [0, 1] segment.
        root = math.sqrt(max(disc, 0.0))
        lo, hi = max(0.0, (-b - root) / (2.0 * a)), min(1.0, (-b + root) / (2.0 * a))
    return (lo, hi) if lo <= hi else None


def _segment_round_interval(a, b, radius, half_length, tolerance):
    """Centreline inside a sphere, or a finite Z-axis cylinder (local frame)."""
    r = radius - tolerance
    if r <= 0.0:
        return None
    delta = b - a
    components = slice(None) if half_length is None else slice(0, 2)
    x, v = a[components], delta[components]
    interval = _quadratic_interval(float(v @ v), float(2.0 * x @ v), float(x @ x - r * r))
    if interval is None or half_length is None:
        return interval
    return _clip_linear_interval(a[2], delta[2], -half_length + tolerance,
                                 half_length - tolerance, interval)


def _segment_segment_distance(a, b, c, d):
    """Exact Euclidean distance between finite segments, including parallels."""
    u, v, w = b - a, d - c, a - c
    aa, bb, cc, dd, ee = float(u @ u), float(u @ v), float(v @ v), float(u @ w), float(v @ w)
    if aa < 1e-28:
        t = float(np.clip(ee / cc, 0.0, 1.0)) if cc > 1e-28 else 0.0
        return float(np.linalg.norm(w - t * v))
    if cc < 1e-28:
        s = float(np.clip(-dd / aa, 0.0, 1.0))
        return float(np.linalg.norm(w + s * u))
    determinant = aa * cc - bb * bb
    s = float(np.clip((bb * ee - cc * dd) / determinant, 0.0, 1.0)) if determinant > 1e-14 * aa * cc else 0.0
    t = (bb * s + ee) / cc
    if t < 0.0:
        t, s = 0.0, float(np.clip(-dd / aa, 0.0, 1.0))
    elif t > 1.0:
        t, s = 1.0, float(np.clip((bb - dd) / aa, 0.0, 1.0))
    return float(np.linalg.norm(w + s * u - t * v))


def _segment_box_distance(a, b, half_size):
    """Exact distance from a finite segment to an axis-aligned solid box.

    Squared point-box distance is piecewise quadratic.  Splitting at the six
    box faces lets us minimize each piece analytically, without a collision
    engine, iterative optimizer or a potentially unsafe sampling approximation.
    """
    delta = b - a
    points = [0.0, 1.0]
    for k in range(3):
        if abs(delta[k]) > 1e-15:
            for face in (-half_size[k], half_size[k]):
                t = (face - a[k]) / delta[k]
                if 0.0 < t < 1.0:
                    points.append(float(t))
    points = sorted(set(points))

    def value(t):
        excess = np.maximum(np.abs(a + t * delta) - half_size, 0.0)
        return float(excess @ excess)

    best = min(value(t) for t in points)
    for lo, hi in zip(points[:-1], points[1:]):
        middle = a + (lo + hi) * 0.5 * delta
        active = np.abs(middle) > half_size
        if not np.any(active):
            return 0.0
        offset = a[active] - np.sign(middle[active]) * half_size[active]
        direction = delta[active]
        denominator = float(direction @ direction)
        if denominator > 1e-28:
            t = float(np.clip(-float(offset @ direction) / denominator, lo, hi))
            best = min(best, value(t))
    return math.sqrt(max(best, 0.0))


class GunRobotGeometryAudit:
    """Parse a source URDF once; audit samples in its root/body frame.

    All other authored robot collision shapes are included by default, not just
    the base/arm.  ``other_link_names`` can explicitly restrict that scope.
    Links without a collision shape are listed in metadata, never silently
    interpreted as a certificate about their visible/physical surfaces.
    """

    def __init__(self, urdf_path, tube_collision_name="vd10_hollow_gun_collision",
                 other_link_names=None, tolerance_m=1e-6):
        self.urdf_path = Path(urdf_path).resolve(strict=True)
        self.tolerance_m = float(tolerance_m)
        if not math.isfinite(self.tolerance_m) or self.tolerance_m <= 0.0:
            raise ValueError("tolerance_m must be finite and positive")
        source = self.urdf_path.read_bytes()
        root = ET.fromstring(source)
        links = {element.get("name"): element for element in root.findall("link")}
        if not links or None in links:
            raise ValueError("URDF must contain named links")
        self._children = {name: [] for name in links}
        child_names = set()
        self._joint_names = []
        for element in root.findall("joint"):
            name, kind = element.get("name"), element.get("type")
            parent, child = element.find("parent").get("link"), element.find("child").get("link")
            if parent not in links or child not in links or child in child_names:
                raise ValueError("URDF joints must form a tree of known links")
            if kind not in ("fixed", "revolute", "continuous", "prismatic"):
                raise ValueError("Unsupported URDF joint type: " + str(kind))
            if element.find("mimic") is not None:
                raise ValueError("Mimic joints need an explicit independent-FK implementation")
            rotation, position = _origin(element)
            axis_element = element.find("axis")
            # Some source fixed joints explicitly carry axis="0 0 0"; it has
            # no kinematic meaning and must not invalidate an otherwise valid
            # URDF.  Only movable joints need a nonzero normalized axis.
            axis = (_vector(axis_element.get("xyz") if axis_element is not None else "1 0 0")
                    if kind != "fixed" else np.array((1.0, 0.0, 0.0)))
            if np.linalg.norm(axis) < 1e-12:
                raise ValueError("Joint axis must be nonzero")
            self._children[parent].append(dict(name=name, kind=kind, child=child,
                                               rotation=rotation, position=position,
                                               axis=axis / np.linalg.norm(axis)))
            child_names.add(child)
            if kind != "fixed":
                self._joint_names.append(name)
        roots = set(links) - child_names
        if len(roots) != 1 or len(set(self._joint_names)) != len(self._joint_names):
            raise ValueError("Expected one rooted URDF and unique movable joint names")
        self.root_link = next(iter(roots))
        self._tube = None
        self._colliders = []
        selected = None if other_link_names is None else set(other_link_names)
        if selected is not None and not selected <= set(links):
            raise ValueError("Unknown other_link_names: " + str(sorted(selected - set(links))))
        for name, element in links.items():
            for number, collision in enumerate(element.findall("collision")):
                rotation, position = _origin(collision)
                geometry = collision.find("geometry")
                if geometry is None or len(geometry) != 1:
                    raise ValueError("Collision must have exactly one geometry")
                shape = geometry[0]
                item = dict(link=name, name=collision.get("name", name + "_collision_" + str(number)),
                            primitive=shape.tag, rotation=rotation, position=position)
                if shape.tag == "box":
                    item["half_size"] = _vector(shape.get("size")) * 0.5
                    if np.any(item["half_size"] <= 0.0):
                        raise ValueError("Collision box dimensions must be positive")
                elif shape.tag in ("cylinder", "sphere"):
                    item["radius"] = float(shape.get("radius"))
                    if not math.isfinite(item["radius"]) or item["radius"] <= 0.0:
                        raise ValueError("Collision radius must be finite and positive")
                    if shape.tag == "cylinder":
                        item["half_length"] = float(shape.get("length")) * 0.5
                        if not math.isfinite(item["half_length"]) or item["half_length"] <= 0.0:
                            raise ValueError("Cylinder length must be finite and positive")
                if item["name"] == tube_collision_name:
                    if self._tube is not None or shape.tag != "cylinder":
                        raise ValueError("Expected exactly one named cylindrical gun collider")
                    self._tube = item
                if selected is None or name in selected:
                    self._colliders.append(item)
        if self._tube is None:
            raise ValueError("Missing named gun cylinder: " + tube_collision_name)
        self._colliders = [shape for shape in self._colliders if shape["link"] != self._tube["link"]]
        if not self._colliders:
            raise ValueError("No other robot colliders in audit scope")
        self.joint_names = tuple(self._joint_names)
        tube = self._tube
        self.metadata = dict(
            version=1, source_urdf=str(self.urdf_path), source_urdf_sha256=hashlib.sha256(source).hexdigest(),
            root_link=self.root_link, joint_names=list(self.joint_names), joint_count=len(self.joint_names),
            tube=dict(parent_link=tube["link"], collision_name=tube["name"],
                      radius_m=tube["radius"], outer_diameter_m=2.0 * tube["radius"],
                      length_m=2.0 * tube["half_length"], centre_local_m=tube["position"].tolist(),
                      axis_local=tube["rotation"][:, 2].tolist(), solid_outer_collision_cylinder=True),
            other_collider_count=len(self._colliders),
            checked_links=sorted({shape["link"] for shape in self._colliders}),
            links_without_authored_collision=sorted(name for name, element in links.items()
                                                    if not element.findall("collision")),
            tolerance_m=self.tolerance_m, evaluation_only=True, actor_input=False,
            scope="Authored URDF collision primitives only; not unmodelled visual surfaces or continuous swept motion",
            lower_bound_note="Negative clearance lower bounds from capsule envelopes are NOT penetration depths",
        )

    def _positions(self, joint_names, joint_positions):
        names = tuple(joint_names)
        if len(names) != len(set(names)) or set(names) != set(self.joint_names):
            raise ValueError("Provide every movable URDF joint exactly once by its real name")
        values = np.asarray(joint_positions, dtype=float)
        if values.shape != (len(names),) or not np.isfinite(values).all():
            raise ValueError("Expected one finite joint-position vector matching joint_names")
        return dict(zip(names, values))

    def _fk(self, positions):
        transforms = {self.root_link: (np.eye(3), np.zeros(3))}
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            parent_r, parent_p = transforms[parent]
            for joint in self._children[parent]:
                rotation = parent_r @ joint["rotation"]
                position = parent_p + parent_r @ joint["position"]
                if joint["kind"] in ("revolute", "continuous"):
                    rotation = rotation @ _axis_rotation(joint["axis"], positions[joint["name"]])
                elif joint["kind"] == "prismatic":
                    position = position + rotation @ (joint["axis"] * positions[joint["name"]])
                child = joint["child"]
                if child in transforms:
                    raise ValueError("URDF kinematic graph contains a cycle")
                transforms[child] = (rotation, position)
                stack.append(child)
        if len(transforms) != len(self._children):
            raise ValueError("URDF contains disconnected or cyclic links")
        return transforms

    def audit(self, joint_names, joint_positions):
        """Return a single JSON-safe report from actual joints (no world truth)."""
        transforms = self._fk(self._positions(joint_names, joint_positions))
        tube = self._tube
        link_r, link_p = transforms[tube["link"]]
        tube_centre = link_p + link_r @ tube["position"]
        tube_axis = link_r @ tube["rotation"][:, 2]
        a = tube_centre - tube["half_length"] * tube_axis
        b = tube_centre + tube["half_length"] * tube_axis
        length = 2.0 * tube["half_length"]
        results = []
        for shape in self._colliders:
            link_r, link_p = transforms[shape["link"]]
            rotation = link_r @ shape["rotation"]
            centre = link_p + link_r @ shape["position"]
            local_a, local_b = rotation.T @ (a - centre), rotation.T @ (b - centre)
            kind = shape["primitive"]
            interval, lower_bound = None, None
            if kind == "box":
                lower_bound = _segment_box_distance(local_a, local_b, shape["half_size"]) - tube["radius"]
                interval = _segment_box_interval(local_a, local_b, shape["half_size"], self.tolerance_m)
            elif kind in ("cylinder", "sphere"):
                half = shape.get("half_length", 0.0)
                lower_bound = (_segment_segment_distance(local_a, local_b,
                                np.array((0.0, 0.0, -half)), np.array((0.0, 0.0, half)))
                               - tube["radius"] - shape["radius"])
                interval = _segment_round_interval(local_a, local_b, shape["radius"],
                                                   shape.get("half_length"), self.tolerance_m)
            # Keep evidence strictly inside the gun's finite axial interval.
            inside = 0.0
            if interval is not None:
                lo, hi = max(interval[0], self.tolerance_m / length), min(interval[1], 1.0 - self.tolerance_m / length)
                inside = max(hi - lo, 0.0) * length
            invalid = inside > self.tolerance_m
            status = "invalid" if invalid else "safe" if lower_bound is not None and lower_bound > self.tolerance_m else "uncertain"
            result = dict(other_link=shape["link"], other_collision=shape["name"], primitive=kind,
                          status=status, clearance_lower_bound_m=lower_bound)
            if invalid:
                result.update(certificate="positive-length gun centreline inside shrunken solid collider",
                              centerline_inside_length_m=inside,
                              centerline_inside_endpoints_root_m=[(a + lo * (b - a)).tolist(),
                                                                 (a + hi * (b - a)).tolist()])
            elif lower_bound is None:
                result["reason"] = "unsupported authored collider; cannot certify safety"
            elif status == "uncertain":
                result["reason"] = "conservative capsule envelopes not separated; intersection not proved"
            results.append(result)
        intersections = [item for item in results if item["status"] == "invalid"]
        uncertain = [item for item in results if item["status"] == "uncertain"]
        bounds = [item["clearance_lower_bound_m"] for item in results if item["clearance_lower_bound_m"] is not None]
        return dict(status="invalid" if intersections else "uncertain" if uncertain else "safe",
                    confirmed_invalid=bool(intersections),
                    minimum_clearance_lower_bound_m=min(bounds) if bounds else None,
                    confirmed_intersections=intersections, uncertain_pairs=uncertain, pair_results=results,
                    tube_axis_root=tube_axis.tolist(), tube_centre_root_m=tube_centre.tolist(),
                    tube_endpoints_root_m=[a.tolist(), b.tolist()],
                    evaluation_only=True, actor_input=False, certificate_scope="authored URDF collision shapes")

    def audit_batch(self, joint_names, joint_positions):
        """Audit independent CPU samples; no interpolation or swept-motion claim."""
        names = tuple(joint_names)
        values = np.asarray(joint_positions, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(names) or not np.isfinite(values).all():
            raise ValueError("Expected finite (sample_count, joint_count) joint positions")
        # Validate even empty batches, without depending on a simulator instance.
        self._positions(names, np.zeros(len(names)))
        return [self.audit(names, row) for row in values]
