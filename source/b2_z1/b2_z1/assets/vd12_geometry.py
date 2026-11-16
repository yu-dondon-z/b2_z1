"""Batched Torch gun/robot keep-out geometry, with no simulator dependency.

``GunRobotClearance`` parses the measured gun URDF once and evaluates actual
rigid-body poses on CPU or GPU.  Other cylinders are enclosed by capsules;
the gun cylinder is also enclosed by its centreline capsule.  Box distance is
the exact finite-segment/OBB distance.  Subtracting radii gives a *clearance
lower bound*, NOT a measured contact force or penetration depth.

The deliberately conservative rule ``lower_bound < 0`` can define a keep-out
region.  It does not prove that the original finite cylinders intersect.  This
module supplies geometry only: no collision masks, observations, actor inputs,
reward weights or termination policy are changed here.

API::

    geometry = GunRobotClearance(urdf_path, actual_body_names, device="cuda")
    result = geometry(actual_body_pos_w, actual_body_quat_wxyz)
    # poses [env, body, 3/4], clearance_lower_bound_m [env, pair]
    # minimum_clearance_lower_bound_m [env], minimum_pair_index [env]

Authored fixed-joint colliders are remapped to their closest available rigid
ancestor in ``actual_body_names``.  Thus fixed-joint merging must not silently
drop tail/foot/rotor collision shapes.  All shapes carried by the gun's rigid
body are explicitly excluded.  Only authored collision shapes are covered;
metadata lists links lacking collision geometry and the exact pair scope.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch import nn


def _vector(text, default=(0.0, 0.0, 0.0)):
    result = tuple(float(x) for x in text.split()) if text is not None else tuple(default)
    if len(result) != 3 or not all(math.isfinite(x) for x in result):
        raise ValueError("URDF vectors must contain three finite numbers")
    return result


def _quat_product(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return (w * v - x * i - y * j - z * k, w * i + x * v + y * k - z * j,
            w * j - x * k + y * v + z * i, w * k + x * j - y * i + z * v)


def _quat_apply_tuple(q, p):
    w, x, y, z = _quat_product(_quat_product(q, (0.0,) + tuple(p)), (q[0], -q[1], -q[2], -q[3]))
    return (x, y, z)


def _origin(element):
    origin = element.find("origin")
    if origin is None:
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
    r, p, y = (x * 0.5 for x in _vector(origin.get("rpy")))
    quaternion = _quat_product(_quat_product((math.cos(y), 0.0, 0.0, math.sin(y)),
                                            (math.cos(p), 0.0, math.sin(p), 0.0)),
                               (math.cos(r), math.sin(r), 0.0, 0.0))
    return _vector(origin.get("xyz")), quaternion


def _compose(a_position, a_quaternion, b_position, b_quaternion):
    offset = _quat_apply_tuple(a_quaternion, b_position)
    return tuple(x + y for x, y in zip(a_position, offset)), _quat_product(a_quaternion, b_quaternion)


def _quat_apply(quaternion, vector):
    """Rotate by already normalized wxyz quaternions with broadcasting."""
    xyz = quaternion[..., 1:]
    uv = torch.linalg.cross(xyz, vector, dim=-1)
    return vector + 2.0 * (quaternion[..., :1] * uv + torch.linalg.cross(xyz, uv, dim=-1))


def _normalized_quaternions(quaternion, minimum_norm):
    finite = torch.isfinite(quaternion).all(dim=-1)
    sanitized = torch.where(torch.isfinite(quaternion), quaternion, torch.zeros_like(quaternion))
    scale = sanitized.abs().amax(dim=-1, keepdim=True)
    tiny = torch.finfo(quaternion.dtype).tiny
    scaled = sanitized / scale.clamp_min(tiny)
    norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    valid = finite & (scale.squeeze(-1) > minimum_norm / norm.squeeze(-1).clamp_min(tiny))
    normalized = scaled / norm.clamp_min(tiny)
    identity = torch.zeros_like(normalized)
    identity[..., 0] = 1.0
    return torch.where(valid.unsqueeze(-1), normalized, identity), valid


def _point_segment_squared_distance(point, a, b):
    delta = b - a
    denominator = delta.square().sum(dim=-1)
    tiny = torch.finfo(point.dtype).tiny
    t = ((point - a) * delta).sum(dim=-1) / denominator.clamp_min(tiny)
    t = t.clamp(0.0, 1.0)
    return (point - a - t.unsqueeze(-1) * delta).square().sum(dim=-1)


def segment_segment_distance(a, b, c, d):
    """Stable finite segment distance for tensors [...,3], including points.

    Four boundary projections plus the valid interior candidate cover the
    constrained quadratic minimum.  A cross-product denominator avoids the
    cancellation in ``(u.u)*(v.v)-(u.v)**2`` for nearly parallel segments.
    """
    u, v, w = b - a, d - c, c - a
    candidates = torch.stack((_point_segment_squared_distance(a, c, d),
                              _point_segment_squared_distance(b, c, d),
                              _point_segment_squared_distance(c, a, b),
                              _point_segment_squared_distance(d, a, b)), dim=-1)
    cross = torch.linalg.cross(u, v, dim=-1)
    denominator = cross.square().sum(dim=-1)
    tiny = torch.finfo(a.dtype).tiny
    s = (torch.linalg.cross(w, v, dim=-1) * cross).sum(dim=-1) / denominator.clamp_min(tiny)
    t = (torch.linalg.cross(w, u, dim=-1) * cross).sum(dim=-1) / denominator.clamp_min(tiny)
    interior_valid = ((denominator > tiny) & (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0))
    # Clamp invalid parameters before multiplication, avoiding irrelevant
    # overflow/NaN candidates when the two segments are almost parallel.
    difference = a + s.clamp(0.0, 1.0).unsqueeze(-1) * u - c - t.clamp(0.0, 1.0).unsqueeze(-1) * v
    interior = difference.square().sum(dim=-1)
    interior = torch.where(interior_valid, interior, torch.full_like(interior, float("inf")))
    squared = torch.minimum(candidates.amin(dim=-1), interior)
    return squared.clamp_min(0.0).sqrt()


def segment_box_distance(a, b, half_size):
    """Exact finite-segment/AABB distance, vectorized for [...,3] tensors.

    The six face crossings split squared point-box distance into at most seven
    quadratic pieces.  Minimize each piece analytically.  This is not dense
    point sampling or a capsule/box AABB approximation.
    """
    delta = b - a
    nonzero = delta != 0.0
    safe_delta = torch.where(nonzero, delta, torch.ones_like(delta))
    first = torch.where(nonzero, (-half_size - a) / safe_delta, torch.zeros_like(delta))
    second = torch.where(nonzero, (half_size - a) / safe_delta, torch.zeros_like(delta))
    ends_shape = a.shape[:-1] + (1,)
    crossings = torch.cat((torch.zeros(ends_shape, device=a.device, dtype=a.dtype),
                            first.clamp(0.0, 1.0), second.clamp(0.0, 1.0),
                            torch.ones(ends_shape, device=a.device, dtype=a.dtype)), dim=-1)
    points = crossings.sort(dim=-1).values
    lo, hi = points[..., :-1], points[..., 1:]
    midpoint = a.unsqueeze(-2) + ((lo + hi) * 0.5).unsqueeze(-1) * delta.unsqueeze(-2)
    active = midpoint.abs() > half_size.unsqueeze(-2)
    offset = torch.where(active, a.unsqueeze(-2) - midpoint.sign() * half_size.unsqueeze(-2), 0.0)
    direction = torch.where(active, delta.unsqueeze(-2), 0.0)
    denominator = direction.square().sum(dim=-1)
    numerator = (offset * direction).sum(dim=-1)
    candidate = -numerator / denominator.clamp_min(torch.finfo(a.dtype).tiny)
    candidate = torch.maximum(torch.minimum(candidate, hi), lo)
    point = a.unsqueeze(-2) + candidate.unsqueeze(-1) * delta.unsqueeze(-2)
    squared = (point.abs() - half_size.unsqueeze(-2)).clamp_min(0.0).square().sum(dim=-1)
    return squared.amin(dim=-1).clamp_min(0.0).sqrt()


class GunRobotClearance(nn.Module):
    """URDF shape registry and large-batch conservative clearance evaluator.

    ``body_names`` must match the second pose dimension.  A missing source link
    is allowed only if its fixed-joint chain reaches a supplied body.  Missing
    movable links and unsupported collision geometry fail loudly at startup.

    Invalid/near-zero quaternions or nonfinite used poses fail closed with
    ``input_valid=False``, finite negative sentinel clearances and pair index
    -1.  This is an input failure, not a claim of physical collision.
    """

    def __init__(self, urdf_path, body_names=None, device=None, dtype=torch.float32,
                 tube_collision_name="vd10_hollow_gun_collision", numerical_guard_m=1e-6,
                 minimum_quaternion_norm=1e-8):
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Geometry requires float32 or float64, not reduced-precision poses")
        if not math.isfinite(numerical_guard_m) or numerical_guard_m < 0.0:
            raise ValueError("numerical_guard_m must be finite and nonnegative")
        if not math.isfinite(minimum_quaternion_norm) or minimum_quaternion_norm <= 0.0:
            raise ValueError("minimum_quaternion_norm must be finite and positive")
        self.numerical_guard_m = float(numerical_guard_m)
        self.minimum_quaternion_norm = float(minimum_quaternion_norm)
        self.invalid_clearance_m = -1e6
        source_path = Path(urdf_path).resolve(strict=True)
        source = source_path.read_bytes()
        root = ET.fromstring(source)
        elements = root.findall("link")
        links = {element.get("name"): element for element in elements}
        if not links or None in links or len(links) != len(elements):
            raise ValueError("Expected unique named URDF links")
        self.body_names = tuple(links if body_names is None else body_names)
        if not self.body_names or len(set(self.body_names)) != len(self.body_names):
            raise ValueError("body_names must be nonempty and unique")
        body_index = {name: number for number, name in enumerate(self.body_names)}
        incoming = {}
        for joint in root.findall("joint"):
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in links or child not in links or child in incoming:
                raise ValueError("Expected a URDF tree of known links")
            position, quaternion = _origin(joint)
            incoming[child] = dict(parent=parent, kind=joint.get("type"), position=position, quaternion=quaternion)

        def carrier_for(link_name):
            current, position, quaternion = link_name, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
            seen = set()
            while current not in body_index:
                if current in seen or current not in incoming or incoming[current]["kind"] != "fixed":
                    raise ValueError("Cannot resolve actual rigid body for URDF collider on " + link_name)
                seen.add(current)
                joint = incoming[current]
                position, quaternion = _compose(joint["position"], joint["quaternion"], position, quaternion)
                current = joint["parent"]
            return current, position, quaternion

        colliders, tube = [], None
        for link_name, element in links.items():
            for number, collision in enumerate(element.findall("collision")):
                node = collision.find("geometry")
                if node is None or len(node) != 1:
                    raise ValueError("Every collision needs exactly one authored shape")
                geometry = node[0]
                primitive = geometry.tag
                if primitive not in ("box", "sphere", "cylinder"):
                    raise ValueError("Unsupported collision primitive on " + link_name + ": " + primitive)
                position, quaternion = _origin(collision)
                carrier, carrier_p, carrier_q = carrier_for(link_name)
                position, quaternion = _compose(carrier_p, carrier_q, position, quaternion)
                name = collision.get("name", link_name + "_collision_" + str(number))
                shape = dict(source_link=link_name, collision_name=name, primitive=primitive,
                             carrier_body=carrier, carrier_body_index=body_index[carrier],
                             local_position_m=position, local_quaternion_wxyz=quaternion,
                             radius_m=0.0, half_length_m=0.0, half_size_m=(0.0, 0.0, 0.0))
                if primitive == "box":
                    shape["half_size_m"] = tuple(x * 0.5 for x in _vector(geometry.get("size")))
                    if min(shape["half_size_m"]) <= 0.0:
                        raise ValueError("Box size must be positive")
                else:
                    shape["radius_m"] = float(geometry.get("radius"))
                    if not math.isfinite(shape["radius_m"]) or shape["radius_m"] <= 0.0:
                        raise ValueError("Radius must be finite and positive")
                    if primitive == "cylinder":
                        shape["half_length_m"] = 0.5 * float(geometry.get("length"))
                        if not math.isfinite(shape["half_length_m"]) or shape["half_length_m"] <= 0.0:
                            raise ValueError("Cylinder length must be finite and positive")
                if name == tube_collision_name:
                    if tube is not None or primitive != "cylinder":
                        raise ValueError("Expected one named cylindrical gun collider")
                    tube = shape
                colliders.append(shape)
        if tube is None:
            raise ValueError("Missing named gun cylinder: " + tube_collision_name)
        excluded = [shape for shape in colliders if shape["carrier_body_index"] == tube["carrier_body_index"]]
        pairs = [shape for shape in colliders if shape["carrier_body_index"] != tube["carrier_body_index"]]
        if not pairs:
            raise ValueError("No other robot collision shapes to evaluate")
        self.num_pairs = len(pairs)
        self.pair_names = tuple(shape["collision_name"] for shape in pairs)
        self.pair_source_links = tuple(shape["source_link"] for shape in pairs)
        self.tube_body_index = tube["carrier_body_index"]
        self.tube_radius_m = tube["radius_m"]
        self.tube_length_m = 2.0 * tube["half_length_m"]
        tensor_options = dict(device=device, dtype=dtype)
        self.register_buffer("shape_local_position", torch.tensor([shape["local_position_m"] for shape in pairs], **tensor_options))
        self.register_buffer("shape_local_quaternion", torch.tensor([shape["local_quaternion_wxyz"] for shape in pairs], **tensor_options))
        self.register_buffer("shape_half_size", torch.tensor([shape["half_size_m"] for shape in pairs], **tensor_options))
        self.register_buffer("shape_radius", torch.tensor([shape["radius_m"] for shape in pairs], **tensor_options))
        self.register_buffer("shape_half_length", torch.tensor([shape["half_length_m"] for shape in pairs], **tensor_options))
        self.register_buffer("carrier_body_index", torch.tensor([shape["carrier_body_index"] for shape in pairs], device=device, dtype=torch.long))
        self.register_buffer("box_pair_index", torch.tensor([i for i, shape in enumerate(pairs) if shape["primitive"] == "box"], device=device, dtype=torch.long))
        self.register_buffer("round_pair_index", torch.tensor([i for i, shape in enumerate(pairs) if shape["primitive"] != "box"], device=device, dtype=torch.long))
        self.register_buffer("used_body_index", torch.tensor(sorted({tube["carrier_body_index"]} | {shape["carrier_body_index"] for shape in pairs}), device=device, dtype=torch.long))
        local_axis = _quat_apply_tuple(tube["local_quaternion_wxyz"], (0.0, 0.0, 1.0))
        local_endpoints = [[tube["local_position_m"][k] + sign * tube["half_length_m"] * local_axis[k] for k in range(3)]
                           for sign in (-1.0, 1.0)]
        self.register_buffer("tube_local_endpoints", torch.tensor(local_endpoints, **tensor_options))
        self.metadata = dict(
            version=1, source_urdf=str(source_path), source_urdf_sha256=hashlib.sha256(source).hexdigest(),
            body_names=list(self.body_names), pair_names=list(self.pair_names), pairs=pairs,
            tube=dict(**tube, length_m=self.tube_length_m, outer_diameter_m=2.0 * self.tube_radius_m),
            excluded_collisions=excluded,
            excluded_reason="All collision shapes on the gun's own merged rigid body are excluded",
            links_without_authored_collision=sorted(name for name, element in links.items() if not element.findall("collision")),
            numerical_guard_m=self.numerical_guard_m, minimum_quaternion_norm=self.minimum_quaternion_norm,
            invalid_clearance_m=self.invalid_clearance_m, actor_input=False,
            scope="Authored collision shapes; no unmodelled visible surfaces or continuous swept-motion certificate",
            semantics="Capsule-envelope clearance lower bound; negative means conservative keep-out, NOT proven physical collision/penetration/contact force",
        )

    def forward(self, body_position_w, body_quaternion_w):
        """Compute all pairs without CPU synchronization or per-env loops."""
        if (body_position_w.ndim != 3 or body_position_w.shape[1:] != (len(self.body_names), 3)
                or body_quaternion_w.shape != body_position_w.shape[:-1] + (4,)):
            raise ValueError("Expected matching [env, body, 3] positions and [env, body, 4] wxyz quaternions")
        if (body_position_w.dtype not in (torch.float32, torch.float64)
                or body_position_w.dtype != self.shape_local_position.dtype
                or body_quaternion_w.dtype != body_position_w.dtype
                or body_position_w.device != self.shape_local_position.device
                or body_quaternion_w.device != body_position_w.device):
            raise ValueError("Poses and geometry buffers must have the same device and float32/float64 dtype")
        quaternion, quaternion_valid = _normalized_quaternions(body_quaternion_w, self.minimum_quaternion_norm)
        position_valid = torch.isfinite(body_position_w).all(dim=-1)
        input_valid = (position_valid & quaternion_valid).index_select(1, self.used_body_index).all(dim=1)
        position = torch.where(torch.isfinite(body_position_w), body_position_w, torch.zeros_like(body_position_w))
        tube_q = quaternion[:, self.tube_body_index, :].unsqueeze(1).expand(-1, 2, -1)
        tube_endpoints = (position[:, self.tube_body_index, :].unsqueeze(1)
                          + _quat_apply(tube_q, self.tube_local_endpoints.unsqueeze(0).expand(body_position_w.shape[0], -1, -1)))
        carrier_q = quaternion.index_select(1, self.carrier_body_index)
        carrier_p = position.index_select(1, self.carrier_body_index)
        local_shape_p = self.shape_local_position.unsqueeze(0).expand(body_position_w.shape[0], -1, -1)
        shape_centre = carrier_p + _quat_apply(carrier_q, local_shape_p)
        # Rotate world endpoints into each shape's carrier, then local origin
        # frame.  Two inverse rotations avoid materializing [env,pair,3,3].
        inverse_carrier = torch.cat((carrier_q[..., :1], -carrier_q[..., 1:]), dim=-1)
        inverse_local = torch.cat((self.shape_local_quaternion[..., :1], -self.shape_local_quaternion[..., 1:]), dim=-1)
        inverse_local = inverse_local.unsqueeze(0).expand(body_position_w.shape[0], -1, -1)
        a = _quat_apply(inverse_local, _quat_apply(inverse_carrier, tube_endpoints[:, 0, :].unsqueeze(1) - shape_centre))
        b = _quat_apply(inverse_local, _quat_apply(inverse_carrier, tube_endpoints[:, 1, :].unsqueeze(1) - shape_centre))
        lower_bounds = body_position_w.new_empty((body_position_w.shape[0], self.num_pairs))
        if self.box_pair_index.numel():
            box_a, box_b = a.index_select(1, self.box_pair_index), b.index_select(1, self.box_pair_index)
            box_half = self.shape_half_size.index_select(0, self.box_pair_index).unsqueeze(0)
            box_bound = segment_box_distance(box_a, box_b, box_half) - self.tube_radius_m
            lower_bounds.index_copy_(1, self.box_pair_index, box_bound)
        if self.round_pair_index.numel():
            round_a, round_b = a.index_select(1, self.round_pair_index), b.index_select(1, self.round_pair_index)
            half = self.shape_half_length.index_select(0, self.round_pair_index)
            c = torch.stack((torch.zeros_like(half), torch.zeros_like(half), -half), dim=-1).unsqueeze(0)
            d = -c
            bound = (segment_segment_distance(round_a, round_b, c, d) - self.tube_radius_m
                     - self.shape_radius.index_select(0, self.round_pair_index).unsqueeze(0))
            lower_bounds.index_copy_(1, self.round_pair_index, bound)
        lower_bounds = lower_bounds - self.numerical_guard_m
        input_valid = (input_valid & torch.isfinite(lower_bounds).all(dim=1)
                       & torch.isfinite(tube_endpoints).all(dim=-1).all(dim=-1))
        lower_bounds = torch.where(input_valid.unsqueeze(1), lower_bounds, torch.full_like(lower_bounds, self.invalid_clearance_m))
        minimum, index = lower_bounds.min(dim=1)
        index = torch.where(input_valid, index, torch.full_like(index, -1))
        tube_endpoints = torch.nan_to_num(tube_endpoints, nan=0.0, posinf=1e6, neginf=-1e6)
        return dict(clearance_lower_bound_m=lower_bounds, minimum_clearance_lower_bound_m=minimum,
                    minimum_pair_index=index, tube_endpoints_w=tube_endpoints, input_valid=input_valid)
