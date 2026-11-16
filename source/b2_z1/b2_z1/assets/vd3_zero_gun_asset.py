"""Generate the VD3 zero-load gun asset without zero-mass PhysX rigid bodies.

The original URDF is kept intact. Gun visual/collision shapes become shapes of
link06; its positive mass and inertia remain unchanged. The TCP is a virtual
frame, not another physical body. This module uses only the standard library.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


ASSET_DIR = Path(__file__).resolve().parent / "b2_z1"
SOURCE_URDF = ASSET_DIR / "b2_z1_gun.urdf"
VD3_URDF = ASSET_DIR / "b2_z1_vd3_zero_gun.urdf"
FIXED_CHAIN = ("joint_mounter", "joint_gun_mounter", "link_dummy_target_joint")
REMOVED_LINKS = ("link_mounter", "link_gun", "dummy_target")


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def origin_matrix(origin):
    xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()] if origin is not None else [0.] * 3
    r, p, y = [float(x) for x in origin.get("rpy", "0 0 0").split()] if origin is not None else [0.] * 3
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, xyz[0]],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, xyz[1]],
        [-sp, cp*sr, cp*cr, xyz[2]],
        [0., 0., 0., 1.],
    ]


def _matrix_origin(transform):
    # All transformed gun shape orientations are away from the Euler singularity.
    pitch = math.asin(max(-1., min(1., -transform[2][0])))
    roll = math.atan2(transform[2][1], transform[2][2])
    yaw = math.atan2(transform[1][0], transform[0][0])
    return ET.Element("origin", {
        "xyz": " ".join(format(transform[i][3], ".17g") for i in range(3)),
        "rpy": " ".join(format(value, ".17g") for value in (roll, pitch, yaw)),
    })


def fixed_chain_transform(root, joint_names=FIXED_CHAIN):
    transform = origin_matrix(None)
    for name in joint_names:
        joint = root.find(f"joint[@name='{name}']")
        if joint is None or joint.get("type") != "fixed":
            raise ValueError(f"Expected fixed source joint {name!r}.")
        transform = _matmul(transform, origin_matrix(joint.find("origin")))
    return transform


def generate_zero_gun_asset(source_path=SOURCE_URDF, output_path=VD3_URDF):
    """Write a deterministic VD3 URDF and a sidecar manifest; return manifest."""
    source_path, output_path = Path(source_path), Path(output_path)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("The source URDF must remain unchanged.")
    root = ET.fromstring(source_path.read_bytes())
    link06 = root.find("link[@name='link06']")
    gun = root.find("link[@name='link_gun']")
    if link06 is None or gun is None:
        raise ValueError("Expected source link06 and link_gun.")
    original_inertial = ET.tostring(link06.find("inertial"))
    gun_transform = fixed_chain_transform(root, FIXED_CHAIN[:2])
    tcp_transform = fixed_chain_transform(root)
    counts = {}
    for kind in ("visual", "collision"):
        shapes = gun.findall(kind)
        counts[kind] = len(shapes)
        for shape in shapes:
            copied = deepcopy(shape)
            old_origin = copied.find("origin")
            combined = _matmul(gun_transform, origin_matrix(old_origin))
            if old_origin is not None:
                copied.remove(old_origin)
            copied.insert(0, _matrix_origin(combined))
            copied.set("name", f"vd3_zero_load_gun_{kind}")
            link06.append(copied)
    for name in FIXED_CHAIN:
        root.remove(root.find(f"joint[@name='{name}']"))
    for name in REMOVED_LINKS:
        root.remove(root.find(f"link[@name='{name}']"))
    if ET.tostring(link06.find("inertial")) != original_inertial:
        raise AssertionError("link06 inertia changed while moving gun geometry.")
    mass = sum(float(e.get("value")) for e in root.findall("link/inertial/mass"))
    if not math.isclose(mass, 78.9990202, abs_tol=1.e-9):
        raise ValueError(f"Unexpected VD3 nominal mass: {mass}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    output_path.write_bytes(payload)
    manifest = {
        "version": "VD3", "gun_added_mass_kg": 0.0, "gun_added_inertia": [0.0]*9,
        "nominal_robot_mass_kg": mass, "physical_tcp_body": False,
        "tcp_parent_body": "link06", "tcp_transform_in_link06": tcp_transform,
        "removed_fixed_bodies": list(REMOVED_LINKS), "gun_shape_count": counts,
        "source_urdf": source_path.name, "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "generated_urdf": output_path.name, "generated_sha256": hashlib.sha256(payload).hexdigest(),
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_URDF)
    parser.add_argument("--output", type=Path, default=VD3_URDF)
    args = parser.parse_args()
    print(json.dumps(generate_zero_gun_asset(args.source, args.output), indent=2))
