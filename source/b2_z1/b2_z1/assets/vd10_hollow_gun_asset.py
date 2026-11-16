"""Build a separate, measured hollow-gun asset; never overwrite VD3/VD9.

The 3-D-printed clamp is massless. Its centre is 100 mm above the Z1 flange
centre in the flange's local +Z direction. The tube points along local +X;
the clamp is 200 mm from the rear end, leaving 900 mm to the tip. These clamp
distances are modelling choices, not dimensions measured from the photograph.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET


ASSETS = Path(__file__).resolve().parent / "b2_z1"
SOURCE = ASSETS / "b2_z1_vd3_zero_gun.urdf"
DESTINATION = ASSETS / "b2_z1_vd10_hollow_gun.urdf"
MESH = ASSETS / "gun/meshes/vd10_hollow_tube.stl"
SOURCE_SHA256 = "a6a1f7143d1ce33d6e25b3cbaf7809d17a971c1ec793e6055b365fda9bb39be0"

OUTER_RADIUS_M = 0.028
INNER_RADIUS_M = 0.026
LENGTH_M = 1.1
MASS_KG = 2.93
REAR_FROM_CLAMP_M = 0.2
TIP_FROM_CLAMP_M = LENGTH_M - REAR_FROM_CLAMP_M
# joint_mounter in the preserved source URDF puts the flange at link06 +X 75 mm.
CLAMP_IN_LINK06_M = (0.075, 0.0, 0.1)
CENTRE_IN_CLAMP_M = (TIP_FROM_CLAMP_M - REAR_FROM_CLAMP_M) / 2.0
TCP_IN_LINK06_M = (CLAMP_IN_LINK06_M[0] + TIP_FROM_CLAMP_M, 0.0, CLAMP_IN_LINK06_M[2])
AXIAL_INERTIA = MASS_KG * (OUTER_RADIUS_M**2 + INNER_RADIUS_M**2) / 2.0
TRANSVERSE_INERTIA = MASS_KG * (3 * (OUTER_RADIUS_M**2 + INNER_RADIUS_M**2) + LENGTH_M**2) / 12.0


def _tube_triangles(sides=48):
    """Closed annular surface, centred on the local X axis (visual only)."""
    if sides < 8:
        raise ValueError("At least eight tube sides are required")
    for i in range(sides):
        a, b = (2 * math.pi * j / sides for j in (i, i + 1))
        for radius, inside in ((OUTER_RADIUS_M, False), (INNER_RADIUS_M, True)):
            p, q = ((radius * math.cos(t), radius * math.sin(t)) for t in (a, b))
            low_p = (-LENGTH_M / 2, *p)
            low_q = (-LENGTH_M / 2, *q)
            high_p = (LENGTH_M / 2, *p)
            high_q = (LENGTH_M / 2, *q)
            if inside:
                yield low_p, high_p, low_q
                yield low_q, high_p, high_q
            else:
                yield low_p, low_q, high_p
                yield low_q, high_q, high_p
        for x, reverse in ((-LENGTH_M / 2, False), (LENGTH_M / 2, True)):
            outer_p = (x, OUTER_RADIUS_M * math.cos(a), OUTER_RADIUS_M * math.sin(a))
            outer_q = (x, OUTER_RADIUS_M * math.cos(b), OUTER_RADIUS_M * math.sin(b))
            inner_p = (x, INNER_RADIUS_M * math.cos(a), INNER_RADIUS_M * math.sin(a))
            inner_q = (x, INNER_RADIUS_M * math.cos(b), INNER_RADIUS_M * math.sin(b))
            triangles = ((outer_p, inner_p, outer_q), (outer_q, inner_p, inner_q))
            for triangle in triangles:
                yield triangle[::-1] if reverse else triangle


def _binary_stl():
    triangles = list(_tube_triangles())
    payload = bytearray(b"VD10 measured hollow tube".ljust(80, b"\0"))
    payload.extend(struct.pack("<I", len(triangles)))
    for triangle in triangles:
        payload.extend(struct.pack("<12fH", 0., 0., 0., *(v for point in triangle for v in point), 0))
    return bytes(payload)


def _merge_inertial(flange):
    """Combine the measured tube with link06 using the parallel-axis theorem."""
    inertial = flange.find("inertial")
    old_mass = float(inertial.find("mass").get("value"))
    old_com = [float(v) for v in inertial.find("origin").get("xyz").split()]
    new_com = [CLAMP_IN_LINK06_M[0] + CENTRE_IN_CLAMP_M, 0.0, CLAMP_IN_LINK06_M[2]]
    total_mass = old_mass + MASS_KG
    combined_com = [(old_mass * old_com[i] + MASS_KG * new_com[i]) / total_mass for i in range(3)]
    original = inertial.find("inertia")
    old_tensor = [[float(original.get("i" + min("xyz"[i], "xyz"[j]) + max("xyz"[i], "xyz"[j])))
                   for j in range(3)] for i in range(3)]
    tube_tensor = [[AXIAL_INERTIA, 0.0, 0.0], [0.0, TRANSVERSE_INERTIA, 0.0],
                   [0.0, 0.0, TRANSVERSE_INERTIA]]
    tensor = [[old_tensor[i][j] + tube_tensor[i][j] for j in range(3)] for i in range(3)]
    for mass, com in ((old_mass, old_com), (MASS_KG, new_com)):
        d = [com[i] - combined_com[i] for i in range(3)]
        distance_sq = sum(v * v for v in d)
        for i in range(3):
            for j in range(3):
                tensor[i][j] += mass * ((distance_sq if i == j else 0.0) - d[i] * d[j])
    inertial.find("origin").set("xyz", " ".join(f"{value:.17g}" for value in combined_com))
    inertial.find("mass").set("value", f"{total_mass:.17g}")
    for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        original.set("i" + "xyz"[i] + "xyz"[j], f"{tensor[i][j]:.17g}")
    return combined_com, tensor


def generate():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("VD3 source asset changed; do not silently modify historical models")
    root = ET.parse(SOURCE).getroot()
    flange = root.find("link[@name='link06']")
    if flange is None:
        raise ValueError("Missing Z1 flange link06")
    for kind in ("visual", "collision"):
        shapes = [s for s in flange.findall(kind) if s.get("name", "").startswith("vd3_zero_load_gun_")]
        if len(shapes) != 1:
            raise ValueError(f"Expected exactly one historical gun {kind}")
        flange.remove(shapes[0])
    combined_com, combined_tensor = _merge_inertial(flange)
    centre = (CLAMP_IN_LINK06_M[0] + CENTRE_IN_CLAMP_M, 0.0, CLAMP_IN_LINK06_M[2])
    centre_xyz = " ".join(f"{v:.17g}" for v in centre)
    visual = ET.SubElement(flange, "visual", name="vd10_hollow_gun_visual")
    ET.SubElement(visual, "origin", xyz=centre_xyz, rpy="0 0 0")
    ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename="gun/meshes/vd10_hollow_tube.stl")
    collision = ET.SubElement(flange, "collision", name="vd10_hollow_gun_collision")
    ET.SubElement(collision, "origin", xyz=centre_xyz, rpy=f"0 {math.pi/2:.17g} 0")
    ET.SubElement(ET.SubElement(collision, "geometry"), "cylinder", radius="0.028", length="1.1")

    xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    mesh = _binary_stl()
    MESH.write_bytes(mesh)
    DESTINATION.write_bytes(xml)
    manifest = {
        "version": "VD10 candidate", "source_sha256": SOURCE_SHA256,
        "urdf_sha256": hashlib.sha256(xml).hexdigest(), "mesh_sha256": hashlib.sha256(mesh).hexdigest(),
        "gun_mass_kg": MASS_KG, "nominal_robot_mass_kg": 78.9990202 + MASS_KG,
        "outer_diameter_m": 2 * OUTER_RADIUS_M, "inner_diameter_m": 2 * INNER_RADIUS_M,
        "length_m": LENGTH_M, "axial_inertia_kg_m2": AXIAL_INERTIA,
        "transverse_inertia_kg_m2": TRANSVERSE_INERTIA,
        "clamp_center_in_link06_m": CLAMP_IN_LINK06_M, "rear_from_clamp_m": REAR_FROM_CLAMP_M,
        "tip_from_clamp_m": TIP_FROM_CLAMP_M, "tcp_in_link06_m": TCP_IN_LINK06_M,
        "link06_combined_mass_kg": .2887581 + MASS_KG,
        "link06_combined_com_m": combined_com, "link06_combined_inertia_kg_m2": combined_tensor,
        "gun_axis_in_link06": [1, 0, 0], "clamp_mass_kg": 0,
        "assumptions": ["100 mm upward means local +Z at the joint6=0 flange pose",
                        "clamp centre is 200 mm from the tube rear end",
                        "collision uses solid outer envelope; the visual mesh and inertial are hollow"],
    }
    DESTINATION.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
