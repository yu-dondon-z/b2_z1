"""Derive a VB10 URDF with a flat-ended gun cylinder, without touching legs.

The URDF importer turns primitive cylinders into longer capsules globally.
Use a conservative convex mesh only for link_gun; retain mass, inertia, TCP,
all other colliders and the original URDF. Generated assets are cached in /tmp.
"""
import hashlib
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


def cylinder_obj(radius=.028, length=1.10, sides=48):
    if not all(math.isfinite(v) and v > 0 for v in (radius,length)) or sides < 12:
        raise ValueError("Invalid gun cylinder dimensions")
    # Circumscribed polygon: each side is tangent to the URDF radius. Never
    # shrink the cylinder to make contact disappear (radial excess <0.22%).
    r = radius / math.cos(math.pi / sides)
    vertices = [(r*math.cos(2*math.pi*i/sides), r*math.sin(2*math.pi*i/sides), z)
                for z in (-length/2, length/2) for i in range(sides)]
    vertices += [(0., 0., -length/2), (0., 0., length/2)]
    faces = []
    for i in range(sides):
        j = (i+1) % sides
        faces += [(i, j, j+sides), (i, j+sides, i+sides),
                  (2*sides, j, i), (2*sides+1, i+sides, j+sides)]
    return "# VB10 flat-ended physical gun cylinder; units metres\n" + "".join(
        "v %.10f %.10f %.10f\n" % v for v in vertices
    ) + "".join("f %d %d %d\n" % tuple(k+1 for k in f) for f in faces)


def flat_gun_urdf(source_path):
    source = Path(source_path).resolve()
    original = source.read_bytes()
    obj = cylinder_obj()
    digest = hashlib.sha256(original + obj.encode()).hexdigest()[:20]
    cache = Path(tempfile.gettempdir()) / ("b2_z1_vb10_" + digest)
    cache.mkdir(exist_ok=True)
    mesh_path, result = cache / "gun_flat_cylinder.obj", cache / "robot.urdf"
    if result.is_file() and mesh_path.is_file():
        return str(result)
    root = ET.fromstring(original)
    gun = root.find("./link[@name='link_gun']")
    collisions = gun.findall('collision')
    if len(collisions) != 1:
        raise ValueError("VB10 expects the original single gun collider")
    geometry = collisions[0].find('geometry')
    cylinder = geometry.find('cylinder')
    if cylinder is None or abs(float(cylinder.get('radius'))-.028)>1e-8 or abs(float(cylinder.get('length'))-1.1)>1e-8:
        raise ValueError("Gun URDF dimensions changed; recheck collider/TCP clearance")
    geometry.remove(cylinder)
    ET.SubElement(geometry, 'mesh', filename=str(mesh_path))
    # A cache URDF is outside the asset directory: resolve existing visual and
    # collision meshes, preserving their scales and all mechanical properties.
    for mesh in root.findall('.//mesh'):
        name = mesh.get('filename')
        if name.startswith('package://'):
            raise ValueError("Resolve package mesh paths before deriving VB10 assets")
        if not Path(name).is_absolute():
            mesh.set('filename', str((source.parent / name).resolve()))
    # Atomic cache publication supports simultaneous read-only config probes.
    import os
    with tempfile.NamedTemporaryFile(dir=cache, suffix='.obj', delete=False) as f:
        f.write(obj.encode()); obj_tmp = f.name
    os.replace(obj_tmp, mesh_path)
    with tempfile.NamedTemporaryFile(dir=cache, suffix='.urdf', delete=False) as f:
        f.write(ET.tostring(root, encoding='utf-8', xml_declaration=True)); xml_tmp = f.name
    os.replace(xml_tmp, result)
    return str(result)
