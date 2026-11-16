"""Independent 750mm clamp-to-tip tube; preserved flange centre is link06 +75mm.

Keep all VD3/VD10 assets intact. Reuse the existing measured hollow mesh but
rebuild the merged link06 COM/inertia from the unloaded VD3 link06.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ASSETS = Path(__file__).resolve().parent
DESTINATION = ASSETS/'b2_z1/b2_z1_vd13_mount75.urdf'
CLAMP_IN_LINK06_M = (.075, 0., .1)
TIP_FROM_CLAMP_M = .75
REAR_FROM_CLAMP_M = .35
TCP_IN_LINK06_M = (.825, 0., .1)


def payload():
    spec = importlib.util.spec_from_file_location('vd13_preserved_tube_math', ASSETS/'vd10_hollow_gun_asset.py')
    tube = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tube)
    if hashlib.sha256(tube.SOURCE.read_bytes()).hexdigest() != tube.SOURCE_SHA256:
        raise ValueError('Preserved unloaded source changed')
    source_manifest = json.loads(tube.DESTINATION.with_suffix('.manifest.json').read_text())
    if hashlib.sha256(tube.MESH.read_bytes()).hexdigest() != source_manifest['mesh_sha256']:
        raise ValueError('Preserved hollow-tube visual mesh changed')
    tube.CLAMP_IN_LINK06_M = CLAMP_IN_LINK06_M
    tube.TIP_FROM_CLAMP_M, tube.REAR_FROM_CLAMP_M = TIP_FROM_CLAMP_M, REAR_FROM_CLAMP_M
    tube.CENTRE_IN_CLAMP_M = (TIP_FROM_CLAMP_M-REAR_FROM_CLAMP_M)/2
    root = ET.parse(tube.SOURCE).getroot()
    flange = root.find("link[@name='link06']")
    for kind in ('visual', 'collision'):
        shapes = [s for s in flange.findall(kind) if s.get('name', '').startswith('vd3_zero_load_gun_')]
        if len(shapes) != 1:
            raise ValueError('Unexpected preserved unloaded gun geometry')
        flange.remove(shapes[0])
    com, inertia = tube._merge_inertial(flange)
    centre = (.275, 0., .1)
    visual = ET.SubElement(flange, 'visual', name='vd10_hollow_gun_visual')
    ET.SubElement(visual, 'origin', xyz='0.275 0 0.1', rpy='0 0 0')
    ET.SubElement(ET.SubElement(visual, 'geometry'), 'mesh', filename='gun/meshes/vd10_hollow_tube.stl')
    collision = ET.SubElement(flange, 'collision', name='vd10_hollow_gun_collision')
    ET.SubElement(collision, 'origin', xyz='0.275 0 0.1', rpy='0 1.5707963267948966 0')
    ET.SubElement(ET.SubElement(collision, 'geometry'), 'cylinder', radius='.028', length='1.1')
    xml = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    manifest = dict(version='VD13 mount75', source_sha256=tube.SOURCE_SHA256,
                    urdf_sha256=hashlib.sha256(xml).hexdigest(), mesh_sha256=source_manifest['mesh_sha256'],
                    gun_mass_kg=2.93, nominal_robot_mass_kg=81.9290202, length_m=1.1,
                    outer_diameter_m=.056, inner_diameter_m=.052,
                    clamp_center_in_link06_m=CLAMP_IN_LINK06_M,
                    tip_from_clamp_m=TIP_FROM_CLAMP_M, rear_from_clamp_m=REAR_FROM_CLAMP_M,
                    tcp_in_link06_m=TCP_IN_LINK06_M, tube_com_in_link06_m=centre,
                    link06_combined_mass_kg=3.2187581, link06_combined_com_m=com,
                    link06_combined_inertia_kg_m2=inertia, clamp_mass_kg=0,
                    note='Flange centre is link06 +75mm, NOT link06 origin; new assets only')
    return xml, manifest


def generate():
    xml, manifest = payload()
    text = json.dumps(manifest, indent=2)+'\n'
    # New named outputs only; refuse to overwrite different user assets.
    for path, data in ((DESTINATION, xml), (DESTINATION.with_suffix('.manifest.json'), text.encode())):
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError('Refusing to replace different candidate asset: '+str(path))
        if not path.exists():
            path.write_bytes(data)
    return manifest


if __name__ == '__main__':
    print(json.dumps(generate(), indent=2))
