"""Static VD10 candidate checks, with no Isaac simulation or policy rollouts."""

import hashlib
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1] / "source/b2_z1/b2_z1/assets"
spec = importlib.util.spec_from_file_location("vd10_hollow_asset", ROOT / "vd10_hollow_gun_asset.py")
asset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asset)


class MeasuredTubeTests(unittest.TestCase):
    def test_original_asset_remains_unchanged(self):
        self.assertEqual(hashlib.sha256(asset.SOURCE.read_bytes()).hexdigest(), asset.SOURCE_SHA256)

    def test_geometry_mass_and_flange_mount(self):
        root = ET.parse(asset.DESTINATION).getroot()
        link = root.find("link[@name='link06']")
        self.assertIsNotNone(link)
        self.assertIsNone(root.find("link[@name='link_gun']"))
        self.assertIsNone(root.find("link[@name='link_vd10_gun']"))
        self.assertAlmostEqual(float(link.find("inertial/mass").get("value")), 3.2187581)
        inertia = link.find("inertial/inertia")
        self.assertAlmostEqual(float(inertia.get("ixx")), .005026923864946012, places=11)
        self.assertAlmostEqual(float(inertia.get("iyy")), .34159733897570543, places=11)
        self.assertAlmostEqual(float(inertia.get("izz")), .33889193093571174, places=11)
        self.assertEqual(link.find("visual[@name='vd10_hollow_gun_visual']/geometry/mesh").get("filename"),
                         "gun/meshes/vd10_hollow_tube.stl")
        collision = link.find("collision[@name='vd10_hollow_gun_collision']/geometry/cylinder")
        self.assertEqual((float(collision.get("radius")), float(collision.get("length"))), (.028, 1.1))
        self.assertAlmostEqual(float(link.find("visual[@name='vd10_hollow_gun_visual']/origin").get("xyz").split()[0]), .425)
        self.assertAlmostEqual(float(link.find("inertial/origin").get("xyz").split()[0]), .38903995318750123)
        self.assertAlmostEqual(sum(float(m.get("value")) for m in root.findall("link/inertial/mass")),
                               81.9290202)

    def test_mesh_has_open_bore_but_closed_surface(self):
        import trimesh
        mesh = trimesh.load(asset.MESH, force="mesh")
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertAlmostEqual(mesh.volume, math.pi * (.028**2 - .026**2) * 1.1, delta=2e-6)
        self.assertAlmostEqual(mesh.bounds[1, 0] - mesh.bounds[0, 0], 1.1, places=6)

    def test_tcp_profile_isolated_from_historical_asset(self):
        spec = importlib.util.spec_from_file_location(
            "tcp_for_vd10_test", ROOT.parent / "tasks/manager_based/b2_z1/mdp/tcp_frame.py")
        tcp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tcp)
        robot = SimpleNamespace(cfg=SimpleNamespace(spawn=SimpleNamespace(asset_path=str(asset.DESTINATION))))
        self.assertEqual(tcp.tcp_transform_for_asset(robot),
                         (tcp.HOLLOW_GUN_TCP_POS_IN_LINK06, (1., 0., 0., 0.)))
        robot.cfg.spawn.asset_path = str(asset.SOURCE)
        self.assertEqual(tcp.tcp_transform_for_asset(robot),
                         (tcp.TCP_POS_IN_LINK06, tcp.TCP_QUAT_IN_LINK06))


if __name__ == "__main__":
    unittest.main()
