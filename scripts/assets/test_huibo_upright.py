"""CPU-only regression checks for the Huibo axial, top-down gun mount.

Run with ``python scripts/assets/test_huibo_upright.py``.  No Isaac import,
simulator, policy, or robot connection is involved.  Forward kinematics is
independent NumPy math over the actual generated URDF, not production FK.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "source/b2_z1/b2_z1/assets"
MODEL_DIR = ASSETS / "b2_z1"
ARM_NAMES = tuple(f"right_joint{i}" for i in range(1, 8))


def read_pose_constants():
    """Read literal configuration without executing the Isaac-dependent file."""
    tree = ast.parse((ASSETS / "b2_huibo.py").read_text())
    constants = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id == "HUIBO_INITIAL_JOINT_POS":
            # The config uses dict(zip(HUIBO_ARM_JOINT_NAMES, (seven literals))).
            expression = node.value
            assert isinstance(expression, ast.Call) and expression.func.id == "dict"
            zipped = expression.args[0]
            assert isinstance(zipped, ast.Call) and zipped.func.id == "zip"
            assert zipped.args[0].id == "HUIBO_ARM_JOINT_NAMES"
            constants[target.id] = dict(zip(ARM_NAMES, ast.literal_eval(zipped.args[1])))
        elif target.id.startswith("HUIBO_"):
            try:
                constants[target.id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
    return constants


def vector(element, attribute, default="0 0 0"):
    return np.fromstring(default if element is None else element.get(attribute, default), sep=" ")


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * skew + (1. - np.cos(angle)) * (skew @ skew)


def forward_kinematics(urdf, positions):
    frames = {"base_link": np.eye(4)}
    pending = list(urdf.findall("joint"))
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in frames:
                continue
            origin = joint.find("origin")
            roll, pitch, yaw = vector(origin, "rpy")
            transform = np.eye(4)
            transform[:3, :3] = (
                axis_rotation([0, 0, 1], yaw)
                @ axis_rotation([0, 1, 0], pitch)
                @ axis_rotation([1, 0, 0], roll)
            )
            transform[:3, 3] = vector(origin, "xyz")
            if joint.get("type") in ("revolute", "continuous"):
                transform[:3, :3] = transform[:3, :3] @ axis_rotation(
                    vector(joint.find("axis"), "xyz", "1 0 0"),
                    positions.get(joint.get("name"), 0.),
                )
            frames[joint.find("child").get("link")] = frames[parent] @ transform
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise AssertionError("unconnected URDF joints: " + str([j.get("name") for j in pending]))
    return frames


def signature(element):
    if element is None:
        return None
    return (element.tag, tuple(sorted(element.attrib.items())),
            (element.text or "").strip(), tuple(signature(child) for child in element))


class HuiboUprightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.constants = read_pose_constants()
        cls.urdf = ET.parse(MODEL_DIR / "b2_huibo.urdf").getroot()
        cls.joints = {joint.get("name"): joint for joint in cls.urdf.findall("joint")}
        cls.q = cls.constants["HUIBO_INITIAL_JOINT_POS"]
        cls.frames = forward_kinematics(cls.urdf, cls.q)

    def test_all_seven_joints_are_finite_and_keep_original_limit_margin(self):
        self.assertEqual(tuple(self.q), ARM_NAMES)
        required = self.constants["HUIBO_INITIAL_MIN_LIMIT_MARGIN_RAD"]
        self.assertGreaterEqual(required, .099)
        for name, value in self.q.items():
            with self.subTest(joint=name):
                self.assertTrue(np.isfinite(value))
                limit = self.joints[name].find("limit")
                self.assertGreaterEqual(value - float(limit.get("lower")), required)
                self.assertGreaterEqual(float(limit.get("upper")) - value, required)

    def test_actual_fk_preserves_the_rear_tcp_work_point(self):
        expected = self.constants["HUIBO_INITIAL_TCP_B"]
        np.testing.assert_allclose(expected, [-.775, 0., -.48], atol=1e-12, rtol=0)
        np.testing.assert_allclose(self.frames["dummy_target"][:3, 3], expected, atol=1e-8, rtol=0)

    def test_command_orientation_constant_matches_actual_tcp_fk(self):
        roll, pitch, yaw = self.constants["HUIBO_TCP_ORIENTATION_RPY"]
        expected = (axis_rotation([0, 0, 1], yaw)
                    @ axis_rotation([0, 1, 0], pitch)
                    @ axis_rotation([1, 0, 0], roll))
        np.testing.assert_allclose(self.frames["dummy_target"][:3, :3], expected,
                                   atol=1e-12, rtol=0)
        config = ROOT / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/b2_huibo_dual_vision_env_cfg.py"
        expressions = {
            ast.unparse(target): node.value
            for node in ast.walk(ast.parse(config.read_text())) if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Attribute)
        }
        expression = expressions["command.tool_orientation_rpy"]
        self.assertIsInstance(expression, ast.Name)
        self.assertEqual(expression.id, "HUIBO_TCP_ORIENTATION_RPY")

    def test_grip_is_axial_not_a_ninety_degree_side_mount(self):
        flange_forward = self.frames["right_arm_end_link"][:3, 0]
        gun_forward = -self.frames["link_gun"][:3, 2]
        np.testing.assert_allclose(flange_forward, gun_forward, atol=1e-10, rtol=0)
        flange_to_tip = (self.frames["dummy_target"][:3, 3]
                         - self.frames["right_arm_end_link"][:3, 3])
        self.assertGreater(np.dot(flange_forward, flange_to_tip), 0)
        np.testing.assert_allclose(np.cross(flange_forward, flange_to_tip), 0., atol=1e-10, rtol=0)

    def test_gun_points_down_with_documented_real_tilt_not_fake_vertical(self):
        direction = -self.frames["link_gun"][:3, 2]
        tilt = np.degrees(np.arccos(np.clip(-direction[2], -1., 1.)))
        self.assertLessEqual(tilt, self.constants["HUIBO_INITIAL_MAX_TILT_DEG"])
        self.assertGreater(tilt, 13.)
        self.assertLessEqual(self.constants["HUIBO_INITIAL_MAX_TILT_DEG"], 14.)
        self.assertLess(direction[2], -.97)

    def test_shorter_huibo_cylinder_starts_below_the_wrist_and_ends_at_tcp(self):
        config = ROOT / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/b2_huibo_dual_vision_env_cfg.py"
        tree = ast.parse(config.read_text())
        expressions = {
            ast.unparse(target): node.value
            for node in ast.walk(tree) if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Attribute)
        }
        self.assertEqual(ast.unparse(expressions["gun.spawn.height"]), "HUIBO_GUN_LENGTH_M")
        local = {"HUIBO_GUN_LENGTH_M": self.constants["HUIBO_GUN_LENGTH_M"]}
        center = np.array(eval(compile(ast.Expression(expressions["gun.init_state.pos"]),
                                      str(config), "eval"), {"__builtins__": {}}, local))
        length = local["HUIBO_GUN_LENGTH_M"]
        self.assertAlmostEqual(length, .60)
        bottom = center - np.array([0., 0., length / 2.])
        top = center + np.array([0., 0., length / 2.])
        np.testing.assert_allclose(bottom, [0., -.03, -.62], atol=1e-12, rtol=0)
        self.assertLess(top[2], 0.)
        tool_to_tip = vector(self.joints["link_dummy_target_joint"].find("origin"), "xyz")
        np.testing.assert_allclose(bottom, tool_to_tip, atol=1e-12, rtol=0)

    def test_axial_adapter_visual_and_collision_bridge_wrist_to_gun(self):
        gun = next(link for link in self.urdf.findall("link") if link.get("name") == "link_gun")
        flange_in_gun = (np.linalg.inv(self.frames["link_gun"])
                         @ self.frames["right_arm_end_link"])
        np.testing.assert_allclose(flange_in_gun[:3, 3], [0., -.03, .075],
                                   atol=1e-12, rtol=0)
        for geometry_kind in ("visual", "collision"):
            with self.subTest(geometry=geometry_kind):
                adapters = [element for element in gun.findall(geometry_kind)
                            if element.get("name") == "huibo_axial_adapter"]
                self.assertEqual(len(adapters), 1)
                adapter = adapters[0]
                cylinder = adapter.find("geometry/cylinder")
                self.assertIsNotNone(cylinder)
                radius = float(cylinder.get("radius"))
                length = float(cylinder.get("length"))
                self.assertAlmostEqual(radius, .022)
                self.assertAlmostEqual(length, .095)
                center = vector(adapter.find("origin"), "xyz")
                np.testing.assert_allclose(center, [0., -.03, .0275], atol=1e-12, rtol=0)
                np.testing.assert_allclose(vector(adapter.find("origin"), "rpy"), 0.,
                                           atol=1e-12, rtol=0)
                ends = center[2] + np.array([-length / 2., length / 2.])
                np.testing.assert_allclose(ends, [-.02, .075], atol=1e-12, rtol=0)
                self.assertAlmostEqual(ends[1], flange_in_gun[2, 3])

    def test_vendor_source_and_generated_joint_geometry_are_not_rewritten(self):
        vendor_path = MODEL_DIR / "huibo_description/urdf/single_right_hand.urdf"
        vendor = ET.parse(vendor_path).getroot()
        with zipfile.ZipFile(MODEL_DIR / "huibo_description_right_hand.zip") as archive:
            original = ET.fromstring(archive.read("huibo_description/urdf/single_right_hand.urdf"))
        self.assertEqual(signature(vendor), signature(original))
        for joint in vendor.findall("joint"):
            if joint.get("name") not in ARM_NAMES:
                continue
            with self.subTest(joint=joint.get("name")):
                generated = deepcopy(self.joints[joint.get("name")])
                # Only the previously documented simulation drive limits differ.
                generated.find("limit").set("effort", "0")
                generated.find("limit").set("velocity", "0")
                self.assertEqual(signature(generated), signature(joint))

    def test_original_z1_tool_and_shared_cylinder_length_are_untouched(self):
        old = ET.parse(MODEL_DIR / "b2_z1_no_stick.urdf").getroot()
        joints = {joint.get("name"): joint for joint in old.findall("joint")}
        self.assertTrue(all(f"joint{i}" in joints for i in range(1, 7)))
        np.testing.assert_allclose(vector(joints["joint_gun_mounter"].find("origin"), "xyz"),
                                   [0., 0., -.2], atol=1e-12, rtol=0)
        old_cfg = ROOT / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/b2_z1_rear_vision_env_cfg.py"
        assignments = {node.targets[0].id: node.value for node in ast.parse(old_cfg.read_text()).body
                       if isinstance(node, ast.Assign) and len(node.targets) == 1
                       and isinstance(node.targets[0], ast.Name)}
        self.assertEqual(ast.literal_eval(assignments["WELDING_GUN_LENGTH_M"]), 1.10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
