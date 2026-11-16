"""CPU checks for the actual VD3 shapes, nominal inertia and virtual TCP."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch


PACKAGE = Path(__file__).resolve().parents[1] / "source/b2_z1/b2_z1"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


asset_tools = load("vd3_asset_cpu", PACKAGE / "assets/vd3_zero_gun_asset.py")
runtime = load("vd3_runtime_cpu", PACKAGE / "assets/vd3_runtime_contract.py")
tcp = load("vd3_tcp_cpu", PACKAGE / "tasks/manager_based/b2_z1/mdp/tcp_frame.py")


def fixture(physical_dummy=False):
    names = ["base_link", "link06"] + (["dummy_target"] if physical_dummy else [])
    pos = torch.tensor([[[1., -2., .7], [.3, .4, .9]]], dtype=torch.float64)
    quat = torch.tensor([[[1., 0., 0., 0.], [.8, .0, .0, .6]]], dtype=torch.float64)
    if physical_dummy:
        pos = torch.cat((pos, torch.tensor([[[2., 3., 4.]]], dtype=pos.dtype)), dim=1)
        quat = torch.cat((quat, torch.tensor([[[1., 0., 0., 0.]]], dtype=quat.dtype)), dim=1)
    vel = torch.zeros_like(pos)
    angular = torch.zeros_like(pos)
    data = SimpleNamespace(body_link_pos_w=pos, body_link_quat_w=quat,
                           body_link_lin_vel_w=vel, body_link_ang_vel_w=angular)
    return SimpleNamespace(data=data, find_bodies=lambda name: ([names.index(name)], [name]) if name in names else ([], []))


class ZeroGunAssetTests(unittest.TestCase):
    def setUp(self):
        self.source = ET.parse(asset_tools.SOURCE_URDF).getroot()
        self.generated = ET.parse(asset_tools.VD3_URDF).getroot()

    def test_source_untouched_and_generated_payload_matches_manifest(self):
        self.assertEqual(hashlib.sha256(asset_tools.SOURCE_URDF.read_bytes()).hexdigest(),
                         "14ae3abccfa4d7e5f74f69347ced43f7b1ec5bed728376972260d8b18130c9c6")
        manifest = json.loads(asset_tools.VD3_URDF.with_suffix(".manifest.json").read_text())
        self.assertEqual(hashlib.sha256(asset_tools.VD3_URDF.read_bytes()).hexdigest(), manifest["generated_sha256"])
        with tempfile.TemporaryDirectory(prefix="vd3_asset_cpu_") as directory:
            path = Path(directory) / "test.urdf"
            asset_tools.generate_zero_gun_asset(output_path=path)
            self.assertEqual(path.read_bytes(), asset_tools.VD3_URDF.read_bytes())

    def test_only_gun_load_removed_and_positive_link06_inertia_preserved(self):
        mass = sum(float(x.get("value")) for x in self.generated.findall("link/inertial/mass"))
        self.assertAlmostEqual(mass, 78.9990202, places=8)
        for name in asset_tools.REMOVED_LINKS:
            self.assertIsNone(self.generated.find(f"link[@name='{name}']"))
        for name in asset_tools.FIXED_CHAIN:
            self.assertIsNone(self.generated.find(f"joint[@name='{name}']"))
        self.assertEqual(18, len([x for x in self.generated.findall("joint") if x.get("type") != "fixed"]))
        old = self.source.find("link[@name='link06']/inertial")
        new = self.generated.find("link[@name='link06']/inertial")
        self.assertEqual(ET.tostring(old), ET.tostring(new))
        for link in self.generated.findall("link"):
            old = self.source.find(f"link[@name='{link.get('name')}']/inertial")
            self.assertEqual(ET.tostring(old), ET.tostring(link.find("inertial")))

    def test_shapes_keep_exact_world_placement_under_any_parent_pose(self):
        transform = np.array(asset_tools.fixed_chain_transform(self.source, asset_tools.FIXED_CHAIN[:2]))
        for kind in ("visual", "collision"):
            old = self.source.find(f"link[@name='link_gun']/{kind}")
            new = self.generated.find(f"link[@name='link06']/{kind}[@name='vd3_zero_load_gun_{kind}']")
            self.assertIsNotNone(new)
            expected = transform @ np.array(asset_tools.origin_matrix(old.find("origin")))
            np.testing.assert_allclose(asset_tools.origin_matrix(new.find("origin")), expected, atol=1.e-14)
            self.assertEqual(ET.tostring(old.find("geometry")), ET.tostring(new.find("geometry")))

    def test_tcp_constants_match_preserved_urdf_fixed_chain(self):
        transform = np.array(asset_tools.fixed_chain_transform(self.source))
        np.testing.assert_allclose(tcp.TCP_POS_IN_LINK06, transform[:3, 3], atol=1.e-14)
        quat = torch.tensor(tcp.TCP_QUAT_IN_LINK06, dtype=torch.float64).expand(3, -1)
        rotation = tcp.quat_apply(quat, torch.eye(3, dtype=torch.float64)).T.numpy()
        np.testing.assert_allclose(rotation, transform[:3, :3], atol=1.e-14)


class VirtualTCPTests(unittest.TestCase):
    def test_link06_pose_and_velocity_include_rotated_tcp_offset(self):
        robot = fixture()
        robot.data.body_link_lin_vel_w[:, 1] = torch.tensor([.2, -.1, .3])
        robot.data.body_link_ang_vel_w[:, 1] = torch.tensor([.0, .4, -.3])
        pos, quat = tcp.tcp_pose_w(robot)
        offset = tcp.quat_apply(robot.data.body_link_quat_w[:, 1], pos.new_tensor(tcp.TCP_POS_IN_LINK06).expand_as(pos))
        torch.testing.assert_close(pos, robot.data.body_link_pos_w[:, 1] + offset)
        vel, angular = tcp.tcp_velocity_w(robot)
        torch.testing.assert_close(vel, robot.data.body_link_lin_vel_w[:, 1] + torch.cross(angular, offset, dim=-1))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(quat, dim=-1), torch.ones(1, dtype=quat.dtype)))

    def test_existing_dummy_fixture_preserves_its_actual_pose(self):
        robot = fixture(physical_dummy=True)
        pos, quat = tcp.tcp_pose_w(robot)
        torch.testing.assert_close(pos, robot.data.body_link_pos_w[:, 2])
        torch.testing.assert_close(quat, robot.data.body_link_quat_w[:, 2])

    def test_com_velocity_compatibility_shift_is_correct(self):
        robot = fixture()
        del robot.data.body_link_lin_vel_w
        del robot.data.body_link_ang_vel_w
        robot.data.body_com_pos_w = robot.data.body_link_pos_w + torch.tensor([.1, .2, -.3])
        robot.data.body_com_lin_vel_w = torch.ones_like(robot.data.body_link_pos_w)
        robot.data.body_com_ang_vel_w = torch.ones_like(robot.data.body_link_pos_w) * .5
        lin, ang = tcp.link_origin_velocity_w(robot, 1)
        expected = robot.data.body_com_lin_vel_w[:, 1] + torch.cross(ang, torch.tensor([[-.1, -.2, .3]], dtype=lin.dtype), dim=-1)
        torch.testing.assert_close(lin, expected)

    def test_partial_command_reset_and_metrics_use_virtual_tcp(self):
        from test_rear_arm import make_command
        command = make_command()
        for name, value in list(vars(command).items()):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                setattr(command, name, value.double())
        robot = fixture()
        # Three environments, same link06 state, no physical dummy_target.
        for name, value in list(vars(robot.data).items()):
            setattr(robot.data, name, value.expand(3, -1, -1).clone())
        robot.data.body_pos_w = robot.data.body_link_pos_w
        robot.data.body_state_w = torch.cat((robot.data.body_link_pos_w, robot.data.body_link_quat_w,
                                          torch.zeros_like(robot.data.body_link_pos_w), torch.zeros_like(robot.data.body_link_pos_w)), dim=-1)
        command.robot, command.base_idx, command.body_idx = robot, 0, 1
        command._capture_current_pose_as_start(torch.tensor([1]))
        expected_pos, expected_quat = tcp.tcp_pose_b(robot)
        torch.testing.assert_close(command.pose_start[1, :3], expected_pos[1])
        command.pose_command = torch.cat((expected_pos, expected_quat), -1)
        command._update_metrics()
        torch.testing.assert_close(command.metrics["position_error"], torch.zeros(3, dtype=torch.float64), atol=1.e-12, rtol=0.)
        torch.testing.assert_close(command.metrics["orientation_error"], torch.zeros(3, dtype=torch.float64), atol=1.e-12, rtol=0.)


class TCPRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute production functions without importing Isaac's app-only modules.
        path = PACKAGE / "tasks/manager_based/b2_z1/mdp/rewards.py"
        tree = ast.parse(path.read_text())
        selected = {"smoothness", "position_command_error_l2", "position_command_error_exp",
                    "position_command_error_tanh", "tcp_orientation_command_error"}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in selected]
        namespace = {"torch": torch, "tcp_pose_b": tcp.tcp_pose_b,
                     "ManagerBasedRLEnv": object, "SceneEntityCfg": object, "RigidObject": object}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        cls.functions = namespace

    def test_second_difference_is_zero_for_a_linear_action_ramp(self):
        env = SimpleNamespace(action_manager=SimpleNamespace(action=torch.tensor([[3., 6.]]),
                                                            prev_action=torch.tensor([[2., 4.]])),
                              pre_pre_action=torch.tensor([[1., 2.]]))
        torch.testing.assert_close(self.functions["smoothness"](env), torch.zeros(1))
        env.action_manager.prev_action.zero_()
        self.assertEqual(float(self.functions["smoothness"](env)), 80.)

    def test_position_rewards_track_virtual_tcp_instead_of_link06_origin(self):
        robot = fixture()
        pos, quat = tcp.tcp_pose_b(robot)
        command = torch.cat((pos, quat), dim=-1)
        env = SimpleNamespace(scene={"robot": robot}, command_manager=SimpleNamespace(get_command=lambda name: command))
        asset_cfg = SimpleNamespace(name="robot", body_ids=[1])
        for name, expected in (("position_command_error_l2", 0.), ("position_command_error_exp", 1.),
                               ("position_command_error_tanh", 1.)):
            value = self.functions[name](env, "ee_target", .1, asset_cfg)
            self.assertAlmostEqual(float(value), expected, places=10)

    def test_orientation_error_uses_virtual_tcp_and_quaternion_sign_invariance(self):
        robot = fixture()
        pos, quat = tcp.tcp_pose_b(robot)
        command = torch.cat((pos, -quat), dim=-1)
        env = SimpleNamespace(scene={"robot": robot}, command_manager=SimpleNamespace(get_command=lambda name: command))
        asset_cfg = SimpleNamespace(name="robot", body_ids=[1])
        error = self.functions["tcp_orientation_command_error"](env, "ee_target", asset_cfg)
        self.assertAlmostEqual(float(error), 0., places=7)
        angle = .6
        rotation = quat.new_tensor([[np.cos(angle / 2), np.sin(angle / 2), 0., 0.]])
        command[:, 3:7] = tcp.quat_mul(quat, rotation)
        error = self.functions["tcp_orientation_command_error"](env, "ee_target", asset_cfg)
        self.assertAlmostEqual(float(error), angle, places=10)


def runtime_fixture():
    root = ET.parse(asset_tools.VD3_URDF).getroot()
    parents = {j.find("child").get("link"): j for j in root.findall("joint")}
    def owner(name):
        joint = parents.get(name)
        if joint is not None and joint.get("type") == "fixed" and joint.get("dont_collapse") != "true":
            return owner(joint.find("parent").get("link"))
        return name
    masses = {}
    for link in root.findall("link"):
        name = owner(link.get("name"))
        mass = link.find("inertial/mass")
        masses[name] = masses.get(name, 0.) + (float(mass.get("value")) if mass is not None else 0.)
    names = list(masses)
    mass = torch.tensor([list(masses.values())], dtype=torch.float32).expand(2, -1).clone()
    inertia = torch.eye(3).reshape(1, 1, 9).expand(2, len(names), -1).clone() * .01
    inertia[:, names.index("link06")] = torch.tensor([.0001833, .0000012, .0000005, .0000012, .0001475, .0000001, .0000005, .0000001, .0001468])
    live = mass.clone()
    live[:, names.index("base_link")] += torch.tensor([-1., 3.])
    return SimpleNamespace(body_names=names, num_joints=18,
                           data=SimpleNamespace(default_mass=mass, default_inertia=inertia),
                           root_physx_view=SimpleNamespace(get_masses=lambda: live))


class RuntimeContractTests(unittest.TestCase):
    def test_valid_nominal_asset_and_base_randomization_report(self):
        report = runtime.audit_vd3_runtime_asset(runtime_fixture())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["body_count"], 24)
        self.assertEqual(report["base_mass_randomization_range_kg"], [-1., 3.])
        json.dumps(report)

    def test_unintended_gun_load_and_inertia_are_rejected(self):
        robot = runtime_fixture()
        robot.data.default_mass[:, robot.body_names.index("link06")] += .01
        with self.assertRaisesRegex(RuntimeError, "nominal total mass"):
            runtime.audit_vd3_runtime_asset(robot)
        robot = runtime_fixture()
        robot.data.default_inertia[:, robot.body_names.index("link06"), 0] += .001
        with self.assertRaisesRegex(RuntimeError, "link06 inertia"):
            runtime.audit_vd3_runtime_asset(robot)

    def test_nonbase_mass_randomization_is_rejected(self):
        robot = runtime_fixture()
        robot.root_physx_view.get_masses()[:, robot.body_names.index("link06")] += .001
        with self.assertRaisesRegex(RuntimeError, "other than base_link"):
            runtime.audit_vd3_runtime_asset(robot)


if __name__ == "__main__":
    unittest.main()
