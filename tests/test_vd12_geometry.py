"""CPU/GPU tensor geometry tests; no Isaac/MuJoCo process or actor inference."""

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "source/b2_z1/b2_z1/assets"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ASSETS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geometry = load("vd12_tensor_geometry_test", "vd12_geometry.py")
cpu_geometry = load("vd11_cpu_geometry_reference", "vd11_geometry_audit.py")
URDF = ASSETS / "b2_z1/b2_z1_vd10_hollow_gun.urdf"
NAMES = ("FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
         "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint", "joint1",
         "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
         "joint2", "joint3", "joint4", "joint5", "joint6")
DEFAULT = np.array((.1, -.1, .1, -.1, .8, .8, 1., 1., 0., -1.5, -1.5, -1.5, -1.5,
                    .66647356, -1.25178768, .88531411, 0., 0.))


def matrix_quaternion(matrix):
    # Independent small matrix->wxyz conversion, used only to feed real FK
    # transforms from the existing CPU reference to the batched pose API.
    k = np.array(((matrix[0, 0] - matrix[1, 1] - matrix[2, 2], matrix[1, 0] + matrix[0, 1], matrix[2, 0] + matrix[0, 2], matrix[2, 1] - matrix[1, 2]),
                  (matrix[1, 0] + matrix[0, 1], matrix[1, 1] - matrix[0, 0] - matrix[2, 2], matrix[2, 1] + matrix[1, 2], matrix[0, 2] - matrix[2, 0]),
                  (matrix[2, 0] + matrix[0, 2], matrix[2, 1] + matrix[1, 2], matrix[2, 2] - matrix[0, 0] - matrix[1, 1], matrix[1, 0] - matrix[0, 1]),
                  (matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1], matrix.trace()))) / 3.
    values, vectors = np.linalg.eigh(k)
    xyzw = vectors[:, values.argmax()]
    return xyzw[[3, 0, 1, 2]]


def poses_for(audit, names, q_batch, dtype=torch.float64, device="cpu"):
    positions, quaternions = [], []
    for row in q_batch:
        fk = audit._fk(dict(zip(NAMES, row)))
        positions.append([fk[name][1] for name in names])
        quaternions.append([matrix_quaternion(fk[name][0]) for name in names])
    return (torch.tensor(np.array(positions), dtype=dtype, device=device),
            torch.tensor(np.array(quaternions), dtype=dtype, device=device))


class TensorDistanceTests(unittest.TestCase):
    def test_random_segment_distances_match_cpu_reference(self):
        rng = np.random.default_rng(1412)
        a, b, c, d = rng.normal(size=(4, 300, 3))
        # Exact parallels, zero-length segments, and almost parallels.
        b[:5], d[:5] = a[:5], c[:5]
        d[5:10] = c[5:10] + (b[5:10] - a[5:10])
        d[10:15] = c[10:15] + (b[10:15] - a[10:15]) + 1e-10
        expected = np.array([cpu_geometry._segment_segment_distance(*row) for row in zip(a, b, c, d)])
        actual = geometry.segment_segment_distance(*(torch.tensor(x, dtype=torch.float64) for x in (a, b, c, d)))
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-9, rtol=1e-8)

    def test_random_exact_box_distance_matches_cpu_reference(self):
        rng = np.random.default_rng(1413)
        a, b = rng.normal(size=(2, 300, 3))
        half = rng.uniform(.01, 1., size=(300, 3))
        b[:10] = a[:10]
        b[10:20, 0] = a[10:20, 0]
        expected = np.array([cpu_geometry._segment_box_distance(*row) for row in zip(a, b, half)])
        actual = geometry.segment_box_distance(*(torch.tensor(x, dtype=torch.float64) for x in (a, b, half)))
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-10, rtol=1e-9)

    def test_face_corner_crossing_and_zero_segment(self):
        a = torch.tensor([[-2., 0., 0.], [-2., 2., 2.], [2., 2., 0.], [0., 0., 0.]], dtype=torch.float64)
        b = torch.tensor([[2., 0., 0.], [2., 2., 2.], [2., 2., 0.], [0., 0., 0.]], dtype=torch.float64)
        values = geometry.segment_box_distance(a, b, torch.ones(3, dtype=torch.float64))
        torch.testing.assert_close(values, torch.tensor([0., math.sqrt(2.), math.sqrt(2.), 0.], dtype=torch.float64))


class MeasuredGunTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpu = cpu_geometry.GunRobotGeometryAudit(URDF)
        cls.model = geometry.GunRobotClearance(URDF, dtype=torch.float64)

    def test_nominal_reset_and_actual_bad_pose_match_all_cpu_pairs(self):
        bad = DEFAULT.copy()
        arm = (.0005393492031739082, -.005490569645256075, .008092494641237846,
               .7009869905470734, -.02510656996845894, -.27599692594020525)
        for i, value in enumerate(arm, 1):
            bad[NAMES.index("joint" + str(i))] = value
        batch = [DEFAULT, bad]
        result = self.model(*poses_for(self.cpu, self.model.body_names, batch))
        self.assertTrue(result["input_valid"].all())
        self.assertGreater(result["minimum_clearance_lower_bound_m"][0], .038)
        self.assertLess(result["minimum_clearance_lower_bound_m"][1], 0.)
        for number, q in enumerate(batch):
            reference = self.cpu.audit(NAMES, q)
            expected = np.array([pair["clearance_lower_bound_m"] for pair in reference["pair_results"]]) - self.model.numerical_guard_m
            np.testing.assert_allclose(result["clearance_lower_bound_m"][number].numpy(), expected, atol=1e-10, rtol=1e-9)
            np.testing.assert_allclose(result["tube_endpoints_w"][number].numpy(), reference["tube_endpoints_root_m"], atol=1e-10)
        self.assertEqual(self.cpu.audit(NAMES, bad)["status"], "invalid")

    def test_random_pose_pairs_match_cpu_and_float32(self):
        rng = np.random.default_rng(1414)
        batch = DEFAULT + rng.normal(scale=.15, size=(24, 18))
        positions, quaternions = poses_for(self.cpu, self.model.body_names, batch)
        result = self.model(positions, quaternions)
        model32 = geometry.GunRobotClearance(URDF, dtype=torch.float32)
        result32 = model32(positions.float(), quaternions.float())
        for number, q in enumerate(batch):
            reference = self.cpu.audit(NAMES, q)
            expected = np.array([pair["clearance_lower_bound_m"] for pair in reference["pair_results"]]) - self.model.numerical_guard_m
            np.testing.assert_allclose(result["clearance_lower_bound_m"][number].numpy(), expected, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(result32["clearance_lower_bound_m"].double(), result["clearance_lower_bound_m"], atol=2e-6, rtol=1e-5)

    def test_fixed_joint_merging_retains_all_source_colliders(self):
        # Actual physics bodies comprise the root and all non-fixed children.
        import xml.etree.ElementTree as ET
        root = ET.parse(URDF).getroot()
        actual = ["base_link"] + [joint.find("child").get("link") for joint in root.findall("joint") if joint.get("type") != "fixed"]
        merged = geometry.GunRobotClearance(URDF, actual, dtype=torch.float64)
        self.assertEqual(merged.num_pairs, self.model.num_pairs)
        self.assertEqual(set(merged.pair_names), set(self.model.pair_names))
        self.assertTrue(any(item["source_link"] != item["carrier_body"] for item in merged.metadata["pairs"]))
        q = [DEFAULT]
        full_result = self.model(*poses_for(self.cpu, self.model.body_names, q))
        merged_result = merged(*poses_for(self.cpu, actual, q))
        torch.testing.assert_close(merged_result["clearance_lower_bound_m"], full_result["clearance_lower_bound_m"], atol=1e-10, rtol=1e-9)

    def test_scaled_and_negative_quaternions_are_safe_normalized(self):
        p, q = poses_for(self.cpu, self.model.body_names, [DEFAULT])
        expected = self.model(p, q)
        for scale in (-3., 1e12, 1e-4):
            result = self.model(p, q * scale)
            self.assertTrue(result["input_valid"].all())
            torch.testing.assert_close(result["clearance_lower_bound_m"], expected["clearance_lower_bound_m"])

    def test_common_world_transform_does_not_change_clearance(self):
        p, q = poses_for(self.cpu, self.model.body_names, [DEFAULT])
        expected = self.model(p, q)
        angle = .7
        rotation = torch.tensor([math.cos(angle / 2.), 0., 0., math.sin(angle / 2.)], dtype=p.dtype).expand_as(q)
        shifted_p = geometry._quat_apply(rotation, p) + torch.tensor([4., -2., .5], dtype=p.dtype)
        # Left multiplication by a yaw quaternion, written independently.
        w, x, y, z = q.unbind(-1)
        c, s = math.cos(angle / 2.), math.sin(angle / 2.)
        rotated_q = torch.stack((c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w), -1)
        result = self.model(shifted_p, rotated_q)
        torch.testing.assert_close(result["clearance_lower_bound_m"], expected["clearance_lower_bound_m"], atol=1e-10, rtol=1e-9)

    def test_invalid_used_poses_fail_closed_without_nan(self):
        p, q = poses_for(self.cpu, self.model.body_names, [DEFAULT] * 4)
        q[0, self.model.tube_body_index] = 0.
        q[1, self.model.tube_body_index, 0] = float("nan")
        p[2, self.model.tube_body_index, 0] = float("inf")
        q[3, self.model.tube_body_index] = 1e-30
        result = self.model(p, q)
        self.assertFalse(result["input_valid"].any())
        self.assertTrue(torch.isfinite(result["clearance_lower_bound_m"]).all())
        self.assertTrue(torch.isfinite(result["tube_endpoints_w"]).all())
        self.assertTrue((result["minimum_pair_index"] == -1).all())
        self.assertTrue((result["minimum_clearance_lower_bound_m"] < 0.).all())

    def test_metadata_is_json_and_scope_is_explicit(self):
        json.dumps(self.model.metadata, allow_nan=False)
        self.assertEqual(self.model.tube_radius_m, .028)
        self.assertEqual(self.model.tube_length_m, 1.1)
        self.assertEqual(self.model.num_pairs, 44)
        self.assertNotIn("link06", self.model.pair_source_links)
        self.assertIn("link04", self.model.metadata["links_without_authored_collision"])
        self.assertEqual([shape["source_link"] for shape in self.model.metadata["excluded_collisions"]], ["link06"])
        self.assertFalse(self.model.metadata["actor_input"])

    def test_shape_and_missing_movable_body_errors(self):
        p, q = poses_for(self.cpu, self.model.body_names, [DEFAULT])
        with self.assertRaises(ValueError):
            self.model(p[0], q[0])
        with self.assertRaises(ValueError):
            self.model(p.float(), q.float())
        with self.assertRaises(ValueError):
            geometry.GunRobotClearance(URDF, [name for name in self.model.body_names if name != "link02"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_large_batch_matches_cpu_without_simulator(self):
        # Tensor kernels only; no CUDA simulator/renderer is launched.
        p, q = poses_for(self.cpu, self.model.body_names, [DEFAULT], dtype=torch.float32)
        cpu_model = geometry.GunRobotClearance(URDF, dtype=torch.float32)
        expected = cpu_model(p, q)
        gpu_model = geometry.GunRobotClearance(URDF, dtype=torch.float32, device="cuda")
        result = gpu_model(p.cuda().expand(2048, -1, -1), q.cuda().expand(2048, -1, -1))
        torch.testing.assert_close(result["clearance_lower_bound_m"].cpu(), expected["clearance_lower_bound_m"].expand(2048, -1), atol=2e-6, rtol=1e-5)


class ConservativeShapeTests(unittest.TestCase):
    def test_capsule_cap_keepout_is_not_claimed_real_collision(self):
        directory = tempfile.TemporaryDirectory(prefix="vd12_geometry_test_")
        self.addCleanup(directory.cleanup)
        source = Path(directory.name) / "fixture.urdf"
        source.write_text('<robot name="test"><link name="base"><collision><origin xyz="0 0 1.05"/>'
                          '<geometry><box size=".02 .02 .02"/></geometry></collision></link><link name="link06">'
                          '<collision name="vd10_hollow_gun_collision"><geometry><cylinder radius=".1" length="2"/>'
                          '</geometry></collision></link><joint name="joint" type="revolute"><parent link="base"/>'
                          '<child link="link06"/><axis xyz="0 1 0"/></joint></robot>', encoding="utf-8")
        model = geometry.GunRobotClearance(source, ["base", "link06"], dtype=torch.float64)
        p = torch.zeros((1, 2, 3), dtype=torch.float64)
        q = torch.zeros((1, 2, 4), dtype=torch.float64)
        q[..., 0] = 1.
        result = model(p, q)
        self.assertLess(result["minimum_clearance_lower_bound_m"], 0.)
        self.assertNotIn("confirmed_invalid", result)
        self.assertNotIn("penetration_depth_m", result)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
