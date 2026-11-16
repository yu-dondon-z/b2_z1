"""Pure CPU regression tests; no simulator import, policy inference or training."""

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "source/b2_z1/b2_z1/assets"
spec = importlib.util.spec_from_file_location("vd11_geometry_audit_cpu", ASSETS / "vd11_geometry_audit.py")
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)
URDF = ASSETS / "b2_z1/b2_z1_vd10_hollow_gun.urdf"
JOINT_NAMES = ("FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
               "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
               "joint1", "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
               "joint2", "joint3", "joint4", "joint5", "joint6")
DEFAULT = np.array((.1, -.1, .1, -.1, .8, .8, 1., 1., 0., -1.5, -1.5, -1.5, -1.5,
                    .66647356, -1.25178768, .88531411, 0., 0.))


class GeometryMathTests(unittest.TestCase):
    def test_segment_distances_parallel_crossing_and_degenerate(self):
        f = geometry._segment_segment_distance
        self.assertAlmostEqual(f(np.array((0., 0., 0.)), np.array((1., 0., 0.)),
                                 np.array((0., 2., 0.)), np.array((1., 2., 0.))), 2.)
        self.assertAlmostEqual(f(np.array((0., 0., 0.)), np.array((1., 0., 0.)),
                                 np.array((.5, -1., 0.)), np.array((.5, 1., 0.))), 0.)
        self.assertAlmostEqual(f(np.zeros(3), np.zeros(3), np.ones(3), np.ones(3)), math.sqrt(3.))

    def test_segment_distance_interior_regression(self):
        # An independent dense grid and analytic constrained minimizer agree
        # here; a generic finite-difference optimizer can stop prematurely.
        points = np.array(((.12097449717377555, -.3550896193953272, -1.39301910909423),
                           (-.8013357792863058, .134873693786895, 2.01029405800062),
                           (1.0291942399473666, -.2356166873234418, -.6243913165481009),
                           (.6464165494798296, -.7587372554052102, 1.5646560640279579)))
        self.assertAlmostEqual(geometry._segment_segment_distance(*points), 1.0787590020138138)

    def test_exact_box_distance_corner_and_through(self):
        size = np.ones(3)
        self.assertAlmostEqual(geometry._segment_box_distance(np.array((-2., 0., 0.)),
                                                              np.array((2., 0., 0.)), size), 0.)
        self.assertAlmostEqual(geometry._segment_box_distance(np.array((-2., 2., 2.)),
                                                              np.array((2., 2., 2.)), size), math.sqrt(2.))
        self.assertAlmostEqual(geometry._segment_box_distance(np.array((2., 2., 0.)),
                                                              np.array((3., 3., 0.)), size), math.sqrt(2.))

    def test_cylinder_centreline_intersection_uses_finite_end_faces(self):
        inside = geometry._segment_round_interval(np.array((-2., 0., 0.)), np.array((2., 0., 0.)), .5, 1., 1e-6)
        self.assertIsNotNone(inside)
        self.assertAlmostEqual((inside[1] - inside[0]) * 4., 1. - 2e-6)
        self.assertIsNone(geometry._segment_round_interval(np.array((-2., 0., 2.)),
                                                          np.array((2., 0., 2.)), .5, 1., 1e-6))


class MeasuredGunAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = geometry.GunRobotGeometryAudit(URDF)

    def test_nominal_reset_is_certifiably_safe(self):
        report = self.audit.audit(JOINT_NAMES, DEFAULT)
        self.assertEqual(report["status"], "safe")
        self.assertFalse(report["confirmed_invalid"])
        self.assertAlmostEqual(report["minimum_clearance_lower_bound_m"], .03883080807284582, places=8)
        self.assertEqual(report["confirmed_intersections"], [])
        self.assertEqual(report["uncertain_pairs"], [])
        self.assertTrue(report["evaluation_only"])
        self.assertFalse(report["actor_input"])

    def test_actual_two_second_pose_proves_gun_through_base(self):
        # Actual latest VD11 viewer, physics t=2.005 s.  FK is base relative,
        # therefore no root world pose or simulator contact force is needed.
        positions = DEFAULT.copy()
        arm = (.0005393492031739082, -.005490569645256075, .008092494641237846,
               .7009869905470734, -.02510656996845894, -.27599692594020525)
        for number, position in enumerate(arm, 1):
            positions[JOINT_NAMES.index("joint" + str(number))] = position
        report = self.audit.audit(JOINT_NAMES, positions)
        self.assertEqual(report["status"], "invalid")
        self.assertTrue(report["confirmed_invalid"])
        base = next(item for item in report["confirmed_intersections"] if item["other_link"] == "base_link")
        self.assertGreater(base["centerline_inside_length_m"], .038)
        self.assertLess(base["centerline_inside_length_m"], .039)
        self.assertNotIn("penetration_depth_m", base)

    def test_joint_name_order_not_position_order_controls_fk(self):
        permutation = np.arange(len(JOINT_NAMES))[::-1]
        report = self.audit.audit([JOINT_NAMES[i] for i in permutation], DEFAULT[permutation])
        original = self.audit.audit(JOINT_NAMES, DEFAULT)
        np.testing.assert_allclose(report["tube_endpoints_root_m"], original["tube_endpoints_root_m"])

    def test_metadata_and_reports_are_strict_json(self):
        json.dumps(self.audit.metadata, allow_nan=False)
        json.dumps(self.audit.audit(JOINT_NAMES, DEFAULT), allow_nan=False)
        self.assertEqual(self.audit.metadata["tube"]["length_m"], 1.1)
        self.assertEqual(self.audit.metadata["tube"]["outer_diameter_m"], .056)
        self.assertEqual(self.audit.metadata["joint_count"], 18)
        self.assertIn("link04", self.audit.metadata["links_without_authored_collision"])

    def test_batch_agrees_and_does_not_change_input(self):
        batch = np.stack((DEFAULT, DEFAULT))
        before = batch.copy()
        reports = self.audit.audit_batch(JOINT_NAMES, batch)
        self.assertEqual([report["status"] for report in reports], ["safe", "safe"])
        np.testing.assert_array_equal(batch, before)
        self.assertEqual(self.audit.audit_batch(JOINT_NAMES, np.empty((0, 18))), [])

    def test_missing_duplicate_nonfinite_or_wrong_shaped_joints_rejected(self):
        for names, values in ((JOINT_NAMES[:-1], DEFAULT[:-1]),
                              (JOINT_NAMES[:-1] + (JOINT_NAMES[0],), DEFAULT),
                              (JOINT_NAMES, np.full(18, np.nan)), (JOINT_NAMES, DEFAULT[None])):
            with self.assertRaises(ValueError):
                self.audit.audit(names, values)
        with self.assertRaises(ValueError):
            self.audit.audit_batch(JOINT_NAMES, DEFAULT)


class ConservativeCertificateTests(unittest.TestCase):
    def audit_for(self, other_shape, other_position="0 0 0"):
        directory = tempfile.TemporaryDirectory(prefix="vd11_geometry_test_")
        self.addCleanup(directory.cleanup)
        source = Path(directory.name) / "fixture.urdf"
        source.write_text('<robot name="test"><link name="base"><collision><origin xyz="'
                          + other_position + '"/><geometry>' + other_shape
                          + '</geometry></collision></link><link name="gun">'
                          '<collision name="vd10_hollow_gun_collision"><geometry>'
                          '<cylinder radius="0.1" length="2"/></geometry></collision></link>'
                          '<joint name="fixed" type="fixed"><parent link="base"/>'
                          '<child link="gun"/></joint></robot>', encoding="utf-8")
        return geometry.GunRobotGeometryAudit(source)

    def test_actual_box_crossing_invalid(self):
        report = self.audit_for('<box size=".5 .5 .5"/>').audit([], [])
        self.assertEqual(report["status"], "invalid")
        self.assertAlmostEqual(report["confirmed_intersections"][0]["centerline_inside_length_m"], .5 - 2e-6)

    def test_capsule_end_overlap_is_uncertain_not_fake_collision(self):
        # The true cylinder stops at z=1.  Its enclosing capsule's hemisphere
        # overlaps this box, but no true collision has been proved.
        report = self.audit_for('<box size=".02 .02 .02"/>', "0 0 1.05").audit([], [])
        self.assertEqual(report["status"], "uncertain")
        self.assertFalse(report["confirmed_invalid"])
        self.assertLess(report["minimum_clearance_lower_bound_m"], 0.)
        self.assertEqual(report["confirmed_intersections"], [])

    def test_cylinder_and_sphere_crossing_invalid(self):
        for shape in ('<cylinder radius=".3" length=".4"/>', '<sphere radius=".25"/>'):
            self.assertEqual(self.audit_for(shape).audit([], [])["status"], "invalid")

    def test_far_shapes_safe(self):
        for shape in ('<box size=".5 .5 .5"/>', '<cylinder radius=".3" length=".4"/>', '<sphere radius=".25"/>'):
            self.assertEqual(self.audit_for(shape, "3 0 0").audit([], [])["status"], "safe")

    def test_unsupported_mesh_cannot_be_silently_safe(self):
        report = self.audit_for('<mesh filename="does_not_need_to_exist.stl"/>', "3 0 0").audit([], [])
        self.assertEqual(report["status"], "uncertain")
        self.assertIsNone(report["minimum_clearance_lower_bound_m"])


if __name__ == "__main__":
    unittest.main()
