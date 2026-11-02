"""Independent CPU checks of the encoder-only TCP model and fixed tool chain."""

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/b2_z1/b2_z1"
URDF = SOURCE / "assets/b2_z1/b2_z1_no_stick.urdf"
HUIBO_URDF = SOURCE / "assets/b2_z1/huibo_description/urdf/single_right_hand.urdf"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kinematics = load("encoder_tool_fk", SOURCE / "tasks/manager_based/b2_z1/mdp/commands/dual_vision_kinematics.py")
reference = load("reference_teacher_fk", SOURCE / "rsl_rl/modules/pose_reference_teacher.py")


class EncoderKinematicsTests(unittest.TestCase):
    def setUp(self):
        self.model = kinematics.URDFToolKinematics(URDF, "cpu")
        self.teacher = reference.PoseReferenceTeacher(torch.nn.Linear(70, 18), URDF,
            [0, 1, 2, 3, 4, 5], [0., 2.55, -1.6, -1.4, 0., 0.])

    def test_matches_existing_fk_over_sixty_four_distinct_encoder_samples(self):
        generator = torch.Generator().manual_seed(1526)
        q = torch.tensor([0., 2.2, -1.7, -.5, 0., 0.]) + torch.rand(64, 6, generator=generator) - .5
        expected, _, _ = self.teacher.fk(q)
        torch.testing.assert_close(self.model.position(q), expected, atol=2e-7, rtol=2e-7)

    def test_double_precision_and_encoder_delta_match_reference(self):
        q = torch.tensor([[.1, 2.3, -1.8, -.5, .05, .03]], dtype=torch.float64)
        dq = torch.tensor([[.03, -.01, .02, .01, -.02, .03]], dtype=torch.float64)
        teacher = self.teacher.double()
        p0, _, _ = teacher.fk(q)
        p1, _, _ = teacher.fk(q+dq)
        torch.testing.assert_close(self.model.position(q+dq)-self.model.position(q), p1-p0,
                                   atol=2e-12, rtol=2e-12)
        self.assertGreater(torch.linalg.vector_norm(p1-p0), .005)

    def test_fixed_dummy_target_offset_is_included_in_measured_joint_fk(self):
        root = ET.parse(URDF).getroot()
        dummy_joint = next(joint for joint in root.findall("joint")
                           if joint.find("child").get("link") == "dummy_target")
        xyz = [float(value) for value in dummy_joint.find("origin").get("xyz").split()]
        xyz[2] += .08
        dummy_joint.find("origin").set("xyz", " ".join(map(str, xyz)))
        with tempfile.TemporaryDirectory() as directory:
            modified = Path(directory) / "tool_offset.urdf"
            ET.ElementTree(root).write(modified)
            shifted = kinematics.URDFToolKinematics(modified, "cpu")
            q = torch.tensor([[.1, 2.4, -2., -.3, .05, .03]])
            difference = shifted.position(q) - self.model.position(q)
            self.assertAlmostEqual(float(torch.linalg.vector_norm(difference)), .08, places=6)

    def test_joint_columns_are_explicit_and_not_action_or_articulation_order(self):
        self.assertEqual(self.model.joint_names, ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"))
        self.assertIn("joint_gun_mounter", self.model.chain_joint_names)
        q = torch.tensor([[.1, 2.3, -1.8, -.5, .05, .03]])
        reversed_q = q.flip(1)
        self.assertGreater(float(torch.linalg.vector_norm(self.model.position(q) - self.model.position(reversed_q))), .1)

    def test_bad_encoder_shape_or_nonfinite_angles_are_rejected(self):
        for q in (torch.zeros(6), torch.zeros(2, 18), torch.zeros(1, 6, dtype=torch.int64),
                  torch.full((1, 6), float("nan"))):
            with self.subTest(shape=q.shape), self.assertRaises(ValueError):
                self.model.position(q)

    def test_unchanged_encoders_have_zero_model_delta(self):
        q = torch.tensor([[0., 2.7673, -2.4606, -.3067, 0., 0.]])
        torch.testing.assert_close(self.model.position(q) - self.model.position(q.clone()), torch.zeros(1, 3))


class HuiboEncoderKinematicsTests(unittest.TestCase):
    """Exercise the supplied seven-axis arm, not a renamed Z1 fixture."""

    names = tuple(f"right_joint{i}" for i in range(1, 8))

    def model(self, names=None):
        return kinematics.URDFToolKinematics(
            HUIBO_URDF, base_link="right_arm_base_link", tcp_link="right_arm_end_link",
            joint_names=self.names if names is None else names)

    def test_actual_zero_pose_matches_cad_link_lengths_and_mount_direction(self):
        # At nominal quarter-turn origins the arm extends along base -Y:
        # shoulder .216 + upper arm .310 + forearm .280 + flange .068 m.
        # The original CAD URDF uses rounded pi and joint4 yaw=3.1459, so
        # retain a 2 mm tolerance for those explicitly supplied rotations.
        actual = self.model().position(torch.zeros(1, 7))
        torch.testing.assert_close(actual, torch.tensor([[0., -.874, 0.]]), atol=.002, rtol=0)

    def test_seventh_encoder_and_explicit_column_order_are_used(self):
        q = torch.tensor([[.1, -.2, .3, .4, -.1, .2, .3]])
        original = self.model().position(q)
        reordered = self.model(self.names[::-1]).position(q.flip(1))
        torch.testing.assert_close(reordered, original, atol=1e-7, rtol=0)
        q[:, 6] += .2
        self.assertGreater(float(torch.linalg.vector_norm(self.model().position(q) - original)), .01)

    def test_all_finite_in_limit_samples_stay_within_cad_shoulder_reach(self):
        root = ET.parse(HUIBO_URDF).getroot()
        joints = {joint.get("name"): joint for joint in root.findall("joint")}
        lower = torch.tensor([float(joints[name].find("limit").get("lower")) for name in self.names])
        upper = torch.tensor([float(joints[name].find("limit").get("upper")) for name in self.names])
        q = lower + (upper-lower) * torch.rand(64, 7, generator=torch.Generator().manual_seed(711))
        positions = self.model().position(q)
        shoulder_distances = torch.linalg.vector_norm(positions - torch.tensor([0., -.216, 0.]), dim=1)
        self.assertEqual(positions.shape, (64, 3))
        self.assertTrue(torch.isfinite(positions).all())
        self.assertLessEqual(float(shoulder_distances.max()), .310 + .280 + .068 + 1e-6)

    def test_joint_configuration_and_six_column_input_are_rejected(self):
        for names in ((), self.names[:-1], self.names + ("missing_joint",),
                      self.names[:-1] + (self.names[0],), self.names[:-1] + (None,)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.model(names)
        with self.assertRaises(ValueError):
            self.model().position(torch.zeros(1, 6))

    def test_live_command_propagates_seven_measured_encoders_without_reading_tcp_pose(self):
        methods = load("huibo_command_fixtures", ROOT / "scripts/vision/test_dual_vision_methods.py")
        command = methods.live_fixture()
        model = self.model()
        capture_q = torch.tensor([[.1, -.2, .3, .4, -.1, .2, .3]])
        current_q = capture_q + torch.tensor([[.01, .02, -.03, .04, -.02, .01, .03]])
        visual_bias = torch.tensor([.01, -.015, .005])
        command._tcp_kinematics = model
        command._dual_arm_joint_ids = list(range(7))
        command.robot.data.joint_pos = current_q
        capture_fk = model.position(capture_q)[0]
        command._source_snapshot["rear"]["encoder_tcp_fk_b"] = capture_fk.tolist()
        command._observations["rear"]["tcp_measured_b"] = (capture_fk + visual_bias).tolist()
        _, measured, _ = command._current_geometry("rear")
        torch.testing.assert_close(measured, model.position(current_q)[0] + visual_bias, atol=1e-7, rtol=0)
        self.assertEqual(command.robot.data.body_pos_w.shape[1], 1)

    def test_policy_status_dimension_comes_from_actual_observation_manager(self):
        methods = load("huibo_status_fixtures", ROOT / "scripts/vision/test_dual_vision_methods.py")
        cls = methods.production_class("dual_vision_command.py", "DualVisionTrajectoryCommand",
                                       ["_policy_observation_count"])
        command = cls()
        command.env = SimpleNamespace(observation_manager=SimpleNamespace(group_obs_dim={"policy": (73,)}))
        self.assertEqual(command._policy_observation_count(), 73)
        command.env.observation_manager.group_obs_dim["policy"] = [(3,), (7,), (3,), (3,), (19,), (19,), (19,)]
        self.assertEqual(command._policy_observation_count(), 73)


if __name__ == "__main__":
    unittest.main(verbosity=2)
