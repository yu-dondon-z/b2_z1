"""CPU checks of VD3 mirror geometry against independent URDF forward kinematics."""

import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp"
ASSET = ROOT / "source/b2_z1/b2_z1/assets/b2_z1"
PACKAGE_NAME = "vd3_symmetry_cpu"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(MDP)]
sys.modules[PACKAGE_NAME] = package
spec = importlib.util.spec_from_file_location(PACKAGE_NAME + ".vd3_symmetry", MDP / "vd3_symmetry.py")
symmetry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(symmetry)


JOINT_NAMES = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "joint1", "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    "joint2", "joint3", "joint4", "joint5", "joint6",
)
REFLECTION = np.diag([1., -1., 1., 1.])


def axis_rotation(axis, angle):
    """Rodrigues rotation, independent of the implementation's torch quaternions."""
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * cross + (1. - math.cos(angle)) * (cross @ cross)


def quaternion_matrix(quaternion):
    w, x, y, z = quaternion
    return np.array([
        [1. - 2. * (y*y + z*z), 2. * (x*y - z*w), 2. * (x*z + y*w)],
        [2. * (x*y + z*w), 1. - 2. * (x*x + z*z), 2. * (y*z - x*w)],
        [2. * (x*z - y*w), 2. * (y*z + x*w), 1. - 2. * (x*x + y*y)],
    ])


def matrix_quaternion(matrix):
    """Independent, stable matrix conversion, including nearly pi rotations."""
    values, vectors = np.linalg.eigh(np.array([
        [matrix[0, 0] - matrix[1, 1] - matrix[2, 2], matrix[0, 1] + matrix[1, 0], matrix[0, 2] + matrix[2, 0], matrix[2, 1] - matrix[1, 2]],
        [matrix[0, 1] + matrix[1, 0], matrix[1, 1] - matrix[0, 0] - matrix[2, 2], matrix[1, 2] + matrix[2, 1], matrix[0, 2] - matrix[2, 0]],
        [matrix[0, 2] + matrix[2, 0], matrix[1, 2] + matrix[2, 1], matrix[2, 2] - matrix[0, 0] - matrix[1, 1], matrix[1, 0] - matrix[0, 1]],
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1], np.trace(matrix)],
    ]) / 3.)
    xyzw = vectors[:, np.argmax(values)]
    return xyzw[[3, 0, 1, 2]]


def pose_matrix(position, quaternion):
    result = np.eye(4)
    result[:3, :3] = quaternion_matrix(quaternion)
    result[:3, 3] = position
    return result


class UrdfKinematics:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}

    def joint_transform(self, name, angle=0.):
        joint = self.joints[name]
        origin = joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        result = np.eye(4)
        result[:3, 3] = xyz
        result[:3, :3] = (
            axis_rotation([0., 0., 1.], yaw)
            @ axis_rotation([0., 1., 0.], pitch)
            @ axis_rotation([1., 0., 0.], roll)
        )
        if joint.get("type") != "fixed":
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            result[:3, :3] = result[:3, :3] @ axis_rotation(axis, angle)
        return result

    def arm(self, configuration):
        result = self.joint_transform("joint0")
        for index in range(1, 7):
            result = result @ self.joint_transform("joint" + str(index), configuration["joint" + str(index)])
        return result

    def foot(self, side, configuration):
        result = np.eye(4)
        for kind in ("hip", "thigh", "calf"):
            name = side + "_" + kind + "_joint"
            result = result @ self.joint_transform(name, configuration[name])
        return result @ self.joint_transform(side + "_foot_joint")


def random_observations(batch=8, dtype=torch.float64):
    generator = torch.Generator().manual_seed(553)
    result = torch.randn(batch, 70, generator=generator, dtype=dtype)
    result[:, 6:10] /= torch.linalg.vector_norm(result[:, 6:10], dim=-1, keepdim=True)
    return result


def make_environment(scale=.25):
    """Expose the fields present in IsaacLab 2.2.1's observation/action APIs."""
    default = torch.tensor([
        [.1, -.1, .1, -.1, .8, .8, 1., 1., 0., -1.5, -1.5, -1.5, -1.5, 2.55, -1.6, -1.4, 0., 0.]
    ]).repeat(2, 1)
    robot = types.SimpleNamespace(
        joint_names=list(JOINT_NAMES), device="cpu",
        data=types.SimpleNamespace(default_joint_pos=default, default_joint_vel=torch.zeros_like(default)),
    )
    action = types.SimpleNamespace(
        _joint_names=list(JOINT_NAMES), _scale=scale,
        cfg=types.SimpleNamespace(use_default_offset=True),
    )
    environment = types.SimpleNamespace(
        scene={"robot": robot},
        observation_manager=types.SimpleNamespace(
            active_terms={"policy": [
                "velocity_commands", "target_commands", "base_ang_vel", "projected_gravity",
                "joint_pos_rel", "joint_vel_rel", "actions",
            ]},
            group_obs_term_dim={"policy": [(3,), (7,), (3,), (3,), (18,), (18,), (18,)]},
        ),
        action_manager=types.SimpleNamespace(
            active_terms=["JointPositionAction"], get_term=lambda name: action,
        ),
    )
    environment.unwrapped = environment
    return environment, robot, action


class MirrorGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinematics = UrdfKinematics(ASSET / "b2_z1_vd3_zero_gun.urdf")
        source = UrdfKinematics(ASSET / "b2_z1_gun.urdf")
        cls.tcp_offset = (
            source.joint_transform("joint_mounter")
            @ source.joint_transform("joint_gun_mounter")
            @ source.joint_transform("link_dummy_target_joint")
        )

    def test_joint_contract_matches_named_actor_order_and_leg_only_outputs(self):
        self.assertEqual(tuple(symmetry.JOINT_NAMES), JOINT_NAMES)
        expected = tuple(index for index, name in enumerate(JOINT_NAMES) if name.endswith("_joint"))
        self.assertEqual(tuple(symmetry.LEG_ACTION_INDICES), expected)

    def test_joint_mirror_matches_independent_arm_and_leg_forward_kinematics(self):
        rng = np.random.default_rng(731)
        for _ in range(32):
            configuration = {}
            for name in JOINT_NAMES:
                if "_hip_" in name:
                    value = rng.uniform(-.4, .4)
                elif "_thigh_" in name:
                    value = rng.uniform(.3, 1.5)
                elif "_calf_" in name:
                    value = rng.uniform(-2.4, -.8)
                else:
                    limit = self.kinematics.joints[name].find("limit")
                    value = rng.uniform(float(limit.get("lower")), float(limit.get("upper")))
                configuration[name] = value
            joint_tensor = torch.tensor([[configuration[name] for name in JOINT_NAMES]], dtype=torch.float64)
            mirrored_tensor = symmetry.mirror_joints(joint_tensor)
            mirrored = dict(zip(JOINT_NAMES, mirrored_tensor[0].tolist()))
            flange = self.kinematics.arm(configuration)
            flange_mirrored = self.kinematics.arm(mirrored)
            np.testing.assert_allclose(flange_mirrored, REFLECTION @ flange @ REFLECTION, atol=1.e-13, rtol=0.)
            for left, right in (("FL", "FR"), ("FR", "FL"), ("RL", "RR"), ("RR", "RL")):
                expected = REFLECTION @ self.kinematics.foot(left, configuration) @ REFLECTION
                actual = self.kinematics.foot(right, mirrored)
                # The source RR foot has an additional 87 um Y offset.
                np.testing.assert_allclose(actual[:3, :3], expected[:3, :3], atol=1.e-13, rtol=0.)
                self.assertLessEqual(np.linalg.norm(actual[:3, 3] - expected[:3, 3]), 8.70001e-5)
            tcp = flange @ self.tcp_offset
            tcp_mirrored = flange_mirrored @ self.tcp_offset
            p = torch.tensor(tcp[None, :3, 3], dtype=torch.float64)
            q = torch.tensor(matrix_quaternion(tcp[:3, :3])[None], dtype=torch.float64)
            actual_p, actual_q = symmetry.mirror_tcp_pose(p, q)
            np.testing.assert_allclose(pose_matrix(actual_p[0].numpy(), actual_q[0].numpy()), tcp_mirrored, atol=2.e-13, rtol=0.)

    def test_tcp_mirror_preserves_source_quaternion_sign_and_is_involutive(self):
        observation = random_observations(64)
        p, q = observation[:, 3:6], observation[:, 6:10]
        original_p, original_q = p.clone(), q.clone()
        mirrored_p, mirrored_q = symmetry.mirror_tcp_pose(p, q)
        restored_p, restored_q = symmetry.mirror_tcp_pose(mirrored_p, mirrored_q)
        self.assertTrue(torch.all((q * mirrored_q).sum(-1) >= -1.e-14))
        torch.testing.assert_close(torch.linalg.vector_norm(mirrored_q, dim=-1), torch.ones(64, dtype=q.dtype), atol=1.e-13, rtol=0.)
        torch.testing.assert_close(restored_p, p, atol=2.e-13, rtol=0.)
        torch.testing.assert_close(restored_q, q, atol=2.e-13, rtol=0.)
        torch.testing.assert_close(p, original_p, atol=0., rtol=0.)
        torch.testing.assert_close(q, original_q, atol=0., rtol=0.)
        negative_p, negative_q = symmetry.mirror_tcp_pose(p, -q)
        torch.testing.assert_close(negative_p, mirrored_p, atol=1.e-13, rtol=0.)
        torch.testing.assert_close(negative_q, -mirrored_q, atol=1.e-13, rtol=0.)

    def test_current_fixed_pi_yaw_command_does_not_flip_to_unseen_minus_q(self):
        pitch, yaw = math.pi / 2. - .45, math.pi
        q = torch.tensor([[
            math.cos(pitch / 2.) * math.cos(yaw / 2.),
            -math.sin(pitch / 2.) * math.sin(yaw / 2.),
            math.sin(pitch / 2.) * math.cos(yaw / 2.),
            math.cos(pitch / 2.) * math.sin(yaw / 2.),
        ]], dtype=torch.float64)
        p = torch.tensor([[-1.02, .01, -.3]], dtype=torch.float64)
        _, mirrored_q = symmetry.mirror_tcp_pose(p, q)
        self.assertGreater((mirrored_q * q).sum().item(), .9999999)

    def test_nonzero_defaults_remain_mirror_symmetric(self):
        defaults = {}
        for name in JOINT_NAMES:
            if "_hip_" in name:
                defaults[name] = .1 if name[1] == "L" else -.1
            elif "_thigh_" in name:
                defaults[name] = .8 if name[0] == "F" else 1.
            elif "_calf_" in name:
                defaults[name] = -1.5
            else:
                defaults[name] = {"joint2": 2.55, "joint3": -1.6, "joint4": -1.4}.get(name, 0.)
        values = torch.tensor([[defaults[name] for name in JOINT_NAMES]], dtype=torch.float64)
        torch.testing.assert_close(symmetry.mirror_joints(values), values, atol=0., rtol=0.)


class MirrorObservationTests(unittest.TestCase):
    def test_vector_parity_and_joint_blocks_match_named_physics(self):
        observation = random_observations(2)
        stacked, _ = symmetry.vd3_left_right(obs=observation)
        self.assertEqual(stacked.shape, (4, 70))
        mirrored = stacked[2:]
        torch.testing.assert_close(mirrored[:, :3], observation[:, :3] * torch.tensor([1., -1., -1.]))
        torch.testing.assert_close(mirrored[:, 10:13], observation[:, 10:13] * torch.tensor([-1., 1., -1.]))
        torch.testing.assert_close(mirrored[:, 13:16], observation[:, 13:16] * torch.tensor([1., -1., 1.]))
        for start in (16, 34, 52):
            torch.testing.assert_close(mirrored[:, start:start+18], symmetry.mirror_joints(observation[:, start:start+18]))

    def test_original_batch_actions_inputs_and_dtype_are_preserved(self):
        for dtype in (torch.float32, torch.float64):
            observation = random_observations(3, dtype=dtype)
            action = torch.arange(54, dtype=dtype).reshape(3, 18)
            observation_copy, action_copy = observation.clone(), action.clone()
            stacked_obs, stacked_actions = symmetry.vd3_left_right(obs=observation, actions=action)
            self.assertEqual(stacked_obs.dtype, dtype)
            self.assertEqual(stacked_actions.dtype, dtype)
            self.assertEqual(stacked_actions.shape, (6, 18))
            torch.testing.assert_close(stacked_obs[:3], observation, atol=0., rtol=0.)
            torch.testing.assert_close(stacked_actions[:3], action, atol=0., rtol=0.)
            torch.testing.assert_close(stacked_actions[3:], symmetry.mirror_joints(action), atol=0., rtol=0.)
            torch.testing.assert_close(observation, observation_copy, atol=0., rtol=0.)
            torch.testing.assert_close(action, action_copy, atol=0., rtol=0.)

    def test_double_observation_and_action_mirror_restore_inputs(self):
        observation = random_observations(4)
        action = torch.arange(72, dtype=torch.float64).reshape(4, 18)
        first_obs, first_action = symmetry.vd3_left_right(obs=observation, actions=action)
        second_obs, second_action = symmetry.vd3_left_right(obs=first_obs[4:], actions=first_action[4:])
        torch.testing.assert_close(second_obs[4:], observation, atol=2.e-13, rtol=0.)
        torch.testing.assert_close(second_action[4:], action, atol=0., rtol=0.)

    def test_observation_and_action_only_calls_have_optional_outputs(self):
        observation = random_observations(2)
        action = torch.zeros(2, 18, dtype=torch.float64)
        obs_result, action_result = symmetry.vd3_left_right(obs=observation)
        self.assertEqual(obs_result.shape, (4, 70))
        self.assertIsNone(action_result)
        obs_result, action_result = symmetry.vd3_left_right(actions=action)
        self.assertIsNone(obs_result)
        self.assertEqual(action_result.shape, (4, 18))

    def test_wrong_joint_and_actor_dimensions_are_rejected(self):
        for size in (17, 19):
            with self.subTest(size=size), self.assertRaises(ValueError):
                symmetry.mirror_joints(torch.zeros(2, size))
            with self.subTest(action_size=size), self.assertRaises(ValueError):
                symmetry.vd3_left_right(actions=torch.zeros(2, size))
        for size in (69, 71):
            with self.subTest(obs_size=size), self.assertRaises(ValueError):
                symmetry.vd3_left_right(obs=torch.zeros(2, size))

    def test_symmetry_remains_differentiable_for_consistency_loss(self):
        observation = random_observations(3).requires_grad_()
        action = torch.randn(3, 18, dtype=torch.float64, requires_grad=True)
        stacked_obs, stacked_actions = symmetry.vd3_left_right(obs=observation, actions=action)
        (stacked_obs[3:].square().sum() + stacked_actions[3:].square().sum()).backward()
        self.assertTrue(torch.isfinite(observation.grad).all())
        self.assertTrue(torch.isfinite(action.grad).all())
        self.assertGreater(observation.grad.abs().sum().item(), 0.)
        self.assertGreater(action.grad.abs().sum().item(), 0.)


class EnvironmentContractTests(unittest.TestCase):
    def test_real_api_field_shapes_validate_for_scalar_and_tensor_action_scale(self):
        for scale in (.25, torch.full((2, 18), .25)):
            environment, _, _ = make_environment(scale)
            observation = random_observations(2)
            augmented, _ = symmetry.vd3_left_right(obs=observation, env=environment)
            self.assertEqual(augmented.shape, (4, 70))
            self.assertTrue(environment._vd3_leg_symmetry_validated)

    def test_asymmetric_defaults_and_action_scales_are_rejected(self):
        environment, robot, _ = make_environment()
        robot.data.default_joint_pos[:, 0] += .01
        with self.assertRaises(ValueError):
            symmetry.vd3_left_right(obs=random_observations(2), env=environment)
        environment, _, action = make_environment(torch.full((2, 18), .25))
        action._scale[:, 0] = .5
        with self.assertRaises(ValueError):
            symmetry.vd3_left_right(obs=random_observations(2), env=environment)

    def test_wrong_articulation_or_observation_term_order_is_rejected(self):
        environment, robot, _ = make_environment()
        robot.joint_names[0], robot.joint_names[1] = robot.joint_names[1], robot.joint_names[0]
        with self.assertRaises(ValueError):
            symmetry.vd3_left_right(obs=random_observations(2), env=environment)
        environment, _, _ = make_environment()
        terms = environment.observation_manager.active_terms["policy"]
        terms[0], terms[1] = terms[1], terms[0]
        with self.assertRaises(ValueError):
            symmetry.vd3_left_right(obs=random_observations(2), env=environment)


if __name__ == "__main__":
    unittest.main()
