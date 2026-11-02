"""Run actual VA16 command methods on CPU, replacing only Isaac integration."""

import ast
from collections import OrderedDict
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
COMMANDS = ROOT / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/commands"
spec = importlib.util.spec_from_file_location("dual_contract_methods", COMMANDS / "dual_vision_contract.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
geometry_spec = importlib.util.spec_from_file_location("path_geometry_methods", COMMANDS / "path_guidance.py")
geometry = importlib.util.module_from_spec(geometry_spec)
geometry_spec.loader.exec_module(geometry)
fk_spec = importlib.util.spec_from_file_location("tool_encoder_method_fk", COMMANDS / "dual_vision_kinematics.py")
fk_module = importlib.util.module_from_spec(fk_spec)
fk_spec.loader.exec_module(fk_module)


def rotate(quaternion, vector):
    xyz = quaternion[:, 1:]
    cross = torch.cross(xyz, vector, dim=-1)
    return vector + 2 * (quaternion[:, :1] * cross + torch.cross(xyz, cross, dim=-1))


def inverse(quaternion, vector):
    conjugate = quaternion.clone()
    conjugate[:, 1:] *= -1
    return rotate(conjugate, vector)


def production_class(filename, class_name, methods, parent=object):
    tree = ast.parse((COMMANDS / filename).read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    if len(selected) != len(methods):
        raise AssertionError(f"missing production methods: {methods}")
    node = ast.ClassDef(name="Production", bases=[ast.Name(id="Parent", ctx=ast.Load())],
                        keywords=[], body=selected, decorator_list=[])
    scope = dict(torch=torch, math=math, json=json, os=os, Path=Path, time=time,
                 OrderedDict=OrderedDict, Parent=parent, quat_apply=rotate,
                 quat_apply_inverse=inverse, wrap_to_pi=lambda angle: torch.atan2(torch.sin(angle), torch.cos(angle)),
                 local_path_geometry=geometry.local_path_geometry, rear_target=contract.rear_target,
                 validate_stream=contract.validate_stream, validate_capture=contract.validate_capture,
                 finite_array=contract.finite_array, SCHEMA=contract.SCHEMA)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(COMMANDS / filename), "exec"), scope)
    return scope["Production"]


def synthetic_fixture():
    parent = production_class("vision_trajectory_command.py", "RearVisionTrajectoryCommand", [
        "_update_synthetic_command", "_update_straight_arc_synthetic_command",
        "_velocity_start_blend", "_constant_curvature_xy_heading"])
    cls = production_class("dual_vision_command.py", "DualVisionTrajectoryCommand",
                           ["_update_synthetic_command"], parent)
    obj = cls()
    obj.device, obj.base_idx, obj.num_envs = "cpu", 0, 3
    obj.cfg = NS(synthetic_world_fixed_path=True, synthetic_path_family="straight_arc",
                 path_speed_mps=.20, path_lookahead_m=.8, cross_track_gain=1.,
                 path_heading_filter_alpha=.1, path_heading_gain=1., max_yaw_rate_rps=.4,
                 command_transition_time_s=.2, velocity_filter_time_constant_s=.15,
                 base_motion_delay_s=0., rear_work_x_range_m=(-1., -.55),
                 synthetic_ground_height_w_m=0.)
    obj.env = NS(step_dt=.02, scene=NS(env_origins=torch.zeros(3, 3)))
    heading = torch.tensor([0., .6, -.6])
    quaternion = torch.zeros(3, 4)
    quaternion[:, 0], quaternion[:, 3] = torch.cos(heading/2), torch.sin(heading/2)
    position = torch.tensor([[0., 0., .5]]).expand(3, -1).clone()
    obj.robot = NS(data=NS(body_pos_w=position.unsqueeze(1),
                           body_state_w=torch.cat((position, quaternion), dim=1).unsqueeze(1),
                           heading_w=heading))
    obj._synthetic_path_origin_xy_w = torch.zeros(3, 2)
    obj._synthetic_path_heading0_w = torch.zeros(3)
    obj._synthetic_curvature = torch.zeros(3)
    obj.is_straight_path = torch.ones(3, dtype=torch.bool)
    obj._synthetic_speed = torch.full((3,), .2)
    obj._synthetic_tool_height = torch.full((3,), .09)
    for name in ("_filtered_path_heading", "_path_lateral_error_b", "_path_heading_error_b",
                 "_command_transition_elapsed"):
        setattr(obj, name, torch.zeros(3))
    for name in ("base_velocity_command_b", "_filtered_velocity_command_b",
                 "_command_transition_start_velocity_b", "_command_transition_start_position_b"):
        setattr(obj, name, torch.zeros(3, 3))
    obj._vision_valid = torch.zeros(3, dtype=torch.bool)
    obj._set_pose_command_b = lambda target: setattr(obj, "last_target_b", target.clone())
    return obj


def live_fixture():
    parent = production_class("vision_trajectory_command.py", "RearVisionTrajectoryCommand",
                              ["_filter_file_velocity_command"])
    cls = production_class("dual_vision_command.py", "DualVisionTrajectoryCommand", [
        "_current_geometry", "_update_dual_command", "_ingest_latest_vision_file",
        "_export_latest_camera_frame", "_atomic_json", "_prepare_visual_tool"], parent)
    obj = cls()
    obj.device, obj.base_idx = "cpu", 0
    obj.cfg = NS(minimum_path_points=6, minimum_forward_tangent=.35,
                 cross_track_gain=1., path_lookahead_m=.8, path_heading_filter_alpha=.2,
                 path_heading_gain=1., max_yaw_rate_rps=.4, stale_timeout_s=1.,
                 rear_work_x_range_m=(-1., -.55), tool_height_m=.09,
                 rear_feedback_gain=.35, rear_feedback_max_correction_m=.03,
                 prepare_position_b=(-.775, 0., -.48), visual_start_tolerance_m=.10,
                 file_stop_epsilon=.01, command_transition_time_s=.2,
                 velocity_filter_time_constant_s=.15, path_speed_mps=.20,
                 camera_export_period_steps=2, camera_ring_slots=8, capture_history_frames=128,
                 front_camera_name="front_rgb", rear_camera_name="rear_d435i")
    obj.env = NS(step_dt=.02, common_step_counter=2)
    position, quaternion = torch.tensor([[0., 0., .5]]), torch.tensor([[1., 0., 0., 0.]])
    obj.robot = NS(data=NS(body_pos_w=position.unsqueeze(1),
                           body_state_w=torch.cat((position, quaternion), dim=1).unsqueeze(1),
                           projected_gravity_b=torch.tensor([[0., 0., -1.]])))
    capture = time.monotonic()
    snapshot = dict(sequence=8, capture_generation=3, capture_timestamp=capture,
                    base_position_w=position[0].tolist(), base_quaternion_w=quaternion[0].tolist())
    front_path = np.column_stack((np.linspace(.3, 3, 40), np.full(40, .2), np.full(40, -.5)))
    rear_path = np.column_stack((np.linspace(-1, -.55, 30), np.zeros(30), np.full(30, -.5)))
    obj._observations = {
        "front": dict(snapshot, points_b=front_path),
        "rear": dict(snapshot, points_b=rear_path, ground_plane_b=[0., 0., 1., .5],
                     tcp_measured_b=[-.775, .04, -.41]),
    }
    obj._source_snapshot = dict(front=dict(snapshot), rear=dict(snapshot))
    obj._captures = dict(front=OrderedDict({8: dict(snapshot)}), rear=OrderedDict({8: dict(snapshot)}))
    obj._consumed = dict(front=-1, rear=-1)
    obj._camera_frames = dict(front=-1, rear=-1)
    obj._reject_reasons, obj._vision_generation = {}, 3
    obj._last_packet_mtime, obj._camera_export_sequence = -1, 0
    obj._visual_start_elapsed = 10.
    obj._visual_started = True
    obj._ground_observation = None
    obj.pose_command = torch.zeros(1, 7)
    obj.base_velocity_command_b = torch.zeros(1, 3)
    obj._filtered_velocity_command_b = torch.zeros(1, 3)
    obj._filtered_path_heading = torch.zeros(1)
    obj._path_lateral_error_b = torch.zeros(1)
    obj._path_heading_error_b = torch.zeros(1)
    obj._vision_valid = torch.zeros(1, dtype=torch.bool)
    obj._set_file_pose_command_b_smooth = lambda target: obj.pose_command[:, :3].copy_(target)
    return obj


class ProductionMethodTests(unittest.TestCase):
    def test_synthetic_forward_speed_is_point_two_for_different_turn_rates(self):
        obj = synthetic_fixture()
        for _ in range(200):
            obj._update_synthetic_command()
        torch.testing.assert_close(obj.base_velocity_command_b[:, 0], torch.full((3,), .2), atol=1e-6, rtol=0)
        torch.testing.assert_close(obj.base_velocity_command_b[:, 1], torch.zeros(3))
        self.assertLess(obj.base_velocity_command_b[1, 2], -.35)
        self.assertGreater(obj.base_velocity_command_b[2, 2], .35)
        self.assertAlmostEqual(float(obj.base_velocity_command_b[0, 2]), 0.)

    def test_synthetic_start_delay_is_kept_while_overriding_heading_speed_scale(self):
        obj = synthetic_fixture()
        obj.cfg.base_motion_delay_s = .4
        for _ in range(19):
            obj._update_synthetic_command()
        torch.testing.assert_close(obj.base_velocity_command_b, torch.zeros(3, 3))
        for _ in range(200):
            obj._update_synthetic_command()
        torch.testing.assert_close(obj.base_velocity_command_b[:, 0], torch.full((3,), .2), atol=1e-6, rtol=0)

    def test_live_actual_tcp_measurement_changes_target_and_keeps_v_fixed(self):
        obj = live_fixture()
        for _ in range(160):
            obj._update_dual_command()
        self.assertTrue(obj._vision_valid[0])
        self.assertLess(obj.pose_command[0, 1], 0)
        self.assertAlmostEqual(float(obj.base_velocity_command_b[0, 0]), .2, places=6)
        self.assertGreater(float(obj.base_velocity_command_b[0, 2]), .15)
        obj._observations["rear"]["tcp_measured_b"][1] = -.04
        obj._update_dual_command()
        self.assertGreater(obj.pose_command[0, 1], 0)
        self.assertAlmostEqual(float(obj.base_velocity_command_b[0, 0]), .2, places=6)

    def test_live_geometry_compensates_base_translation_and_plane_offset(self):
        obj = live_fixture()
        obj.robot.data.body_pos_w[0, 0] += torch.tensor([.1, -.02, .03])
        points, tcp, plane = obj._current_geometry("rear")
        np.testing.assert_allclose(tcp.numpy(), [-.875, .06, -.44], atol=1e-6)
        np.testing.assert_allclose(plane.numpy(), [0, 0, 1, .53], atol=1e-6)
        self.assertLess(float((points @ plane[:3] + plane[3]).abs().max()), 1e-6)

    def encoder_model(self, obj):
        model = fk_module.URDFToolKinematics(ROOT / "source/b2_z1/b2_z1/assets/b2_z1/b2_z1_no_stick.urdf", "cpu")
        obj._tcp_kinematics = model
        obj._dual_arm_joint_ids = list(range(6))
        q_capture = torch.tensor([[.1, 2.3, -1.8, -.5, .05, .03]])
        obj.robot.data.joint_pos = q_capture.clone()
        capture_fk = model.position(q_capture)[0]
        obj._source_snapshot["rear"]["encoder_tcp_fk_b"] = capture_fk.tolist()
        return model, q_capture, capture_fk

    def test_encoder_propagation_keeps_stationary_arm_body_relative_during_base_translation(self):
        obj = live_fixture()
        _, _, capture_fk = self.encoder_model(obj)
        measured = capture_fk + torch.tensor([.01, .02, -.005])
        obj._observations["rear"]["tcp_measured_b"] = measured.tolist()
        obj.robot.data.body_pos_w[0, 0] += torch.tensor([.15, -.03, .02])
        points, predicted_tcp, _ = obj._current_geometry("rear")
        torch.testing.assert_close(predicted_tcp, measured, atol=1e-6, rtol=0)
        # The static seam still moves in base coordinates; the body-attached
        # unchanged arm must not be mistaken for another fixed world object.
        np.testing.assert_allclose(points[0].numpy(), [-1.15, .03, -.52], atol=1e-6)
        self.assertEqual(obj.robot.data.body_pos_w.shape[1], 1)  # no TCP link pose available

    def test_encoder_joint_delta_preserves_camera_bias_and_rotates_it_to_current_base(self):
        obj = live_fixture()
        model, q_capture, capture_fk = self.encoder_model(obj)
        residual = torch.tensor([.01, .02, -.005])
        obj._observations["rear"]["tcp_measured_b"] = (capture_fk + residual).tolist()
        q_now = q_capture + torch.tensor([[.03, -.01, .02, .01, -.02, .03]])
        obj.robot.data.joint_pos = q_now
        yaw = math.pi / 2
        obj.robot.data.body_state_w[0, 0, 3:7] = torch.tensor([math.cos(yaw/2), 0, 0, math.sin(yaw/2)])
        _, predicted_tcp, _ = obj._current_geometry("rear")
        expected = model.position(q_now)[0] + torch.tensor([.02, -.01, -.005])
        torch.testing.assert_close(predicted_tcp, expected, atol=1e-6, rtol=0)
        self.assertGreater(float(torch.linalg.vector_norm(predicted_tcp - model.position(q_now)[0])), .02)
        self.assertEqual(obj.robot.data.body_pos_w.shape[1], 1)  # cannot read a simulated TCP

    def test_unknown_capture_cannot_be_consumed_even_with_matching_clock_and_reset(self):
        obj = live_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest_dual_trajectory.json"
            payload = dict(schema=contract.SCHEMA, capture_generation=3)
            for stream, observation in obj._observations.items():
                packet = {**observation, "sequence": 9, "valid": True,
                          "points_b": observation["points_b"].tolist()}
                payload[stream] = packet
            path.write_text(json.dumps(payload))
            obj.cfg.trajectory_file = str(path)
            obj._observations = dict(front=None, rear=None)
            obj._ingest_latest_vision_file()
            self.assertIsNone(obj._observations["front"])
            self.assertIsNone(obj._observations["rear"])
            self.assertIn("capture no longer", obj._reject_reasons["front"])

    def test_prepare_moves_tool_toward_visible_area_without_faking_a_tcp_measurement(self):
        obj = live_fixture()
        obj._visual_started = False
        obj._observations = dict(front=None, rear=None)
        obj._rear_measured_b = None
        obj.pose_command[:, :3] = torch.tensor([[.2, 0, .3]])
        obj._update_dual_command()
        torch.testing.assert_close(obj.pose_command[0, :3], torch.tensor([-.775, 0., -.48]))
        torch.testing.assert_close(obj.base_velocity_command_b, torch.zeros(1, 3))
        self.assertIsNone(obj._rear_measured_b)
        self.assertFalse(obj._vision_valid[0])
        self.assertFalse(obj._visual_started)

    def test_prepare_can_use_measured_ground_even_before_markers_are_visible(self):
        obj = live_fixture()
        obj._visual_started = False
        snapshot = obj._source_snapshot["rear"]
        obj._ground_observation = (np.array([0., 0., 1., .6]), snapshot)
        obj._observations = dict(front=None, rear=None)
        obj._rear_measured_b = None
        obj._update_dual_command()
        torch.testing.assert_close(obj.pose_command[0, :3], torch.tensor([-.775, 0., -.51]))
        self.assertIsNone(obj._rear_measured_b)
        self.assertFalse(obj._visual_started)

    def test_far_measured_tcp_does_not_release_locomotion_but_near_visual_tcp_does(self):
        obj = live_fixture()
        obj._visual_started = False
        obj._visual_start_elapsed = 0.
        obj._observations["rear"]["tcp_measured_b"] = [-.775, .30, -.41]
        obj._update_dual_command()
        self.assertFalse(obj._visual_started)
        torch.testing.assert_close(obj.base_velocity_command_b, torch.zeros(1, 3))
        obj._observations["rear"]["tcp_measured_b"] = [-.775, .04, -.41]
        obj._update_dual_command()
        self.assertTrue(obj._visual_started)
        self.assertGreater(float(obj.base_velocity_command_b[0, 0]), 0.)

    def test_unchanged_renderer_frame_does_not_create_new_capture_or_refresh_clock(self):
        obj = live_fixture()
        output = dict(rgb=torch.zeros(1, 8, 12, 4, dtype=torch.uint8),
                      distance_to_image_plane=torch.full((1, 8, 12, 1), .6))
        camera_cfg = NS(offset=NS(convention="ros", pos=(0, 0, 0), rot=(1, 0, 0, 0)))
        obj.env.scene = {
            name: NS(frame=torch.tensor([17]), data=NS(output=output, intrinsic_matrices=torch.eye(3).unsqueeze(0)),
                     cfg=camera_cfg) for name in ("front_rgb", "rear_d435i")}
        with tempfile.TemporaryDirectory() as directory:
            obj.cfg.vision_exchange_directory = directory
            obj._captures = dict(front=OrderedDict(), rear=OrderedDict())
            obj._export_latest_camera_frame()
            snapshot = dict(obj._captures["front"][0])
            obj.env.common_step_counter = 4
            obj._export_latest_camera_frame()
            self.assertEqual(list(obj._captures["front"]), [0])
            self.assertEqual(obj._captures["front"][0]["capture_timestamp"], snapshot["capture_timestamp"])
            self.assertEqual(list(obj._captures["rear"]), [0])
            front_meta = json.loads((Path(directory) / "front/latest_frame.json").read_text())
            self.assertNotIn("depth_file", front_meta)


if __name__ == "__main__":
    unittest.main(verbosity=2)
