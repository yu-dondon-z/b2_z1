"""Fixed-speed front guidance and an independently observed rear tool loop.

Rear RGB-D measurements close a bounded target-residual loop outside the
actor. Arm encoder names come from the task configuration; changing the arm
also requires a compatible actor. This is not a policy trained to consume new
visual observations, nor a real-robot qualification.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import time
import weakref

import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .dual_vision_contract import SCHEMA, finite_array, rear_target, validate_capture, validate_stream
from .dual_vision_rewards import physical_crack_lateral_distance_w, physical_crack_polyline_xy
from .dual_vision_kinematics import URDFToolKinematics
from .path_guidance import local_path_geometry
from .vision_trajectory_command import RearVisionTrajectoryCommand, RearVisionTrajectoryCommandCfg


class DualVisionTrajectoryCommand(RearVisionTrajectoryCommand):
    """Two independent image streams; forward speed does not depend on heading."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._captures = {name: OrderedDict() for name in ("front", "rear")}
        self._camera_frames = {"front": -1, "rear": -1}
        self._consumed = {"front": -1, "rear": -1}
        self._observations = {"front": None, "rear": None}
        self._reject_reasons = {"front": "waiting for images", "rear": "waiting for images"}
        self._rear_nominal_b = None
        self._rear_measured_b = None
        self._rear_correction_b = None
        self._rear_projection_error_m = None
        self._last_packet_mtime = -1
        self._source_snapshot = {}
        self._visual_start_elapsed = 0.0
        self._visual_started = False
        self._ground_observation = None
        self._tcp_kinematics = None
        if cfg.mode == "file":
            self._tcp_kinematics = URDFToolKinematics(
                self.robot.cfg.spawn.asset_path, self.device, joint_names=cfg.arm_joint_names)
            self._dual_arm_joint_ids, resolved_names = self.robot.find_joints(
                list(cfg.arm_joint_names), preserve_order=True)
            if tuple(resolved_names) != tuple(cfg.arm_joint_names):
                raise ValueError(f"articulation arm joint order differs from configured encoder order: {resolved_names}")
        self._rear_camera_visual = None
        if cfg.mode == "file" and cfg.show_rear_camera_frustum and env.sim.has_gui():
            from .dual_camera_visuals import RearCameraViewportVisual
            self._rear_camera_visual = RearCameraViewportVisual(
                env.scene[cfg.rear_camera_name], near_m=0.10,
                far_m=cfg.rear_camera_frustum_display_length_m,
            )
            # RL env.close releases commands before the scene and window.
            weakref.finalize(self, self._rear_camera_visual.close)
        if cfg.path_speed_mps != 0.20:
            raise ValueError("VA16 fixed-speed contract requires path_speed_mps=0.20")
        if cfg.camera_ring_slots < 4 or cfg.camera_export_period_steps < 1:
            raise ValueError("invalid camera export cadence/ring")
        if not 0 <= cfg.rear_feedback_gain <= 1 or not 0 <= cfg.rear_feedback_max_correction_m <= 0.05:
            raise ValueError("rear target feedback must be bounded")

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if not hasattr(self, "_captures"):
            return
        self._captures = {name: OrderedDict() for name in ("front", "rear")}
        self._camera_frames = {"front": -1, "rear": -1}
        self._consumed = {"front": -1, "rear": -1}
        self._observations = {"front": None, "rear": None}
        self._source_snapshot = {}
        self._reject_reasons = {"front": "waiting after reset", "rear": "waiting after reset"}
        self._visual_start_elapsed = 0.0
        self._visual_started = False
        self._ground_observation = None
        self._last_packet_mtime = -1

    def _update_synthetic_command(self):
        # Preserve the trained TCP/steering curriculum, but replace every
        # synthetic branch's cos(heading)-scaled forward component.
        previous = self._filtered_velocity_command_b[:, 0].clone()
        super()._update_synthetic_command()
        dt = float(self.env.step_dt)
        alpha = 1.0 - math.exp(-dt / max(self.cfg.velocity_filter_time_constant_s, dt))
        blend = self._velocity_start_blend()
        desired = self._command_transition_start_velocity_b[:, 0] * (1 - blend) + self.cfg.path_speed_mps * blend
        self._filtered_velocity_command_b[:, 0] = previous + alpha * (desired - previous)
        self._filtered_velocity_command_b[:, 1] = 0.0
        self.base_velocity_command_b[:] = self._filtered_velocity_command_b

    def _update_command(self):
        if self.cfg.mode == "synthetic":
            super()._update_command()
            return
        self._export_latest_camera_frame()
        self._ingest_latest_vision_file()
        self._update_dual_command()
        self.metrics["vision_valid"][:] = self._vision_valid.float()
        self.metrics["path_lateral_error"][:] = self._path_lateral_error_b.abs()
        self.metrics["path_heading_error"][:] = self._path_heading_error_b.abs()
        self._write_dual_status()
        if self._rear_camera_visual is not None:
            if not self._rear_camera_visual.available:
                # Isaac's environment UI initially selects World after the
                # viewer config is loaded. Restore asset following once the
                # viewport exists; later user view changes remain available.
                self.env.viewport_camera_controller.update_view_to_asset_root("robot")
            self._rear_camera_visual.update()

    def crack_lateral_distance_w(self, points_w):
        """Physical road geometry is available to rewards/evaluation only."""
        if self.cfg.mode != "file":
            return super().crack_lateral_distance_w(points_w)
        if not hasattr(self, "_reward_centerline_xy"):
            road = self.env.cfg.scene.road_left.spawn
            self._reward_centerline_xy = physical_crack_polyline_xy(
                amplitude_m=road.amplitude, wavelength_m=road.wavelength,
                curve_half_length_m=road.curve_half_length, road_half_length_m=road.half_length,
                mesh_step_m=road.mesh_step, device=self.device, dtype=points_w.dtype)
        return physical_crack_lateral_distance_w(points_w, self.env.scene.env_origins, self._reward_centerline_xy)

    def _export_latest_camera_frame(self):
        if self.env.common_step_counter % self.cfg.camera_export_period_steps:
            return
        import cv2
        import numpy as np

        root = Path(self.cfg.vision_exchange_directory).expanduser()
        sequence = self._camera_export_sequence
        # Both cameras are sampled at the same physics boundary. Wall capture
        # is host receipt, not a fabricated hardware exposure timestamp.
        received = time.monotonic()
        base_p = self.robot.data.body_pos_w[0, self.base_idx].detach().cpu().tolist()
        base_q = self.robot.data.body_state_w[0, self.base_idx, 3:7].detach().cpu().tolist()
        capture_fk = None
        if getattr(self, "_tcp_kinematics", None) is not None:
            angles = self.robot.data.joint_pos[0, self._dual_arm_joint_ids].reshape(1, -1)
            capture_fk = self._tcp_kinematics.position(angles)[0].detach().cpu().tolist()
        for stream, camera_name in (("front", self.cfg.front_camera_name), ("rear", self.cfg.rear_camera_name)):
            camera = self.env.scene[camera_name]
            with torch.inference_mode(False):
                data = camera.data
                frame_id = int(camera.frame[0].item())
                if frame_id == self._camera_frames[stream]:
                    continue
                rgb = data.output["rgb"][0].detach().cpu().numpy()[..., :3]
                intrinsic = data.intrinsic_matrices[0].detach().cpu().tolist()
                depth = (data.output["distance_to_image_plane"][0].detach().cpu().numpy().squeeze()
                         if stream == "rear" else None)
            if not rgb.size:
                continue
            directory = root / stream
            directory.mkdir(parents=True, exist_ok=True)
            slot = sequence % self.cfg.camera_ring_slots
            rgb_name = f"rgb_slot_{slot}.png"
            temporary = directory / f".{rgb_name}.tmp.png"
            if not cv2.imwrite(str(temporary), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError("cannot export camera RGB")
            os.replace(temporary, directory / rgb_name)
            if camera.cfg.offset.convention != "ros":
                raise ValueError("VA16 requires ROS optical camera offsets")
            metadata = {
                "schema": "b2_z1_dual_camera_frame_v1", "frame_id": "b2_base",
                "camera_name": camera_name, "stream": stream, "sequence": sequence,
                "camera_frame": frame_id, "capture_generation": self._vision_generation,
                "capture_timestamp": received, "capture_timestamp_basis": "host_receive_monotonic",
                "capture_sim_time_s": self.env.common_step_counter * float(self.env.step_dt),
                "timestamp_ns": time.time_ns(), "rgb_file": rgb_name,
                "image_size": [int(rgb.shape[1]), int(rgb.shape[0])], "intrinsic_matrix": intrinsic,
                "camera_to_base": {"position": list(camera.cfg.offset.pos),
                                   "quaternion_wxyz": list(camera.cfg.offset.rot)},
                "base_position_w": base_p, "base_quaternion_w": base_q,
            }
            if depth is not None:
                depth_name = f"depth_slot_{slot}_mm.png"
                millimeters = np.where(np.isfinite(depth) & (depth > 0),
                                       np.clip(depth * 1000, 0, 65535), 0).astype(np.uint16)
                temporary = directory / f".{depth_name}.tmp.png"
                if not cv2.imwrite(str(temporary), millimeters):
                    raise RuntimeError("cannot export camera depth")
                os.replace(temporary, directory / depth_name)
                metadata.update(depth_file=depth_name, depth_unit_m=0.001)
            self._atomic_json(directory / f"frame_slot_{slot}.json", metadata)
            self._atomic_json(directory / "latest_frame.json", metadata)
            # Encoder FK stays private to the controller; perception receives
            # images/calibration only. This is used solely for motion deltas.
            self._captures[stream][sequence] = dict(metadata, encoder_tcp_fk_b=capture_fk)
            while len(self._captures[stream]) > self.cfg.capture_history_frames:
                self._captures[stream].popitem(last=False)
            self._camera_frames[stream] = frame_id
        self._camera_export_sequence += 1

    @staticmethod
    def _atomic_json(path, payload):
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)

    def _ingest_latest_vision_file(self):
        path = Path(self.cfg.trajectory_file)
        try:
            modified = path.stat().st_mtime_ns
            if modified == self._last_packet_mtime:
                return
            payload = json.loads(path.read_text(encoding="utf-8"))
            self._last_packet_mtime = modified
            if payload.get("schema") != SCHEMA or payload.get("capture_generation") != self._vision_generation:
                raise ValueError("wrong dual-camera schema/reset generation")
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as error:
            self._reject_reasons = dict.fromkeys(("front", "rear"), str(error))
            return
        now = time.monotonic()
        for stream in ("front", "rear"):
            packet = payload.get(stream, {})
            sequence = packet.get("sequence", -1) if isinstance(packet, dict) else -1
            if type(sequence) is not int or sequence <= self._consumed[stream]:
                continue
            snapshot = self._captures[stream].get(sequence)
            # Ground can be observed before the folded tool/markers enter the
            # rear field of view. It supports preparation, never a fake TCP.
            if stream == "rear" and snapshot is not None and "ground_plane_b" in packet:
                try:
                    validate_capture(packet, snapshot, self._vision_generation, now, self.cfg.stale_timeout_s)
                    plane = finite_array(packet["ground_plane_b"], (4,))
                    length = float((plane[:3] ** 2).sum() ** 0.5)
                    if not 0.5 <= length <= 2 or abs(plane[2] / length) < 0.4:
                        raise ValueError("invalid observed preparation ground")
                    self._ground_observation = (plane / length, snapshot)
                except (ValueError, TypeError):
                    pass
            try:
                if snapshot is None:
                    raise ValueError("capture no longer in this session's history")
                observation = validate_stream(packet, snapshot, self._vision_generation, now,
                                              self.cfg.stale_timeout_s, rear=stream == "rear")
            except (ValueError, TypeError) as error:
                self._reject_reasons[stream] = str(error)
                continue
            self._observations[stream] = observation
            self._source_snapshot[stream] = snapshot
            self._consumed[stream] = sequence
            self._reject_reasons[stream] = ""
        self._last_source_sequence = self._consumed["front"]

    def _current_geometry(self, stream):
        observation = self._observations[stream]
        if observation is None or time.monotonic() - observation["capture_timestamp"] > self.cfg.stale_timeout_s:
            return None
        snapshot = self._source_snapshot[stream]
        capture_p = torch.tensor(snapshot["base_position_w"], device=self.device)
        capture_q = torch.tensor(snapshot["base_quaternion_w"], device=self.device).unsqueeze(0)
        current_p = self.robot.data.body_pos_w[0, self.base_idx]
        current_q = self.robot.data.body_state_w[0, self.base_idx, 3:7].unsqueeze(0)

        def transform(points):
            points = torch.as_tensor(points, device=self.device, dtype=current_p.dtype).reshape(-1, 3)
            world = capture_p + quat_apply(capture_q.expand(len(points), -1), points)
            return quat_apply_inverse(current_q.expand(len(points), -1), world - current_p)

        points = transform(observation["points_b"])
        if stream == "front":
            return points
        measured = transform(observation["tcp_measured_b"])[0]
        if getattr(self, "_tcp_kinematics", None) is not None:
            capture_fk = torch.tensor(snapshot["encoder_tcp_fk_b"], device=self.device, dtype=current_p.dtype)
            angles = self.robot.data.joint_pos[0, self._dual_arm_joint_ids].reshape(1, -1)
            current_fk = self._tcp_kinematics.position(angles)[0]
            visual_capture = torch.as_tensor(observation["tcp_measured_b"], device=self.device, dtype=current_p.dtype)
            # Preserve the camera-measured model residual, then propagate the
            # nominal encoder motion (including base motion) to the present.
            residual_w = quat_apply(capture_q, (visual_capture - capture_fk).unsqueeze(0))
            measured = current_fk + quat_apply_inverse(current_q, residual_w)[0]
        plane = torch.as_tensor(observation["ground_plane_b"], device=self.device, dtype=current_p.dtype)
        normal_w = quat_apply(capture_q, plane[:3].unsqueeze(0))[0]
        normal_b = quat_apply_inverse(current_q, normal_w.unsqueeze(0))[0]
        plane_current = torch.cat((normal_b, (plane[3] + torch.dot(normal_w, current_p - capture_p)).reshape(1)))
        return points, measured, plane_current

    def _update_dual_command(self):
        front, rear = self._current_geometry("front"), self._current_geometry("rear")
        self._vision_valid[:] = False
        self._latest_path_b = front if front is not None else torch.empty((0, 3), device=self.device)
        if front is None or rear is None:
            self._filter_file_velocity_command(torch.zeros_like(self.base_velocity_command_b))
            if not self._visual_started:
                self._prepare_visual_tool()
            # Keep the previous task target; do not re-anchor it to live TCP.
            if torch.linalg.vector_norm(self.base_velocity_command_b[0]) < self.cfg.file_stop_epsilon:
                self._visual_start_elapsed = 0.0
            return
        try:
            forward = front[front[:, 0] > 0.10]
            if len(forward) < self.cfg.minimum_path_points:
                raise ValueError("not enough forward seam points")
            tangent, lateral = local_path_geometry(forward)
            if tangent[0] < self.cfg.minimum_forward_tangent:
                raise ValueError("front seam has no forward tangent")
            heading = torch.atan2(tangent[1], tangent[0])
            correction_heading = torch.atan2(self.cfg.cross_track_gain * lateral,
                                            torch.full_like(lateral, self.cfg.path_lookahead_m))
            raw_heading = (heading + correction_heading).clamp(-1.2, 1.2)
            self._filtered_path_heading.lerp_(raw_heading.reshape(1), self.cfg.path_heading_filter_alpha)
            self._path_lateral_error_b[:] = lateral
            self._path_heading_error_b[:] = heading
            points, tcp, plane = rear
            up = -self.robot.data.projected_gravity_b[0]
            target, nominal, residual, error = rear_target(
                points.detach().cpu().numpy(), plane.detach().cpu().numpy(), tcp.detach().cpu().numpy(),
                up.detach().cpu().numpy(), self.cfg.rear_work_x_range_m, self.cfg.tool_height_m,
                self.cfg.rear_feedback_gain, self.cfg.rear_feedback_max_correction_m)
        except (ValueError, RuntimeError) as error:
            self._reject_reasons["control"] = str(error)
            self._filter_file_velocity_command(torch.zeros_like(self.base_velocity_command_b))
            if not self._visual_started:
                self._prepare_visual_tool()
            return
        self._rear_nominal_b, self._rear_measured_b = nominal.tolist(), tcp.detach().cpu().tolist()
        self._rear_correction_b, self._rear_projection_error_m = residual.tolist(), error
        self._set_file_pose_command_b_smooth(torch.tensor(target, dtype=self.pose_command.dtype,
                                                        device=self.device).unsqueeze(0))
        if not self._visual_started:
            # Preparation is separate from visual tracking. A measured TCP is
            # needed before locomotion starts; FK/initial targets are not vision.
            if float(torch.linalg.vector_norm(torch.tensor(nominal, device=self.device) - tcp)) > self.cfg.visual_start_tolerance_m:
                self._filter_file_velocity_command(torch.zeros_like(self.base_velocity_command_b))
                return
            self._visual_started = True
        self._visual_start_elapsed += float(self.env.step_dt)
        duration = max(self.cfg.command_transition_time_s, float(self.env.step_dt))
        fraction = min(self._visual_start_elapsed / duration, 1.0)
        blend = fraction * fraction * (3 - 2 * fraction)
        desired = torch.zeros_like(self.base_velocity_command_b)
        desired[:, 0] = self.cfg.path_speed_mps * blend
        desired[:, 2] = (self.cfg.path_heading_gain * self._filtered_path_heading).clamp(
            -self.cfg.max_yaw_rate_rps, self.cfg.max_yaw_rate_rps) * blend
        self._filter_file_velocity_command(desired)
        self._vision_valid[:] = True
        self._last_vision_wall_time = min(self._observations[name]["capture_timestamp"] for name in ("front", "rear"))
        self._reject_reasons.pop("control", None)

    def _prepare_visual_tool(self):
        """Move into the visible work area while B2 remains commanded to stand."""
        position = torch.tensor([list(self.cfg.prepare_position_b)], dtype=self.pose_command.dtype,
                                device=self.device)
        if self._ground_observation is not None:
            plane_array, snapshot = self._ground_observation
            if time.monotonic() - snapshot["capture_timestamp"] <= self.cfg.stale_timeout_s:
                capture_q = torch.tensor([snapshot["base_quaternion_w"]], device=self.device)
                capture_p = torch.tensor(snapshot["base_position_w"], device=self.device)
                current_q = self.robot.data.body_state_w[0, self.base_idx, 3:7].unsqueeze(0)
                current_p = self.robot.data.body_pos_w[0, self.base_idx]
                plane = torch.as_tensor(plane_array, dtype=current_p.dtype, device=self.device)
                normal_w = quat_apply(capture_q, plane[:3].unsqueeze(0))[0]
                normal_b = quat_apply_inverse(current_q, normal_w.unsqueeze(0))[0]
                d = plane[3] + torch.dot(normal_w, current_p - capture_p)
                position[0, 0] = sum(self.cfg.rear_work_x_range_m) * 0.5
                position[0, 2] = -(normal_b[0] * position[0, 0] + normal_b[1] * position[0, 1] + d) / normal_b[2]
                position[0] += self.cfg.tool_height_m * -self.robot.data.projected_gravity_b[0]
        self._set_file_pose_command_b_smooth(position)

    def _write_dual_status(self):
        if self.env.common_step_counter % self.cfg.controller_status_period_steps:
            return
        now = time.monotonic()
        base = self.robot.data.body_pos_w[0, self.base_idx]
        quat = self.robot.data.body_state_w[0, self.base_idx, 3:7]
        # Evaluation only. This value never feeds the visual residual above.
        actual_w = self.robot.data.body_pos_w[0, self.body_idx]
        actual_b = quat_apply_inverse(quat.unsqueeze(0), (actual_w - base).unsqueeze(0))[0]
        payload = {
            "schema": SCHEMA, "phase": "external_visual_feedback_baseline",
            "timestamp_ns": time.time_ns(), "simulation_step": self.env.common_step_counter,
            "simulation_time_s": self.env.common_step_counter * float(self.env.step_dt),
            "capture_generation": self._vision_generation,
            "visual_tracking_started": self._visual_started,
            "tool_target_active": bool(self._vision_valid[0]), "source_sequence_consumed": dict(self._consumed),
            "capture_age_s": {name: None if self._observations[name] is None else
                              now - self._observations[name]["capture_timestamp"] for name in ("front", "rear")},
            "reject_reasons": self._reject_reasons, "fixed_forward_command_mps": self.cfg.path_speed_mps,
            "base_velocity_command_b": self.base_velocity_command_b[0].detach().cpu().tolist(),
            "base_linear_velocity_b": self.robot.data.root_lin_vel_b[0].detach().cpu().tolist(),
            "base_position_w": base.detach().cpu().tolist(),
            "base_projected_gravity_b": self.robot.data.projected_gravity_b[0].detach().cpu().tolist(),
            "tool_target_position_b": self.pose_command[0, :3].detach().cpu().tolist(),
            "tcp_visual_measured_b": self._rear_measured_b, "tcp_visual_nominal_b": self._rear_nominal_b,
            "tcp_visual_target_correction_b": self._rear_correction_b,
            "tcp_visual_projection_error_m": self._rear_projection_error_m,
            "tcp_visual_measurement_model": "captured RGB-D TCP plus encoder FK motion delta",
            "evaluation_only_tcp_ground_truth_b": actual_b.detach().cpu().tolist(),
            "evaluation_only_tcp_ground_truth_w": actual_w.detach().cpu().tolist(),
            "evaluation_only_tcp_projection_to_physical_crack_m": float(
                self.crack_lateral_distance_w(actual_w.reshape(1, 1, 3))[0, 0]),
            "evaluation_only_tcp_visual_position_error_m": None if self._rear_measured_b is None else
                float(torch.linalg.vector_norm(actual_b - torch.tensor(self._rear_measured_b, device=self.device))),
            "actor_observations": self._policy_observation_count(),
            "actor_actions": int(self.env.action_manager.total_action_dim),
            "arm_joint_names": list(self.cfg.arm_joint_names), "hardware_qualified": False,
        }
        road = self.env.cfg.scene.road_left.spawn
        payload["evaluation_scene_geometry"] = {
            "amplitude_m": road.amplitude, "wavelength_m": road.wavelength,
            "curve_half_length_m": road.curve_half_length, "road_half_length_m": road.half_length,
            "mesh_step_m": road.mesh_step,
            "env_origin_w": self.env.scene.env_origins[0].detach().cpu().tolist(),
        }
        root = Path(self.cfg.vision_exchange_directory)
        root.mkdir(parents=True, exist_ok=True)
        self._atomic_json(root / "latest_controller_status.json", payload)
        with (root / "controller_history.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(payload, allow_nan=False) + "\n")

    def _policy_observation_count(self):
        """Read the actual manager shape, including non-concatenated groups."""
        shape = self.env.observation_manager.group_obs_dim["policy"]
        if isinstance(shape, tuple):
            return math.prod(shape)
        return sum(math.prod(term_shape) for term_shape in shape)


@configclass
class DualVisionTrajectoryCommandCfg(RearVisionTrajectoryCommandCfg):
    class_type: type = DualVisionTrajectoryCommand
    arm_joint_names: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 7))
    front_camera_name: str = "front_rgb"
    rear_camera_name: str = "rear_d435i"
    show_rear_camera_frustum: bool = False
    rear_camera_frustum_display_length_m: float = 1.50
    path_speed_mps: float = 0.20
    camera_export_period_steps: int = 2
    camera_ring_slots: int = 8
    capture_history_frames: int = 128
    rear_feedback_gain: float = 0.35
    rear_feedback_max_correction_m: float = 0.03
    prepare_position_b: tuple[float, float, float] = (-0.775, 0.0, -0.48)
    visual_start_tolerance_m: float = 0.10
