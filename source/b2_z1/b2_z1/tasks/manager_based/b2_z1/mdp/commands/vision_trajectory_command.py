"""Rear-tool command driven by a cached visual crack trajectory."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Literal

import torch
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_mul,
)

from .pose_command_wbc import UniformPoseWBCCommand
from .command_cfg import UniformPoseWBCCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RearVisionTrajectoryCommand(UniformPoseWBCCommand):
    """Track a crack after it has passed underneath B2 into the rear workspace.

    In ``file`` mode, fresh robot-frame points produced by RHACrackNet are
    transformed to world coordinates and accumulated. B2 follows the tangent
    of the latest forward path while Z1 tracks a cached point in a rear x
    window. In ``synthetic`` mode, equivalent rear-workspace commands are
    generated without rendering so the policy can be trained efficiently.
    """

    cfg: RearVisionTrajectoryCommandCfg

    def __init__(self, cfg: RearVisionTrajectoryCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.base_velocity_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._vision_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_file_mtime_ns = -1
        self._last_vision_wall_time = -math.inf
        self._cached_path_w = torch.empty((0, 3), device=self.device)
        self._latest_path_b = torch.empty((0, 3), device=self.device)
        self._synthetic_phase = torch.zeros(self.num_envs, device=self.device)
        self._synthetic_amplitude = torch.zeros(self.num_envs, device=self.device)
        self._synthetic_wavenumber = torch.zeros(self.num_envs, device=self.device)
        self._synthetic_speed = torch.zeros(self.num_envs, device=self.device)
        self._camera_export_sequence = 0
        self.metrics["vision_valid"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cached_path_points"] = torch.zeros(self.num_envs, device=self.device)
        if cfg.mode == "file" and self.num_envs != 1:
            raise ValueError("RearVisionTrajectoryCommand file mode requires scene.num_envs == 1")

    def __str__(self) -> str:
        return (
            "RearVisionTrajectoryCommand:\n"
            f"\tMode: {self.cfg.mode}\n"
            f"\tRear work x: {self.cfg.rear_work_x_range_m}\n"
            f"\tVision file: {self.cfg.trajectory_file}"
        )

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        if self.cfg.mode == "synthetic":
            count = len(env_ids)
            amp_range = self.cfg.synthetic_lateral_amplitude_range_m
            wavelength_range = self.cfg.synthetic_wavelength_range_m
            speed_range = self.cfg.synthetic_speed_range_mps
            self._synthetic_amplitude[env_ids] = (
                torch.rand(count, device=self.device) * (amp_range[1] - amp_range[0]) + amp_range[0]
            )
            wavelength = (
                torch.rand(count, device=self.device) * (wavelength_range[1] - wavelength_range[0])
                + wavelength_range[0]
            )
            self._synthetic_wavenumber[env_ids] = 2.0 * math.pi / wavelength
            self._synthetic_speed[env_ids] = (
                torch.rand(count, device=self.device) * (speed_range[1] - speed_range[0]) + speed_range[0]
            )
            self._synthetic_phase[env_ids] = torch.rand(count, device=self.device) * 2.0 * math.pi
            self._vision_valid[env_ids] = True
        else:
            self._cached_path_w = torch.empty((0, 3), device=self.device)
            self._latest_path_b = torch.empty((0, 3), device=self.device)
            self._vision_valid[env_ids] = False
        self._hold_current_tool_pose(env_ids)
        self._reset_tracked_trajectory(env_ids)

    def _update_command(self):
        if self.cfg.mode == "synthetic":
            self._update_synthetic_command()
        else:
            self._export_latest_camera_frame()
            self._ingest_latest_vision_file()
            self._update_file_command()
        self.metrics["vision_valid"][:] = self._vision_valid.float()
        self.metrics["cached_path_points"][:] = float(len(self._cached_path_w))

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["vision_valid"][:] = self._vision_valid.float()
        self.metrics["cached_path_points"][:] = float(len(self._cached_path_w))

    def _fixed_orientation_b(self) -> torch.Tensor:
        roll, pitch, yaw = self.cfg.tool_orientation_rpy
        values = torch.full((self.num_envs,), roll, device=self.device)
        pitches = torch.full((self.num_envs,), pitch, device=self.device)
        yaws = torch.full((self.num_envs,), yaw, device=self.device)
        return quat_from_euler_xyz(values, pitches, yaws)

    def _set_pose_command_b(self, position_b: torch.Tensor):
        base_pos_w = self.robot.data.body_pos_w[:, self.base_idx]
        base_quat_w = self.robot.data.body_state_w[:, self.base_idx, 3:7]
        quat_b = self._fixed_orientation_b()
        self.pose_command[:, :3] = position_b
        self.pose_command[:, 3:] = quat_b
        position_w = base_pos_w + quat_apply(base_quat_w, position_b)
        quat_w = quat_mul(base_quat_w, quat_b)
        self.tracking_pose_w[:, :3] = position_w
        self.tracking_pose_w[:, 3:] = quat_w
        self.target_pose_w[:] = self.tracking_pose_w

    def _hold_current_tool_pose(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        base_pos_w = self.robot.data.body_pos_w[env_ids, self.base_idx]
        base_quat_w = self.robot.data.body_state_w[env_ids, self.base_idx, 3:7]
        ee_pos_w = self.robot.data.body_pos_w[env_ids, self.body_idx]
        position_b = quat_apply_inverse(base_quat_w, ee_pos_w - base_pos_w)
        quat_b = self._fixed_orientation_b()[env_ids]
        self.pose_command[env_ids, :3] = position_b
        self.pose_command[env_ids, 3:] = quat_b
        self.tracking_pose_w[env_ids, :3] = ee_pos_w
        self.tracking_pose_w[env_ids, 3:] = quat_mul(base_quat_w, quat_b)
        self.target_pose_w[env_ids] = self.tracking_pose_w[env_ids]

    def _update_synthetic_command(self):
        self._synthetic_phase += self._synthetic_speed * self.env.step_dt * self._synthetic_wavenumber
        lateral = self._synthetic_amplitude * torch.sin(self._synthetic_phase)
        rear_x = sum(self.cfg.rear_work_x_range_m) * 0.5
        target_b = torch.zeros(self.num_envs, 3, device=self.device)
        target_b[:, 0] = rear_x
        target_b[:, 1] = lateral
        target_b[:, 2] = self.cfg.synthetic_ground_z_b_m + self.cfg.tool_height_m
        self._set_pose_command_b(target_b)

        tangent_y = self._synthetic_amplitude * self._synthetic_wavenumber * torch.cos(self._synthetic_phase)
        heading_error = torch.atan(tangent_y)
        self.base_velocity_command_b[:, 0] = self._synthetic_speed
        self.base_velocity_command_b[:, 1] = 0.0
        self.base_velocity_command_b[:, 2] = (
            self.cfg.path_heading_gain * heading_error
        ).clamp(-self.cfg.max_yaw_rate_rps, self.cfg.max_yaw_rate_rps)
        self._vision_valid[:] = True

    def _ingest_latest_vision_file(self):
        path = Path(self.cfg.trajectory_file).expanduser()
        try:
            stat = path.stat()
        except FileNotFoundError:
            return
        if time.time_ns() - stat.st_mtime_ns > int(self.cfg.stale_timeout_s * 1.0e9):
            return
        if stat.st_mtime_ns == self._last_file_mtime_ns:
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("frame_id", "b2_base") != "b2_base":
                raise ValueError("vision trajectory frame_id must be 'b2_base'")
            points = torch.as_tensor(
                payload["control_waypoints_robot_xyz"], device=self.device, dtype=torch.float32
            ).reshape(-1, 3)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return
        finite = torch.all(torch.isfinite(points), dim=1)
        points = points[finite]
        if len(points) < self.cfg.minimum_path_points:
            return

        base_pos_w = self.robot.data.body_pos_w[0, self.base_idx]
        base_quat_w = self.robot.data.body_state_w[0, self.base_idx, 3:7]
        points_w = base_pos_w.unsqueeze(0) + quat_apply(base_quat_w.unsqueeze(0).expand(len(points), -1), points)
        self._latest_path_b = points
        self._merge_world_points(points_w)
        self._last_file_mtime_ns = stat.st_mtime_ns
        self._last_vision_wall_time = time.monotonic()

    def _export_latest_camera_frame(self):
        """Publish one aligned RGB-D frame for an external segmentation process.

        Four ring-buffer slots keep the transfer bounded and prevent a reader
        from seeing an image while that same file is being replaced. Metadata
        is replaced last and is therefore the commit marker for a new frame.
        """
        if self.env.common_step_counter % self.cfg.camera_export_period_steps != 0:
            return
        try:
            camera = self.env.scene[self.cfg.vision_camera_name]
            # Policy playback normally wraps env.step() in inference_mode.
            # Isaac camera buffers must not be allocated as inference tensors
            # because the renderer updates those buffers in-place later.
            with torch.inference_mode(False):
                camera_data = camera.data
                rgb_tensor = camera_data.output["rgb"]
                depth_tensor = camera_data.output["distance_to_image_plane"]
                intrinsic_tensor = camera_data.intrinsic_matrices
        except (KeyError, AttributeError):
            return
        if rgb_tensor.numel() == 0 or depth_tensor.numel() == 0:
            return

        # Import image dependencies only in one-environment file mode. This
        # keeps synthetic training free from CPU image conversion overhead.
        import cv2
        import numpy as np

        exchange_dir = Path(self.cfg.vision_exchange_directory).expanduser()
        exchange_dir.mkdir(parents=True, exist_ok=True)
        sequence = self._camera_export_sequence
        slot = sequence % self.cfg.camera_ring_slots
        rgb_name = f"rgb_slot_{slot}.png"
        depth_name = f"depth_slot_{slot}_mm.png"

        rgb = rgb_tensor[0].detach().cpu().numpy()[..., :3]
        depth = depth_tensor[0].detach().cpu().numpy().squeeze()
        finite = np.isfinite(depth) & (depth > 0.0)
        depth_mm = np.zeros(depth.shape, dtype=np.uint16)
        depth_mm[finite] = np.clip(depth[finite] * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)

        rgb_tmp = exchange_dir / f".{rgb_name}.tmp.png"
        depth_tmp = exchange_dir / f".{depth_name}.tmp.png"
        if not cv2.imwrite(str(rgb_tmp), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            return
        if not cv2.imwrite(str(depth_tmp), depth_mm):
            return
        os.replace(rgb_tmp, exchange_dir / rgb_name)
        os.replace(depth_tmp, exchange_dir / depth_name)

        if camera.cfg.offset.convention != "ros":
            raise ValueError("The vision bridge requires a ROS-convention camera offset")
        metadata = {
            "sequence": sequence,
            "timestamp_ns": time.time_ns(),
            "camera_name": self.cfg.vision_camera_name,
            "rgb_file": rgb_name,
            "depth_file": depth_name,
            "depth_unit_m": 0.001,
            "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "intrinsic_matrix": intrinsic_tensor[0].detach().cpu().tolist(),
            # The camera is rigidly attached to base_link. Using its nominal
            # offset avoids mixing a previous RTX render pose with the current
            # physics-frame base pose while the robot is moving.
            "camera_position_b": list(camera.cfg.offset.pos),
            "camera_quaternion_b_ros_wxyz": list(camera.cfg.offset.rot),
            "coordinate_convention": "b2_base: x forward, y left, z up; ROS optical: z forward, x right, y down",
        }
        metadata_tmp = exchange_dir / ".latest_frame.tmp.json"
        metadata_tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(metadata_tmp, exchange_dir / "latest_frame.json")
        self._camera_export_sequence += 1

    def _merge_world_points(self, new_points_w: torch.Tensor):
        if len(self._cached_path_w) == 0:
            self._cached_path_w = new_points_w[-self.cfg.max_cached_points :].clone()
            return
        accepted = []
        existing = self._cached_path_w
        for point in new_points_w:
            if torch.min(torch.linalg.norm(existing - point, dim=1)) >= self.cfg.merge_distance_m:
                accepted.append(point)
                existing = torch.cat((existing, point.unsqueeze(0)), dim=0)
        if accepted:
            self._cached_path_w = existing[-self.cfg.max_cached_points :]

    def _update_file_command(self):
        fresh = time.monotonic() - self._last_vision_wall_time <= self.cfg.stale_timeout_s
        if not fresh or len(self._latest_path_b) < self.cfg.minimum_path_points:
            self.base_velocity_command_b.zero_()
            self._vision_valid[:] = False
            return

        # The current RGB-D path is in the instantaneous B2 frame. Its tangent
        # controls locomotion forward; it is intentionally independent of the
        # rear tool target position.
        tangent_b = self._latest_path_b[-1, :2] - self._latest_path_b[0, :2]
        if tangent_b[0] < 0.0:
            tangent_b = -tangent_b
        tangent_b = tangent_b / torch.linalg.norm(tangent_b).clamp_min(1.0e-6)
        if tangent_b[0] < self.cfg.minimum_forward_tangent:
            self.base_velocity_command_b.zero_()
            self._vision_valid[:] = False
            return
        self.base_velocity_command_b[0, 0] = tangent_b[0] * self.cfg.path_speed_mps
        self.base_velocity_command_b[0, 1] = tangent_b[1] * self.cfg.path_speed_mps
        self.base_velocity_command_b[0, 2] = (
            self.cfg.path_heading_gain * torch.atan2(tangent_b[1], tangent_b[0])
        ).clamp(-self.cfg.max_yaw_rate_rps, self.cfg.max_yaw_rate_rps)

        base_pos_w = self.robot.data.body_pos_w[0, self.base_idx]
        base_quat_w = self.robot.data.body_state_w[0, self.base_idx, 3:7]
        cached_b = quat_apply_inverse(
            base_quat_w.unsqueeze(0).expand(len(self._cached_path_w), -1),
            self._cached_path_w - base_pos_w.unsqueeze(0),
        )
        keep = cached_b[:, 0] >= self.cfg.prune_behind_x_m
        self._cached_path_w = self._cached_path_w[keep]
        cached_b = cached_b[keep]
        if len(cached_b) == 0:
            self._vision_valid[:] = False
            return

        rear_min, rear_max = self.cfg.rear_work_x_range_m
        reachable = (cached_b[:, 0] >= rear_min) & (cached_b[:, 0] <= rear_max)
        if not torch.any(reachable):
            # Continue moving along a fresh path while holding the arm until a
            # detected point has physically passed into the rear workspace.
            self._vision_valid[:] = False
            return
        candidates = torch.nonzero(reachable, as_tuple=False).flatten()
        desired_x = 0.5 * (rear_min + rear_max)
        index = candidates[torch.argmin(torch.abs(cached_b[candidates, 0] - desired_x))]
        target_b = cached_b[index].clone()
        target_b[2] += self.cfg.tool_height_m
        self._set_pose_command_b(target_b.unsqueeze(0))
        self._vision_valid[:] = True

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        elif hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer.set_visibility(False)
            self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.goal_pose_visualizer.visualize(self.tracking_pose_w[:, :3], self.tracking_pose_w[:, 3:])
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])


@configclass
class RearVisionTrajectoryCommandCfg(UniformPoseWBCCommandCfg):
    """Configuration for rear-facing Z1 visual trajectory tracking."""

    class_type: type = RearVisionTrajectoryCommand
    mode: Literal["synthetic", "file"] = "synthetic"
    trajectory_file: str = ""
    vision_camera_name: str = "d435i"
    vision_exchange_directory: str = "/tmp/b2_z1_vision_bridge"
    camera_export_period_steps: int = 3
    camera_ring_slots: int = 4
    rear_work_x_range_m: tuple[float, float] = (-1.0, -0.55)
    tool_height_m: float = 0.10
    synthetic_ground_z_b_m: float = -0.45
    tool_orientation_rpy: tuple[float, float, float] = (0.0, math.pi / 2.0, math.pi)
    path_speed_mps: float = 0.25
    path_heading_gain: float = 1.5
    max_yaw_rate_rps: float = 0.5
    stale_timeout_s: float = 0.5
    minimum_path_points: int = 3
    minimum_forward_tangent: float = 0.2
    merge_distance_m: float = 0.03
    max_cached_points: int = 2048
    prune_behind_x_m: float = -2.0
    synthetic_lateral_amplitude_range_m: tuple[float, float] = (-0.15, 0.15)
    synthetic_wavelength_range_m: tuple[float, float] = (3.0, 8.0)
    synthetic_speed_range_mps: tuple[float, float] = (0.10, 0.60)
    goal_pose_visualizer_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/RearVision/goal_pose")
    current_pose_visualizer_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/RearVision/current_pose")
