"""VA15/VB1 environments with gun inertia and an effective hanging-hose load."""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply

from .b2z1_rl_amp_env import B2Z1ManagerAMPBasedRLEnv


class B2Z1LoadedGunAMPEnv(B2Z1ManagerAMPBasedRLEnv):
    """Apply gravity, acceleration reaction and drag at the gun's upper connector.

    The gun's own mass/inertia and collision come from b2_z1_gun.urdf. The
    unsupported part of the hose is represented by a configurable effective
    mass, not by a rigid rod or a ground-contacting deformable mesh.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._gun_body_id = self.scene["robot"].find_bodies("link_gun")[0][0]
        self._hose_point_b = torch.tensor(
            cfg.hose_attachment_gun_frame_m, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, 3)
        self._last_hose_velocity_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._filtered_hose_acceleration_w = torch.zeros_like(self._last_hose_velocity_w)
        self._hose_velocity_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hose_force_w = torch.zeros_like(self._last_hose_velocity_w)
        self.hose_torque_w = torch.zeros_like(self._last_hose_velocity_w)
        self.hose_tension_n = torch.zeros(self.num_envs, device=self.device)
        self.hose_tension_raw_n = torch.zeros(self.num_envs, device=self.device)
        self._hose_mass_kg = torch.full((self.num_envs,), cfg.hose_effective_mass_kg, device=self.device)
        self._hose_slack_m = torch.full((self.num_envs,), cfg.hose_slack_length_m, device=self.device)
        self._hose_anchor_w = self.scene.env_origins + torch.tensor(
            cfg.hose_truck_anchor_env_m, device=self.device, dtype=torch.float32
        )
        self._hose_force_sum_n = torch.zeros(self.num_envs, device=self.device)
        self._hose_torque_sum_nm = torch.zeros(self.num_envs, device=self.device)
        self._hose_tension_peak_n = torch.zeros(self.num_envs, device=self.device)
        self._hose_tension_raw_peak_n = torch.zeros(self.num_envs, device=self.device)
        self._hose_sample_count = torch.zeros(self.num_envs, device=self.device)
        self._sample_hose_parameters(torch.arange(self.num_envs, device=self.device))
        self._physics_wrench_hook = self._update_hose_wrench
        mass_range = getattr(cfg, "hose_mass_range_kg", None)
        print(
            f"[Loaded gun] Physical gun plus "
            f"{mass_range if mass_range is not None else cfg.hose_effective_mass_kg} kg effective hose; "
            f"truck anchor={cfg.hose_truck_anchor_env_m} m, slack={cfg.hose_slack_length_m:g} m",
            flush=True,
        )

    def _reset_idx(self, env_ids):
        if hasattr(self, "_hose_mass_kg"):
            counts = self._hose_sample_count[env_ids].clamp_min(1.0)
            mean_force = (self._hose_force_sum_n[env_ids] / counts).mean().item()
            mean_torque = (self._hose_torque_sum_nm[env_ids] / counts).mean().item()
            peak_tension = self._hose_tension_peak_n[env_ids].max().item()
            peak_raw_tension = self._hose_tension_raw_peak_n[env_ids].max().item()
        super()._reset_idx(env_ids)
        if hasattr(self, "_hose_velocity_initialized"):
            self._hose_velocity_initialized[env_ids] = False
            self._filtered_hose_acceleration_w[env_ids] = 0.0
            self._last_hose_velocity_w[env_ids] = 0.0
            self._sample_hose_parameters(env_ids)
            self.extras["log"].update({
                "VB1/hose_mass_kg": self._hose_mass_kg[env_ids].mean().item(),
                "VB1/hose_force_mean_n": mean_force,
                "VB1/hose_torque_mean_nm": mean_torque,
                "VB1/hose_tension_peak_n": peak_tension,
                "VB2/hose_raw_tension_peak_n": peak_raw_tension,
            })
            self._hose_force_sum_n[env_ids] = 0.0
            self._hose_torque_sum_nm[env_ids] = 0.0
            self._hose_tension_peak_n[env_ids] = 0.0
            self._hose_tension_raw_peak_n[env_ids] = 0.0
            self._hose_sample_count[env_ids] = 0.0

    def _sample_hose_parameters(self, env_ids):
        """Hold each sampled load constant until that environment resets."""
        count = len(env_ids)
        mass_range = getattr(self.cfg, "hose_mass_range_kg", None)
        if mass_range is None:
            self._hose_mass_kg[env_ids] = self.cfg.hose_effective_mass_kg
        else:
            low, high = mass_range
            self._hose_mass_kg[env_ids] = low + (high - low) * torch.rand(count, device=self.device)
        jitter = getattr(self.cfg, "hose_anchor_jitter_m", (0.0, 0.0, 0.0))
        anchor_center = torch.tensor(self.cfg.hose_truck_anchor_env_m, device=self.device)
        anchor_jitter = torch.tensor(jitter, device=self.device)
        self._hose_anchor_w[env_ids] = (
            self.scene.env_origins[env_ids] + anchor_center
            + (2.0 * torch.rand(count, 3, device=self.device) - 1.0) * anchor_jitter
        )
        slack_jitter = getattr(self.cfg, "hose_slack_jitter_m", 0.0)
        self._hose_slack_m[env_ids] = (
            self.cfg.hose_slack_length_m
            + (2.0 * torch.rand(count, device=self.device) - 1.0) * slack_jitter
        )

    def _update_hose_wrench(self):
        robot = self.scene["robot"]
        body = self._gun_body_id
        gun_quat_w = robot.data.body_link_quat_w[:, body]
        attachment_w = robot.data.body_link_pos_w[:, body] + quat_apply(gun_quat_w, self._hose_point_b)
        offset_from_com_w = attachment_w - robot.data.body_com_pos_w[:, body]
        velocity_w = robot.data.body_com_lin_vel_w[:, body] + torch.cross(
            robot.data.body_com_ang_vel_w[:, body], offset_from_com_w, dim=-1
        )
        dt = float(self.physics_dt)
        initialized = self._hose_velocity_initialized.unsqueeze(-1)
        acceleration_w = torch.where(
            initialized, (velocity_w - self._last_hose_velocity_w) / dt, 0.0
        )
        self._last_hose_velocity_w[:] = velocity_w
        self._hose_velocity_initialized[:] = True
        alpha = dt / (max(float(self.cfg.hose_acceleration_filter_s), 0.0) + dt)
        self._filtered_hose_acceleration_w.lerp_(acceleration_w, alpha)
        acceleration_w = self._filtered_hose_acceleration_w.clamp(
            -self.cfg.hose_max_acceleration_mps2, self.cfg.hose_max_acceleration_mps2
        )

        # A slack cable from the truck roof only pulls when stretched; it does
        # not push the gun. Extension damping resists motion away from the roof.
        to_anchor_w = self._hose_anchor_w - attachment_w
        cable_length = torch.linalg.vector_norm(to_anchor_w, dim=-1).clamp_min(1.0e-6)
        cable_direction_w = to_anchor_w / cable_length.unsqueeze(-1)
        stretch_m = (cable_length - self._hose_slack_m).clamp_min(0.0)
        moving_away_mps = (-torch.sum(velocity_w * cable_direction_w, dim=-1)).clamp_min(0.0)
        raw_tension_n = torch.where(
            stretch_m > 0.0,
            self.cfg.hose_tension_stiffness_n_per_m * stretch_m
            + self.cfg.hose_tension_damping_ns_per_m * moving_away_mps,
            0.0,
        )
        tension_cap_n = getattr(self.cfg, "hose_tension_max_n", None)
        tension_n = raw_tension_n.clamp(max=tension_cap_n) if tension_cap_n is not None else raw_tension_n
        self.hose_tension_n[:] = tension_n
        self.hose_tension_raw_n[:] = raw_tension_n

        # Weight acts at the upper connector. Its moment arm gives the Z1
        # shoulder/elbow a gravity torque; -m*a and drag add motion loading.
        force_w = (
            -self._hose_mass_kg.unsqueeze(-1) * acceleration_w
            - self.cfg.hose_linear_drag_ns_per_m * velocity_w
            + tension_n.unsqueeze(-1) * cable_direction_w
        )
        force_w[:, 2] -= self._hose_mass_kg * abs(float(self.cfg.sim.gravity[2]))
        force_norm = torch.linalg.vector_norm(force_w, dim=-1, keepdim=True).clamp_min(1.0e-6)
        force_w = force_w * (self.cfg.hose_max_force_n / force_norm).clamp(max=1.0)
        torque_w = torch.cross(offset_from_com_w, force_w, dim=-1)
        self.hose_force_w[:] = force_w
        self.hose_torque_w[:] = torque_w
        self._hose_force_sum_n += torch.linalg.vector_norm(force_w, dim=-1)
        self._hose_torque_sum_nm += torch.linalg.vector_norm(torque_w, dim=-1)
        self._hose_tension_peak_n = torch.maximum(self._hose_tension_peak_n, tension_n)
        self._hose_tension_raw_peak_n = torch.maximum(self._hose_tension_raw_peak_n, raw_tension_n)
        self._hose_sample_count += 1.0
        robot.set_external_force_and_torque(
            forces=force_w.unsqueeze(1),
            torques=torque_w.unsqueeze(1),
            body_ids=[body],
            is_global=True,
        )
