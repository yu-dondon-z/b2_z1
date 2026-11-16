"""VD9 rewards sampled after every physics step, never exposed to the actor."""

import torch
from isaaclab.managers import ManagerTermBase

from .tcp_frame import link_pose_w, quat_apply_inverse, tcp_pose_w
from .vd9_physics import descending_foot_penalty, highpass_tracking_penalty


class PhysicsRateTCPHighpass(ManagerTermBase):
    """Accumulate causal TCP tracking-error high-pass samples at physics rate."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params["asset_cfg"]
        self.command_name = cfg.params.get("command_name", "ee_target")
        self.base_name = cfg.params.get("base_name", "base_link")
        self.cutoff = float(cfg.params.get("cutoff", 3.0))
        self.deadzone = float(cfg.params.get("deadzone", 0.001))
        self.scale = float(cfg.params.get("scale", 0.005))
        self.active_after = float(cfg.params.get("active_after", 2.0))
        self.dt = float(env.physics_dt)
        self.decimation = int(env.cfg.decimation)
        if abs(env.step_dt - self.dt * self.decimation) > 1e-9:
            raise ValueError("VD9 physics-rate reward requires an integral physics decimation")
        self.asset = env.scene[self.asset_cfg.name]
        base_ids, _ = self.asset.find_bodies(self.base_name)
        if len(base_ids) != 1:
            raise ValueError(f"Expected exactly one {self.base_name!r} body")
        self.base_id = int(base_ids[0])
        self.lowpass = torch.zeros(env.num_envs, 3, device=env.device)
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.total = torch.zeros(env.num_envs, device=env.device)
        self.sample_count = 0
        env._physics_step_callbacks.append(self._sample)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.lowpass[ids] = 0.0
        self.valid[ids] = False
        self.total[ids] = 0.0

    def _sample(self):
        position_w, _ = tcp_pose_w(self.asset)
        base_position_w, base_quaternion_w = link_pose_w(self.asset, self.base_id)
        position_b = quat_apply_inverse(base_quaternion_w, position_w - base_position_w)
        target_b = self._env.command_manager.get_command(self.command_name)[:, :3]
        active = self._env.episode_length_buf * self._env.step_dt >= self.active_after
        penalty, self.lowpass, self.valid = highpass_tracking_penalty(
            position_b - target_b, self.lowpass, self.valid, active, self.dt,
            self.cutoff, self.deadzone, self.scale,
        )
        self.total += penalty
        self.sample_count += 1

    def __call__(self, env, asset_cfg, command_name="ee_target", base_name="base_link",
                 cutoff=3.0, deadzone=0.001, scale=0.005, active_after=2.0):
        if self.sample_count != self.decimation:
            raise RuntimeError(
                f"VD9 TCP reward expected {self.decimation} physics samples, got {self.sample_count}"
            )
        result = self.total / self.sample_count
        self.total.zero_()
        self.sample_count = 0
        return result


class PhysicsRateFootDescent(ManagerTermBase):
    """Track each foot's worst near-ground descent speed within the actor step."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params["asset_cfg"]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.band = float(cfg.params.get("band", 0.12))
        self.speed_limit = float(cfg.params.get("speed_limit", 0.35))
        self.scale = float(cfg.params.get("scale", 0.50))
        self.active_after = float(cfg.params.get("active_after", 2.0))
        self.decimation = int(env.cfg.decimation)
        self.asset = env.scene[self.asset_cfg.name]
        self.sensor = env.scene[self.sensor_cfg.name]
        self.peak = torch.zeros(env.num_envs, len(self.asset_cfg.body_ids), device=env.device)
        self.sample_count = 0
        env._physics_step_callbacks.append(self._sample)

    def reset(self, env_ids=None):
        self.peak[slice(None) if env_ids is None else env_ids] = 0.0

    def _sample(self):
        height = (
            self.asset.data.body_link_pos_w[:, self.asset_cfg.body_ids, 2]
            - self._env.scene.env_origins[:, 2:3] - 0.032
        )
        velocity = self.asset.data.body_link_lin_vel_w[:, self.asset_cfg.body_ids, 2]
        contact = self.sensor.data.net_forces_w[:, self.sensor_cfg.body_ids, 2] > 10.0
        penalty = descending_foot_penalty(
            height, velocity, contact, self.band, self.speed_limit, self.scale
        )
        active = self._env.episode_length_buf * self._env.step_dt >= self.active_after
        self.peak = torch.maximum(self.peak, torch.where(active.unsqueeze(-1), penalty, 0.0))
        self.sample_count += 1

    def __call__(self, env, asset_cfg, sensor_cfg, band=0.12,
                 speed_limit=0.35, scale=0.50, active_after=2.0):
        if self.sample_count != self.decimation:
            raise RuntimeError(
                f"VD9 foot reward expected {self.decimation} physics samples, got {self.sample_count}"
            )
        result = self.peak.mean(dim=-1)
        self.peak.zero_()
        self.sample_count = 0
        return result
