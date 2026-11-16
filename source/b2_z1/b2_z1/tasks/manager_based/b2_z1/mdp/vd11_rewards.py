"""VD11-only 200 Hz reward terms; no privileged actor or physics changes."""

import math

import torch
from isaaclab.managers import ManagerTermBase

from .tcp_frame import link_pose_w, tcp_pose_w
from .vd11_physics import foot_impact_step, landing_sample_counts, world_tcp_highpass_penalty


def _validate_physics_rate(env):
    dt, decimation = float(env.physics_dt), int(env.cfg.decimation)
    if (not math.isclose(dt, 0.005, abs_tol=1.0e-12, rel_tol=0.0)
            or decimation != 4
            or not math.isclose(float(env.step_dt), dt * decimation, abs_tol=1.0e-9)):
        raise ValueError("VD11 rewards require 200 Hz physics and a 50 Hz actor (decimation=4)")
    terrain = getattr(getattr(env.cfg.scene, "terrain", None), "terrain_type", None)
    if terrain != "plane":
        raise ValueError("VD11 rewards require the existing flat plane terrain")
    return dt, decimation


class PhysicsRateWorldTCPHighpass(ManagerTermBase):
    """Two-stage high-pass of real world TCP motion, with lateral emphasis.

    Unlike the VD9 term, neither a moving base nor a held 50 Hz target enters
    the filter input. The current yaw is used only to label left/right after
    filtering. Uniform linear travel has zero asymptotic residual.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.dt, self.decimation = _validate_physics_rate(env)
        self.asset_cfg = cfg.params["asset_cfg"]
        self.base_name = cfg.params.get("base_name", "base_link")
        self.cutoff = float(cfg.params.get("cutoff", 3.0))
        self.deadzone = float(cfg.params.get("deadzone", 0.001))
        self.scale = float(cfg.params.get("scale", 0.005))
        self.lateral_weight = float(cfg.params.get("lateral_weight", 2.0))
        self.active_after = float(cfg.params.get("active_after", 2.0))
        if not math.isfinite(self.active_after) or self.active_after < 0.0:
            raise ValueError("VD11 active_after must be finite and nonnegative")
        self.asset = env.scene[self.asset_cfg.name]
        base_ids, _ = self.asset.find_bodies(self.base_name)
        if len(base_ids) != 1:
            raise ValueError(f"Expected exactly one {self.base_name!r} body")
        self.base_id = int(base_ids[0])
        self.lowpass_position = torch.zeros(env.num_envs, 3, device=env.device)
        self.lowpass_highpass = torch.zeros_like(self.lowpass_position)
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.total = torch.zeros(env.num_envs, device=env.device)
        self.last_highpass_w = torch.zeros_like(self.lowpass_position)
        self.sample_count = 0
        # Validate filter parameters before simulation starts.
        world_tcp_highpass_penalty(
            self.lowpass_position, self.lowpass_position, self.lowpass_highpass,
            self.valid, self.valid, torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device)
            .expand(env.num_envs, -1), self.dt, self.cutoff, self.deadzone,
            self.scale, self.lateral_weight,
        )
        env._physics_step_callbacks.append(self._sample)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.lowpass_position[ids] = 0.0
        self.lowpass_highpass[ids] = 0.0
        self.valid[ids] = False
        self.total[ids] = 0.0
        self.last_highpass_w[ids] = 0.0

    def _sample(self):
        position_w, _ = tcp_pose_w(self.asset)
        # Local world origins reduce float32 cancellation without rotating axes.
        position_w = position_w - self._env.scene.env_origins
        _, base_quaternion_w = link_pose_w(self.asset, self.base_id)
        active = self._env.episode_length_buf * self._env.step_dt >= self.active_after
        (penalty, self.lowpass_position, self.lowpass_highpass,
         self.valid, self.last_highpass_w) = world_tcp_highpass_penalty(
            position_w, self.lowpass_position, self.lowpass_highpass, self.valid,
            active, base_quaternion_w, self.dt, self.cutoff, self.deadzone,
            self.scale, self.lateral_weight,
        )
        self.total += penalty
        self.sample_count += 1

    def __call__(self, env, asset_cfg, base_name="base_link", cutoff=3.0,
                 deadzone=0.001, scale=0.005, lateral_weight=2.0, active_after=2.0):
        if self.sample_count != self.decimation:
            raise RuntimeError(
                f"VD11 TCP reward expected {self.decimation} physics samples, got {self.sample_count}"
            )
        result = self.total / self.sample_count
        self.total.zero_()
        self.sample_count = 0
        return result


class PhysicsRateFootImpact(ManagerTermBase):
    """Real first-contact force, integrated once per measured landing event.

    Force limit and scale are fixed configured values in N, not a physical
    load rating or a termination. VD11 uses a 300 N soft target and 200 N
    scale based on measured touchdown distributions. Heavier randomized
    bases do not receive a more permissive force limit.
    This sensor value is *contact force*, never actual joint drive torque.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.dt, self.decimation = _validate_physics_rate(env)
        self.asset_cfg = cfg.params.get("asset_cfg")
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.force_limit = float(cfg.params.get("force_limit", 804.0))
        self.force_scale = float(cfg.params.get("force_scale", 804.0))
        self.contact_threshold = float(cfg.params.get("contact_threshold", 10.0))
        self.airborne_duration = float(cfg.params.get("airborne_duration", 0.02))
        self.window_s = float(cfg.params.get("window_s", 0.05))
        self.active_after = float(cfg.params.get("active_after", 2.0))
        if not math.isfinite(self.active_after) or self.active_after < 0.0:
            raise ValueError("VD11 active_after must be finite and nonnegative")
        landing_sample_counts(self.dt, self.airborne_duration, self.window_s)
        self.sensor = env.scene[self.sensor_cfg.name]
        if not math.isclose(float(self.sensor.cfg.update_period), self.dt,
                            abs_tol=1.0e-12, rel_tol=0.0):
            raise ValueError("VD11 impact reward requires real contact sensor updates every physics step")
        foot_count = len(self.sensor_cfg.body_ids)
        if foot_count != 4:
            raise ValueError("VD11 impact reward requires the four physical foot sensor bodies")
        if self.asset_cfg is not None:
            asset = env.scene[self.asset_cfg.name]
            asset_names = tuple(asset.body_names[index] for index in self.asset_cfg.body_ids)
            sensor_names = tuple(self.sensor.body_names[index] for index in self.sensor_cfg.body_ids)
            if asset_names != sensor_names:
                raise ValueError("VD11 impact robot feet and contact sensor must have identical name ordering")
        shape = (env.num_envs, foot_count)
        self.airborne_samples = torch.zeros(shape, dtype=torch.long, device=env.device)
        self.window_remaining = torch.zeros_like(self.airborne_samples)
        self.peak_cost = torch.zeros(shape, device=env.device)
        self.total = torch.zeros(env.num_envs, device=env.device)
        self.last_onset = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.sample_count = 0
        # Validate the force parameters now, before the first callback.
        foot_impact_step(
            self.peak_cost, self.airborne_samples, self.window_remaining,
            self.peak_cost, self.dt, self.force_limit, self.force_scale,
            self.contact_threshold, self.airborne_duration, self.window_s,
        )
        env._physics_step_callbacks.append(self._sample)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.airborne_samples[ids] = 0
        self.window_remaining[ids] = 0
        self.peak_cost[ids] = 0.0
        self.total[ids] = 0.0
        self.last_onset[ids] = False

    def _sample(self):
        force_z = self.sensor.data.net_forces_w[:, self.sensor_cfg.body_ids, 2]
        (increment, self.airborne_samples, self.window_remaining,
         self.peak_cost, self.last_onset) = foot_impact_step(
            force_z, self.airborne_samples, self.window_remaining, self.peak_cost,
            self.dt, self.force_limit, self.force_scale, self.contact_threshold,
            self.airborne_duration, self.window_s,
        )
        active = self._env.episode_length_buf * self._env.step_dt >= self.active_after
        self.total += torch.where(active, increment.mean(dim=-1), 0.0)
        self.sample_count += 1

    def __call__(self, env, sensor_cfg, force_limit=804.0, force_scale=804.0,
                 contact_threshold=10.0, airborne_duration=0.02, window_s=0.05,
                 active_after=2.0, asset_cfg=None):
        if self.sample_count != self.decimation:
            raise RuntimeError(
                f"VD11 impact reward expected {self.decimation} physics samples, got {self.sample_count}"
            )
        # RewardManager multiplies by actor dt. Undo that factor for event costs
        # so one impact has the same integrated cost on either side of a boundary.
        result = self.total / float(env.step_dt)
        self.total.zero_()
        self.sample_count = 0
        return result
