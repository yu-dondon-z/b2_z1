"""VD12-only conservative whole-gun clearance and episode rejection.

This is an explicit geometric constraint, NOT enabled physical self-contact.
The actor and contact sensors are unchanged. No geometry truth is an actor input.
"""

import torch
from isaaclab.managers import ManagerTermBase

from b2_z1.assets.vd12_geometry import GunRobotClearance
from b2_z1.assets.vd12_guard_state import GeometryWindow


class _GunGuard:
    def __init__(self, env):
        if env.physics_dt != 0.005 or env.cfg.decimation != 4:
            raise ValueError("VD12 geometry requires 200 Hz physics / 50 Hz actor")
        self.env = env
        self.profile = dict(env.cfg.vd12_geometry_profile)
        self.robot = env.scene["robot"]
        self.geometry = GunRobotClearance(
            env.cfg.scene.robot.spawn.asset_path, self.robot.body_names,
            device=env.device,
            numerical_guard_m=self.profile["numerical_guard_m"],
            minimum_quaternion_norm=self.profile["minimum_quaternion_norm"],
        )
        if self.geometry.metadata["source_urdf_sha256"] != self.profile["urdf_sha256"]:
            raise ValueError("VD12 runtime geometry differs from the saved source contract")
        self.window = GeometryWindow(
            env.num_envs, device=env.device, decimation=env.cfg.decimation,
            soft_margin_m=self.profile["soft_margin_m"],
            hard_threshold_m=self.profile["hard_threshold_m"], cost_cap=self.profile["cost_cap"],
        )
        self.last_result = None
        env._physics_step_callbacks.append(self.sample)

    def sample(self):
        # These are measured body link poses, not COM poses, commanded FK or
        # PD-estimated torque. Fixed child colliders are folded by the parser.
        self.last_result = self.geometry(
            self.robot.data.body_pos_w, self.robot.data.body_quat_w,
        )
        self.window.sample(
            self.last_result["minimum_clearance_lower_bound_m"],
            self.last_result["input_valid"], window_id=self.env.common_step_counter,
            physics_step=self.env._sim_step_counter,
        )


def geometry_guard(env):
    guard = getattr(env, "_vd12_gun_guard", None)
    if guard is None:
        guard = env._vd12_gun_guard = _GunGuard(env)
    elif guard.profile != env.cfg.vd12_geometry_profile:
        raise ValueError("VD12 geometry contract changed after construction")
    return guard


class GunClearanceTermination(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.guard = geometry_guard(env)

    def reset(self, env_ids=None):
        self.guard.window.reset(env_ids)

    def __call__(self, env):
        self.guard.window.require_complete(env._sim_step_counter)
        return self.guard.window.violated.clone()


class GunClearancePenalty(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.guard = geometry_guard(env)

    def reset(self, env_ids=None):
        self.guard.window.reset(env_ids)

    def __call__(self, env):
        return self.guard.window.mean_cost(env._sim_step_counter)


def gun_exclusion_failure(env):
    """One episode-ending cost; does not label the exclusion as physical contact."""
    guard = geometry_guard(env)
    guard.window.require_complete(env._sim_step_counter)
    return guard.window.violated.to(dtype=torch.float32)
