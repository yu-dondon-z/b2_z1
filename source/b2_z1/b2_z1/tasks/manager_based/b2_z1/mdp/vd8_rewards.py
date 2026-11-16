"""Contact-onset force, avoiding stance load as a surrogate for landing impact."""

import torch
from isaaclab.managers import ManagerTermBase

from .vd8_contact import onset_force_penalty


class LandingOnsetForce(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.following = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.following[:] = False
        else:
            self.following[env_ids] = False

    def __call__(self, env, sensor_cfg, threshold=300., scale=200.):
        sensor = env.scene[sensor_cfg.name]
        count = round(env.step_dt / env.physics_dt)
        history = sensor.data.net_forces_w_history[:, :count + 1, sensor_cfg.body_ids, 2]
        penalty, onset = onset_force_penalty(history, self.following, threshold, scale)
        self.following = onset
        return penalty
