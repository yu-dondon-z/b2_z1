"""Finite-duration physical base roll disturbances for VD8 training only."""

import torch
from isaaclab.managers import ManagerTermBase


class RollPulse(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.next_step = torch.randint(round(4 / env.step_dt), round(8 / env.step_dt) + 1,
                                       (env.num_envs,), device=env.device)
        self.remaining = torch.zeros_like(self.next_step)
        self.torque = torch.zeros(env.num_envs, device=env.device)
        self.zero = torch.zeros(env.num_envs, 1, 3, device=env.device)
        self._announced = False

    def __call__(self, env, env_ids, asset_cfg, min_torque=4., max_torque=10., duration_s=.12):
        if env_ids is not None or min_torque <= 0 or max_torque < min_torque:
            raise ValueError('RollPulse requires global interval and positive bounded torque')
        step = env.step_dt
        if abs(duration_s / step - round(duration_s / step)) > 1e-5:
            raise ValueError('Pulse duration must align to actor period')
        age = env.episode_length_buf
        reset = age == 0
        self.remaining[reset] = 0
        self.torque[reset] = 0
        reset_count = int(reset.sum())
        if reset_count:
            self.next_step[reset] = torch.randint(round(4 / step), round(8 / step) + 1,
                                                  (reset_count,), device=env.device)
        start = (age >= self.next_step) & ~reset
        count = int(start.sum())
        if count:
            if not self._announced:
                print(f'[VD8] First physical roll pulse: {count} environments, {duration_s:.2f} s, '
                      f'{min_torque:.1f}..{max_torque:.1f} Nm, body-local x', flush=True)
                self._announced = True
            self.next_step[start] = age[start] + torch.randint(round(5 / step), round(9 / step) + 1,
                                                                 (count,), device=env.device)
            self.remaining[start] = round(duration_s / step)
            magnitude = min_torque + (max_torque - min_torque) * torch.rand(count, device=env.device)
            sign = torch.where(torch.rand(count, device=env.device) < .5, -1., 1.)
            self.torque[start] = magnitude * sign
        active = self.remaining > 0
        torques = self.zero.clone()
        torques[:, 0, 0] = torch.where(active, self.torque, 0.)
        env.scene[asset_cfg.name].set_external_force_and_torque(
            self.zero, torques, body_ids=asset_cfg.body_ids, is_global=False)
        self.remaining[active] -= 1
