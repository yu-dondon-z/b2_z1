"""Add discriminator observations to the standard Isaac Lab RSL-RL wrapper."""

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class AMPRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    def get_amp_observations(self) -> torch.Tensor:
        return self.unwrapped.obs_buf["amp_obs"]
