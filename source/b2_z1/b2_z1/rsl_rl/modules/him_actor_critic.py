# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

from .him_estimator import HIMEstimator

from .actor_critic import ActorCritic


class HIM_ActorCritic(ActorCritic):
    is_recurrent = False
    
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_one_step_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        # super().__init__(
        #     num_actor_obs=num_actor_obs,
        #     num_critic_obs=num_critic_obs,
        #     num_actions=num_actions,
        #     actor_hidden_dims=actor_hidden_dims,
        #     critic_hidden_dims=critic_hidden_dims,
        #     activation=activation,
        #     init_noise_std=init_noise_std,
        #     noise_std_type=noise_std_type,
        #     **kwargs,
        # )
        
        nn.Module.__init__(self)
        activation = resolve_nn_activation(activation)
        
        self.history_size = int(num_actor_obs/num_one_step_obs)
        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions
        self.num_one_step_obs = num_one_step_obs
        

        mlp_input_dim_a = num_one_step_obs + 3 + 16
        mlp_input_dim_c = num_critic_obs
        
        # Estimator
        self.estimator = HIMEstimator(temporal_steps=self.history_size, num_one_step_obs=num_one_step_obs)
        
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f'Estimator: {self.estimator.encoder}')

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)




    def update_distribution(self, obs_history):
        with torch.no_grad():
            vel, latent = self.estimator(obs_history)
        actor_input = torch.cat((obs_history[:,:self.num_one_step_obs], vel, latent), dim=-1)

        # Check for NaN/Inf in inputs
        if torch.isnan(actor_input).any() or torch.isinf(actor_input).any():
            print(f"[ERROR] NaN/Inf detected in actor_input before normalization!")
            print(f"  - obs_history stats: min={obs_history.min():.4f}, max={obs_history.max():.4f}, mean={obs_history.mean():.4f}, has_nan={torch.isnan(obs_history).any()}")
            print(f"  - vel stats: min={vel.min():.4f}, max={vel.max():.4f}, mean={vel.mean():.4f}, has_nan={torch.isnan(vel).any()}")
            print(f"  - latent stats: min={latent.min():.4f}, max={latent.max():.4f}, mean={latent.mean():.4f}, has_nan={torch.isnan(latent).any()}")
            raise ValueError("NaN/Inf in actor_input before normalization")
        
        mean = self.actor(actor_input)
        
        # Check for NaN/Inf in actor output
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            print(f"[ERROR] NaN/Inf in actor output (mean)!")
            print(f"  - actor_input stats: min={actor_input.min():.4f}, max={actor_input.max():.4f}, mean={actor_input.mean():.4f}")
            print(f"  - Batch size: {actor_input.shape[0]}")

            # Check actor network weights AFTER forward (they should still be OK)
            for name, param in self.actor.named_parameters():
                if torch.isnan(param).any():
                    print(f"  - NaN in actor weight: {name}")
                if torch.isinf(param).any():
                    print(f"  - Inf in actor weight: {name}")
            
            raise ValueError("NaN or Inf detected in actor network output!")
        
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)
        


    def act_inference(self, obs_history, observations=None):
        vel, latent = self.estimator(obs_history)
        actions_mean = self.actor(torch.cat((obs_history[:,:self.num_one_step_obs], vel, latent), dim=-1))
        return actions_mean
