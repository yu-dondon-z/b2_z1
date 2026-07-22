# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .distillation import Distillation
from .ppo import PPO

from .him_ppo import HIM_PPO
from .amp_ppo import AMP_PPO


__all__ = ["PPO", "Distillation", "HIM_PPO", "AMP_PPO"]
