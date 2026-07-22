# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural networks."""

from .memory import Memory

from .amp_discriminator import AMP_Discriminator

__all__ = ["Memory", "AMP_Discriminator"]
