"""Causal training-only high-pass of measured TCP tracking error, in metres."""
import math
import torch


def filter_error(error, lowpass, valid, active, dt, cutoff=3., deadzone=.002, scale=.01):
    """Reset/inactive samples seed history; never penalize episode teleports.

    Subtracting commanded position before filtering exempts perfectly tracked
    motion. A stationary offset decays out instead of duplicating accuracy reward.
    """
    alpha = math.exp(-2*math.pi*cutoff*dt)
    ready = valid & active
    updated = torch.where(ready[:,None], alpha*lowpass+(1-alpha)*error, error)
    residual = error-updated
    penalty = ((residual.abs()-deadzone).clamp_min(0)/scale).square().sum(-1)
    return penalty, updated, torch.ones_like(valid)
