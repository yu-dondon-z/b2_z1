"""Physics-rate reward math for high-rate TCP jitter and foot descent."""

import math

import torch


def highpass_tracking_penalty(error, lowpass, valid, active, dt, cutoff=3.0,
                              deadzone=0.001, scale=0.005):
    """Causal high-pass penalty on measured TCP tracking error."""
    if dt <= 0.0 or cutoff <= 0.0 or deadzone < 0.0 or scale <= 0.0:
        raise ValueError("Require positive dt/cutoff/scale and nonnegative deadzone")
    alpha = math.exp(-2.0 * math.pi * cutoff * dt)
    ready = valid & active
    updated = torch.where(ready.unsqueeze(-1), alpha * lowpass + (1.0 - alpha) * error, error)
    residual = error - updated
    penalty = ((residual.abs() - deadzone).clamp_min(0.0) / scale).square().sum(dim=-1)
    return penalty, updated, torch.ones_like(valid)


def descending_foot_penalty(height, vertical_velocity, contact, band=0.12,
                            speed_limit=0.35, scale=0.50):
    """Penalize fast descent in the final 12 cm before actual contact."""
    if band <= 0.0 or speed_limit < 0.0 or scale <= 0.0:
        raise ValueError("Require positive band/scale and a nonnegative speed limit")
    near_ground = (height >= -0.01) & (height < band) & ~contact
    excess = (-vertical_velocity - speed_limit).clamp_min(0.0)
    return torch.where(near_ground, (excess / scale).square(), 0.0)
