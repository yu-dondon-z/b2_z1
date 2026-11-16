"""Torch-only contact-onset window logic; independent of the simulation app."""

import torch


def onset_force_penalty(force_history, following, threshold=300., scale=200.):
    # Contact history is newest-first; require a genuine crossing at 200 Hz.
    contact = force_history[:, :-1] > 10.
    onset = (contact & (force_history[:, 1:] <= 10.)).any(dim=1)
    peak = force_history[:, :-1].amax(dim=1)
    active = onset | following
    penalty = ((peak - threshold).clamp_min(0.) / scale).square()
    return torch.where(active, penalty, torch.zeros_like(penalty)).amax(dim=-1), onset
