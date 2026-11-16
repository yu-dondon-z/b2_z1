"""Torch-only VD11 jitter and genuine landing-event math.

The TCP filter consumes measured world positions, never the 50 Hz held actor
command. Two cascaded causal high-pass stages reject constant position and
constant-velocity drift. Landing input is real vertical contact force in N,
not an estimate of joint drive torque.
"""

import math

import torch


def yaw_frame_components(vector_w, base_quaternion_w):
    """Project a world vector using only the current base yaw (wxyz input).

    Filtering is intentionally performed *before* this projection, so a moving
    base cannot turn a stationary world TCP into a jitter signal.
    """
    w, x, y, z = base_quaternion_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((
        cosine * vector_w[..., 0] + sine * vector_w[..., 1],
        -sine * vector_w[..., 0] + cosine * vector_w[..., 1],
        vector_w[..., 2],
    ), dim=-1)


def world_tcp_highpass_penalty(position_w, lowpass_position, lowpass_highpass,
                               valid, active, base_quaternion_w, dt, cutoff=3.0,
                               deadzone=0.001, scale=0.005, lateral_weight=2.0):
    """Return cost, new filter states, validity and actual world high-pass.

    The first real sample after reset seeds position history and zeroes the
    second stage. Subsequent inactive samples still warm the filter; only the
    cost is gated. This prevents a fresh constant-speed transient at 2 seconds.
    """
    values = (dt, cutoff, deadzone, scale, lateral_weight)
    if (not all(math.isfinite(value) for value in values)
            or dt <= 0.0 or cutoff <= 0.0 or deadzone < 0.0
            or scale <= 0.0 or lateral_weight < 0.0):
        raise ValueError("Require finite positive dt/cutoff/scale and nonnegative deadzone/lateral weight")
    alpha = math.exp(-2.0 * math.pi * cutoff * dt)
    initialized = valid.unsqueeze(-1)
    first = torch.where(initialized,
                        alpha * lowpass_position + (1.0 - alpha) * position_w,
                        position_w)
    first_highpass = position_w - first
    second = torch.where(initialized,
                         alpha * lowpass_highpass + (1.0 - alpha) * first_highpass,
                         torch.zeros_like(first_highpass))
    residual_w = first_highpass - second
    residual_yaw = yaw_frame_components(residual_w, base_quaternion_w)
    cost_components = ((residual_yaw.abs() - deadzone).clamp_min(0.0) / scale).square()
    weights = cost_components.new_tensor((1.0, lateral_weight, 1.0))
    penalty = (cost_components * weights).sum(dim=-1)
    penalty = torch.where(active & valid, penalty, torch.zeros_like(penalty))
    return penalty, first, second, torch.ones_like(valid), residual_w


def landing_sample_counts(dt, airborne_duration=0.02, window_s=0.05):
    """Return required airborne intervals and onset-inclusive sample count.

    At 200 Hz, 20 ms of airborne rearming requires four real intervals. A
    50 ms force window contains onset plus ten following samples (11 points).
    """
    if (not all(math.isfinite(value) for value in (dt, airborne_duration, window_s))
            or dt <= 0.0 or airborne_duration <= 0.0 or window_s < 0.0):
        raise ValueError("Require positive finite dt/airborne duration and nonnegative window")
    return (max(1, math.ceil(airborne_duration / dt - 1.0e-9)),
            max(0, math.ceil(window_s / dt - 1.0e-9)) + 1)


def foot_impact_step(force_z, airborne_samples, window_remaining, peak_cost, dt,
                     force_limit=804.0, force_scale=804.0, contact_threshold=10.0,
                     airborne_duration=0.02, window_s=0.05):
    """Charge only new maxima of one measured, onset-conditioned impact.

    All force tensors have shape (environments, feet) and units N. A currently
    supported foot, including the first sample after reset, is not invented as
    an impact: onset requires actual preceding airborne intervals. Contact
    chatter shorter than ``airborne_duration`` cannot rearm an event.

    Returned increments telescope to each event's final maximum normalized
    squared excess. Caller integrates the increments over actor boundaries,
    dividing by actor dt before RewardManager applies its normal dt factor.
    """
    values = (force_limit, force_scale, contact_threshold)
    if (not all(math.isfinite(value) for value in values)
            or force_limit <= 0.0 or force_scale <= 0.0 or contact_threshold < 0.0):
        raise ValueError("Require positive finite force limit/scale and nonnegative contact threshold")
    required_airborne, window_samples = landing_sample_counts(dt, airborne_duration, window_s)
    contact = force_z > contact_threshold
    onset = contact & (airborne_samples >= required_airborne)
    next_airborne = torch.where(contact, torch.zeros_like(airborne_samples),
                               (airborne_samples + 1).clamp_max(required_airborne))
    remaining = torch.where(onset, torch.full_like(window_remaining, window_samples), window_remaining)
    previous_peak = torch.where(onset, torch.zeros_like(peak_cost), peak_cost)
    candidate = ((force_z - force_limit).clamp_min(0.0) / force_scale).square()
    in_window = remaining > 0
    next_peak = torch.where(in_window, torch.maximum(previous_peak, candidate), previous_peak)
    increment = torch.where(in_window, next_peak - previous_peak, torch.zeros_like(peak_cost))
    return increment, next_airborne, (remaining - 1).clamp_min(0), next_peak, onset
