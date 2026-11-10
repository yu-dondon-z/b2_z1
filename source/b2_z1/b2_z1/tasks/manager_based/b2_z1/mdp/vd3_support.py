"""Keep the massless welding gun from becoming an extra support leg.

The gun collision is part of ``link06`` and can transmit ground reaction forces
despite adding no payload mass. Small working contacts are permitted; prolonged
upward loading and large impacts are detected separately. These terms retain no
episode state: IsaacLab clears contact history on reset.

Only torch is needed to import this module, so the force and tilt calculations
can also be checked without starting Isaac Sim.
"""

from __future__ import annotations

import math
from numbers import Real

import torch


__all__ = [
    "upward_support_diagnostics", "upward_support_force_penalty",
    "inclination_deadzone_penalty", "tool_support_diagnostics",
    "tool_support_force", "tool_support_termination", "base_tilt_penalty",
]


def _finite_parameter(name: str, value: float, *, allow_zero: bool = False) -> None:
    if (
        isinstance(value, bool) or not isinstance(value, Real)
        or not math.isfinite(value) or value < 0.0
        or (not allow_zero and value == 0.0)
    ):
        comparison = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be finite and {comparison}.")


def _upward_forces(force_history_w: torch.Tensor) -> torch.Tensor:
    if force_history_w.ndim != 3 or force_history_w.shape[-1] != 3 or force_history_w.shape[1] == 0:
        raise ValueError("Expected nonempty force history with shape (num_envs, history_steps, 3).")
    return force_history_w[..., 2].clamp_min(0.0)


def upward_support_diagnostics(
    force_history_w: torch.Tensor, sustained_force: float = 20.0, peak_force: float = 150.0,
) -> dict[str, torch.Tensor]:
    """Summarize upward reaction force for one body in an already sized window.

    Each force sample represents one physics integration interval. Sustained
    support requires *every* sample to exceed its threshold; zero-filled reset
    history consequently cannot trigger it before a full window has elapsed.
    A peak over the impact threshold is intentionally detected immediately.
    Horizontal and downward forces do not count as supporting body weight.
    """
    _finite_parameter("sustained_force", sustained_force)
    _finite_parameter("peak_force", peak_force)
    if peak_force < sustained_force:
        raise ValueError("peak_force must be >= sustained_force.")
    upward = _upward_forces(force_history_w)
    peak = upward.amax(dim=1)
    minimum = upward.amin(dim=1)
    sustained = minimum > sustained_force
    impact = peak > peak_force
    return {
        "peak_upward_force": peak,
        "min_upward_force": minimum,
        "mean_upward_force": upward.mean(dim=1),
        "support_sample_fraction": (upward > sustained_force).to(upward.dtype).mean(dim=1),
        "sustained_support": sustained,
        "peak_support": impact,
        "tool_support": sustained | impact,
    }


def upward_support_force_penalty(
    force_history_w: torch.Tensor, deadzone: float = 5.0, force_scale: float = 100.0,
) -> torch.Tensor:
    """Return nonnegative linear excess upward load, without upper saturation."""
    _finite_parameter("deadzone", deadzone, allow_zero=True)
    _finite_parameter("force_scale", force_scale)
    peak = _upward_forces(force_history_w).amax(dim=1)
    return (peak - deadzone).clamp_min(0.0) / force_scale


def inclination_deadzone_penalty(
    projected_gravity_b: torch.Tensor, deadzone: float = 0.1, scale: float = 0.2,
) -> torch.Tensor:
    """Squared excess inclination normalized by a radian scale, without a cap."""
    _finite_parameter("deadzone", deadzone, allow_zero=True)
    _finite_parameter("scale", scale)
    if deadzone >= math.pi:
        raise ValueError("deadzone must be < pi radians.")
    inclination = _inclination(projected_gravity_b)
    return ((inclination - deadzone).clamp_min(0.0) / scale).square()


def _inclination(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    if projected_gravity_b.ndim != 2 or projected_gravity_b.shape[-1] != 3:
        raise ValueError("Expected projected gravity with shape (num_envs, 3).")
    return torch.acos((-projected_gravity_b[:, 2]).clamp(-1.0, 1.0))


def _record_metrics(env, metrics: dict[str, torch.Tensor]) -> None:
    command_manager = getattr(env, "command_manager", None)
    if command_manager is not None:
        command_manager.get_term("ee_target").metrics.update(metrics)


def _tool_force_history(env, sensor_cfg, duration_s: float) -> torch.Tensor:
    """Select link06 and require enough samples at the actual physics cadence."""
    _finite_parameter("duration_s", duration_s)
    physics_dt = env.cfg.sim.dt
    _finite_parameter("physics_dt", physics_dt)
    sensor = env.scene.sensors[sensor_cfg.name]
    update_period = sensor.cfg.update_period
    _finite_parameter("sensor update_period", update_period, allow_zero=True)
    if update_period != 0.0 and not math.isclose(update_period, physics_dt, rel_tol=1.e-6, abs_tol=1.e-9):
        raise ValueError("Tool support history must update every physics step (update_period=sim.dt or 0).")

    body_ids = sensor_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(len(sensor.body_names)))[body_ids]
    elif isinstance(body_ids, int):
        body_ids = [body_ids]
    else:
        body_ids = list(body_ids)
    if len(body_ids) != 1 or sensor.body_names[body_ids[0]] != "link06":
        raise ValueError("Tool support requires sensor_cfg to select exactly the merged gun body 'link06'.")

    # ContactSensor history is newest first and is forcibly refreshed on each
    # physics update whenever history_length > 0, including lazy sensor scenes.
    # Twenty integration intervals at 0.005 s therefore cover 0.10 s.
    history_steps = max(1, math.ceil(duration_s / physics_dt - 1.e-9))
    history = sensor.data.net_forces_w_history
    if history.ndim != 4 or history.shape[-1] != 3 or history.shape[1] < history_steps:
        raise ValueError(f"Tool support requires at least {history_steps} physics samples for {duration_s:g} s.")
    return history[:, :history_steps, body_ids[0], :]


def tool_support_diagnostics(
    env, sensor_cfg, sustained_force: float = 20.0, peak_force: float = 150.0, duration_s: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Return force levels and support flags using reset-safe sensor history."""
    return upward_support_diagnostics(
        _tool_force_history(env, sensor_cfg, duration_s), sustained_force, peak_force,
    )


def tool_support_force(
    env, sensor_cfg, deadzone: float = 5.0, force_scale: float = 100.0, duration_s: float = 0.1,
) -> torch.Tensor:
    """Negative-weight reward term and 20 N / 150 N support diagnostics."""
    history = _tool_force_history(env, sensor_cfg, duration_s)
    penalty = upward_support_force_penalty(history, deadzone, force_scale)
    diagnostics = upward_support_diagnostics(history)
    _record_metrics(env, {
        "tool_upward_force_peak": diagnostics["peak_upward_force"],
        "tool_upward_force_mean": diagnostics["mean_upward_force"],
        "tool_support_active": diagnostics["tool_support"].to(history.dtype),
    })
    return penalty


def tool_support_termination(
    env, sensor_cfg, sustained_force: float = 20.0, peak_force: float = 150.0, duration_s: float = 0.1,
) -> torch.Tensor:
    """End an episode for sustained tool loading or a large upward impact."""
    return tool_support_diagnostics(env, sensor_cfg, sustained_force, peak_force, duration_s)["tool_support"]


def base_tilt_penalty(env, asset_cfg=None, deadzone: float = 0.1, scale: float = 0.2) -> torch.Tensor:
    """Negative-weight reward term for leaning beyond a small allowed angle."""
    asset = env.scene["robot" if asset_cfg is None else asset_cfg.name]
    gravity = asset.data.projected_gravity_b
    penalty = inclination_deadzone_penalty(gravity, deadzone, scale)
    _record_metrics(env, {"base_inclination_rad": _inclination(gravity)})
    return penalty
