"""Serializable VD2 command curriculum state for training checkpoints."""

import math


AXES = ("pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw")


def _terms(env):
    env = env.unwrapped
    return env, env.command_manager.get_term("base_velocity"), env.command_manager.get_term("ee_target")


def capture_curriculum_state(env):
    env, base, arm = _terms(env)
    return {
        "version": 1,
        "curve_level": float(base.cfg.trajectory_curve_offset_scale),
        "position_ranges": {axis: list(getattr(arm.cfg.ranges, axis)) for axis in AXES},
        "curve_elapsed_steps": int(env.common_step_counter - base._last_curve_curriculum_step),
        "position_elapsed_steps": int(env.common_step_counter - arm._last_pos_curriculum_step),
    }


def restore_curriculum_state(env, state):
    env, base, arm = _terms(env)
    if not isinstance(state, dict) or state.get("version") != 1:
        raise ValueError("Missing or unsupported VD2 curriculum state")
    curve = float(state["curve_level"])
    if not math.isfinite(curve) or not 0 <= curve <= base.cfg.limit_trajectory_curve_offset_scale:
        raise ValueError("VD2 curve level is outside the configured limits")
    ranges = {}
    for axis in AXES:
        values = state["position_ranges"][axis]
        if len(values) != 2:
            raise ValueError(f"Invalid VD2 range for {axis}")
        lo, hi = map(float, values)
        limit_lo, limit_hi = getattr(arm.cfg.limit_ranges, axis)
        if not all(map(math.isfinite, (lo, hi))) or lo > hi or lo < limit_lo - 1e-6 or hi > limit_hi + 1e-6:
            raise ValueError(f"VD2 range for {axis} is outside the configured limits")
        ranges[axis] = [lo, hi]
    elapsed = []
    for key in ("curve_elapsed_steps", "position_elapsed_steps"):
        value = state[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid VD2 curriculum timer {key}")
        elapsed.append(value)
    base.cfg.trajectory_curve_offset_scale = curve
    for axis, values in ranges.items():
        setattr(arm.cfg.ranges, axis, values)
    base._last_curve_curriculum_step = env.common_step_counter - elapsed[0]
    arm._last_pos_curriculum_step = env.common_step_counter - elapsed[1]


def legacy_curriculum_state(env, curve_level, position_steps):
    """Conservatively reconstruct old checkpoints; this is not historical state."""
    if isinstance(position_steps, bool) or not isinstance(position_steps, int) or position_steps < 0:
        raise ValueError("VD2 legacy position steps must be a nonnegative integer")
    env, base, arm = _terms(env)
    state = capture_curriculum_state(env)
    state["curve_level"] = float(curve_level)
    for axis in AXES[:3]:
        lo, hi = state["position_ranges"][axis]
        limit_lo, limit_hi = getattr(arm.cfg.limit_ranges, axis)
        state["position_ranges"][axis] = [max(limit_lo, lo - 0.01 * position_steps), min(limit_hi, hi + 0.01 * position_steps)]
    state["curve_elapsed_steps"] = state["position_elapsed_steps"] = 0
    if not math.isfinite(state["curve_level"]) or not 0 <= state["curve_level"] <= base.cfg.limit_trajectory_curve_offset_scale:
        raise ValueError("VD2 legacy curve level is outside the configured limits")
    return state
