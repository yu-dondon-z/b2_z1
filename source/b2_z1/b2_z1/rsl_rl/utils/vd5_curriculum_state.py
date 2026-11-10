"""VD5 checkpoint state for its ground-target blend curriculum."""

from __future__ import annotations

import math

from .curriculum_state import capture_curriculum_state, restore_curriculum_state


def _command(env):
    return env.unwrapped.command_manager.get_term("ee_target")


def _startup_profile(env):
    env = env.unwrapped
    duration = float(getattr(_command(env).cfg, "vd5_startup_duration_s", 0.0))
    if duration == 0.0:
        return None
    base_cfg = env.command_manager.get_term("base_velocity").cfg
    base_duration = float(base_cfg.vd5_startup_duration_s)
    probability = float(env.cfg.reference_state_initialization_prob)
    base_height = float(getattr(env.cfg, "vd5_nominal_base_height_m", 0.0))
    configured_height = float(env.cfg.scene.robot.init_state.pos[2])
    joint_velocity_range = [float(value) for value in
                            env.cfg.events.reset_robot_joints.params["velocity_range"]]
    if (not math.isfinite(duration) or duration <= 0.0 or base_duration != duration
            or not math.isfinite(probability) or not 0.0 <= probability <= 1.0
            or not math.isfinite(base_height) or base_height <= 0.0
            or configured_height != base_height or joint_velocity_range != [0.0, 0.0]):
        raise ValueError("Invalid VD5 startup profile.")
    return {"version": 1, "duration_s": duration, "reference_initialization_prob": probability,
            "nominal_base_height_m": base_height, "nominal_joint_velocity_range": joint_velocity_range}


def capture_vd5_curriculum_state(env):
    env = env.unwrapped
    command = _command(env)
    blend = float(command._vd5_ground_blend)
    if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("VD5 ground blend is outside [0, 1].")
    state = {
        "version": 1,
        "common_curriculum": capture_curriculum_state(env),
        "ground_blend": blend,
        "ground_transition_steps": int(command.cfg.vd5_ground_transition_steps),
        "seam_trailing_distance_m": float(command.cfg.vd5_seam_trailing_distance),
    }
    profile = _startup_profile(env)
    if profile is not None:
        state["startup_profile"] = profile
    return state


def restore_vd5_curriculum_state(env, state, *, allow_startup_migration=False):
    env = env.unwrapped
    if not isinstance(state, dict) or state.get("version") != 1:
        raise ValueError("Missing or unsupported VD5 curriculum state.")
    expected_profile = _startup_profile(env)
    saved_profile = state.get("startup_profile")
    if saved_profile != expected_profile:
        if not (allow_startup_migration and saved_profile is None and expected_profile is not None):
            raise ValueError("VD5 startup profile differs; use explicit startup migration for the original VD5 checkpoint.")
    blend = state.get("ground_blend")
    if isinstance(blend, bool) or not isinstance(blend, (int, float)):
        raise ValueError("Invalid VD5 ground blend.")
    blend = float(blend)
    if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("VD5 ground blend is outside [0, 1].")
    command = _command(env)
    transition_steps = int(command.cfg.vd5_ground_transition_steps)
    if (state.get("ground_transition_steps") != transition_steps
            or state.get("seam_trailing_distance_m") != float(command.cfg.vd5_seam_trailing_distance)):
        raise ValueError("VD5 checkpoint geometry or transition duration differs from the configuration.")
    if transition_steps < 0 or (transition_steps == 0 and blend < 1.0):
        raise ValueError("VD5 ground blend conflicts with the configured transition.")
    restore_curriculum_state(env, state.get("common_curriculum"))
    command._vd5_ground_blend = blend
    command._vd5_last_ground_curriculum_step = int(env.common_step_counter)
