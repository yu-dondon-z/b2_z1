"""Dense task error and explicit stage/physical-failure signals; no control."""
import torch
from .vb11_tool_control import huber


def dense_tcp_error(env, command_name="ee_target", scale_m=.08):
    term = env.command_manager.get_term(command_name)
    actual = term.robot.data.body_pos_w[:, term.body_idx]
    return huber((actual - term.tracking_pose_w[:, :3]).norm(dim=-1), scale_m)


def preparation_success(env, command_name="ee_target"):
    return env.command_manager.get_term(command_name)._success_event.float()


def preparation_failure(env, command_name="ee_target"):
    return env.command_manager.get_term(command_name)._failure_event.float()


def physical_failure(env):
    # Timeout is not a physical failure. This is the already computed
    # termination tensor, consumed BEFORE auto-reset, not post-reset state.
    return env.termination_manager.terminated.float()
