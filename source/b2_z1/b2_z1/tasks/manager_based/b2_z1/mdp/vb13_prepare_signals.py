"""Loaded standby acquisition signals; never edit actions."""
import torch
from .vb11_tool_control import huber


def final_pose_error(env):
    term = env.command_manager.get_term('ee_target')
    actual = term.robot.data.body_pos_w[:, term.body_idx]
    return huber((actual-term.standby_position_w()).norm(dim=-1), .03)


def clear_hold(env):
    return env.command_manager.get_term('ee_target')._measured_ready().float()
