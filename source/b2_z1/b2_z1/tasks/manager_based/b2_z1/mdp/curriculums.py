# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _safe_update_range(
    target_obj, limit_obj, attr_name: str, delta: torch.Tensor, device: torch.device
):
    """Helper: safely update a config range while clamping to limit_obj thresholds."""
    if hasattr(target_obj, attr_name) and hasattr(limit_obj, attr_name):
        curr_val = torch.tensor(getattr(target_obj, attr_name), device=device)
        limits = getattr(limit_obj, attr_name)
        new_val = torch.clamp(curr_val + delta, limits[0], limits[1])
        setattr(target_obj, attr_name, new_val.tolist())


def ang_vel_cmd_levels(
    env,  # : ManagerBasedRLEnv
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    cfg = env.command_manager.get_term("base_velocity").cfg

    # 1. Safely get attribute (compatible with typos)
    is_curriculum_enabled = getattr(
        cfg, "curriculum_enabled", getattr(cfg, "currirulum_enabled", False)
    )
    limit_ranges = getattr(cfg, "limit_ranges", None)
    ranges = getattr(cfg, "ranges", None)

    # 2. Handle disabled curriculum
    if not is_curriculum_enabled:
        # When disabled, force ranges to limit_ranges (only needs to happen once, or rely on init)
        if ranges is not None and limit_ranges is not None:
            cfg.ranges = limit_ranges
        # Don't return -1.0; return actual limit value for log consistency
        val = (
            limit_ranges.ang_vel_z[1]
            if limit_ranges and hasattr(limit_ranges, "ang_vel_z")
            else 0.0
        )
        return torch.tensor(-1.0, device=env.device)

    # Basic safety check
    if ranges is None or limit_ranges is None or not hasattr(ranges, "ang_vel_z"):
        return torch.tensor(0.0, device=env.device)

    # 3. Performance: only compute reward and update difficulty at episode end
    if env.common_step_counter % (env.max_episode_length) == 0:
        reward_sums = env.reward_manager._episode_sums.get(reward_term_name)
        if reward_sums is not None:
            reward = torch.mean(reward_sums[env_ids]) / env.max_episode_length_s
            reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

            if reward > reward_term_cfg.weight * 0.8:
                delta_command = torch.tensor([-0.05, 0.05], device=env.device)
                _safe_update_range(
                    ranges, limit_ranges, "ang_vel_z", delta_command, env.device
                )

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def lin_vel_cmd_levels(
    env,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    cfg = env.command_manager.get_term("base_velocity").cfg

    is_curriculum_enabled = getattr(
        cfg, "curriculum_enabled", getattr(cfg, "currirulum_enabled", False)
    )
    ranges = getattr(cfg, "ranges", None)
    limit_ranges = getattr(cfg, "limit_ranges", None)
    force_range = getattr(cfg, "force_range", None)
    limit_force_range = getattr(cfg, "limit_force_range", None)

    # --- Handle disabled curriculum ---
    if not is_curriculum_enabled:
        # Fix bug: original directly assigned ranges = limit_ranges (only assigned local variable)
        if ranges is not None and limit_ranges is not None:
            cfg.ranges = limit_ranges
        val = (
            limit_ranges.lin_vel_x[1]
            if limit_ranges and hasattr(limit_ranges, "lin_vel_x")
            else 0.0
        )
        return torch.tensor(-1.0, device=env.device)

    if ranges is None or not hasattr(ranges, "lin_vel_x"):
        return torch.tensor(0.0, device=env.device)

    # --- Performance: only compute at episode end ---
    if (
        env.common_step_counter > 0
        and env.common_step_counter % (env.max_episode_length) == 0
    ):
        reward_sums = env.reward_manager._episode_sums.get(reward_term_name)
        if reward_sums is not None:
            reward = torch.mean(reward_sums[env_ids]) / env.max_episode_length_s
            reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

            if reward > reward_term_cfg.weight * 0.8:
                # print(f"Episode ended. Reward: {reward:.4f}, Threshold: {reward_term_cfg.weight * 0.8:.4f}")

                # -- Update velocity range --
                if limit_ranges is not None:
                    delta_vel = torch.tensor([-0.05, 0.05], device=env.device)
                    _safe_update_range(
                        ranges, limit_ranges, "lin_vel_x", delta_vel, env.device
                    )
                    _safe_update_range(
                        ranges, limit_ranges, "lin_vel_y", delta_vel, env.device
                    )

                # -- Update force range --
                if force_range is not None and limit_force_range is not None:
                    delta_force = torch.tensor([-5.0, 5.0], device=env.device)
                    _safe_update_range(
                        force_range,
                        limit_force_range,
                        "lin_force_x",
                        delta_force,
                        env.device,
                    )
                    _safe_update_range(
                        force_range,
                        limit_force_range,
                        "lin_force_y",
                        delta_force,
                        env.device,
                    )

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def pos_cmd_levels(
    env,
    env_ids: Sequence[int],
    reward_term_name: str = "end_effector_position_tracking_exp",
) -> torch.Tensor:
    cfg = env.command_manager.get_term("ee_target").cfg

    is_curriculum_enabled = getattr(
        cfg, "curriculum_enabled", getattr(cfg, "currirulum_enabled", False)
    )
    ranges = getattr(cfg, "ranges", None)
    limit_ranges = getattr(cfg, "limit_ranges", None)
    force_range = getattr(cfg, "force_range", None)
    limit_force_range = getattr(cfg, "limit_force_range", None)

    # --- Handle disabled curriculum ---
    if not is_curriculum_enabled:
        if ranges is not None and limit_ranges is not None:
            cfg.ranges = limit_ranges
        val = (
            limit_ranges.pos_x[1]
            if limit_ranges and hasattr(limit_ranges, "pos_x")
            else 0.0
        )
        return torch.tensor(-1.0, device=env.device)

    if ranges is None or not hasattr(ranges, "pos_x"):
        return torch.tensor(0.0, device=env.device)

    # --- Performance: only compute at episode end ---
    if (
        env.common_step_counter > 0
        and env.common_step_counter % (env.max_episode_length) == 0
        
    ):
        reward_sums = env.reward_manager._episode_sums.get(reward_term_name)
        if reward_sums is not None:
            reward = torch.mean(reward_sums[env_ids]) / env.max_episode_length_s
            reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

            if reward > reward_term_cfg.weight * 0.8:
                pos_delta = torch.tensor([-0.05, 0.05], device=env.device)
                ori_delta = torch.tensor([-3.14 / 18, 3.14 / 18], device=env.device)
                force_delta = torch.tensor([-5.0, 5.0], device=env.device)

                # Update position ranges
                if limit_ranges is not None:
                    for axis in ["pos_x", "pos_y", "pos_z"]:
                        _safe_update_range(
                            ranges, limit_ranges, axis, pos_delta, env.device
                        )
                    for axis in ["roll", "pitch", "yaw"]:
                        _safe_update_range(
                            ranges, limit_ranges, axis, ori_delta, env.device
                        )

                # Update force ranges
                if force_range is not None and limit_force_range is not None:
                    for axis in ["lin_force_x", "lin_force_y", "lin_force_z"]:
                        _safe_update_range(
                            force_range,
                            limit_force_range,
                            axis,
                            force_delta,
                            env.device,
                        )

    return torch.tensor(ranges.pos_x[1], device=env.device)




def vel_cmd_to_ee_pos_levels(
    env,
    env_ids: Sequence[int],
    reward_threshold: float = 0.8,
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    # cfg = env.command_manager.get_term("base_velocity").cfg
    
    cfg = env.command_manager.get_term("ee_target").cfg

    is_curriculum_enabled = getattr(
        cfg, "curriculum_enabled", getattr(cfg, "currirulum_enabled", False)
    )
    ranges = getattr(cfg, "ranges", None)
    limit_ranges = getattr(cfg, "limit_ranges", None)
    force_range = getattr(cfg, "force_range", None)
    limit_force_range = getattr(cfg, "limit_force_range", None)

    # --- Handle disabled curriculum ---
    if not is_curriculum_enabled:
        if ranges is not None and limit_ranges is not None:
            cfg.ranges = limit_ranges
        val = (
            limit_ranges.pos_x[1]
            if limit_ranges and hasattr(limit_ranges, "pos_x")
            else 0.0
        )
        return torch.tensor(-1.0, device=env.device)

    if ranges is None or not hasattr(ranges, "pos_x"):
        return torch.tensor(0.0, device=env.device)

    # --- Performance: only compute at episode end ---
    if (
        env.common_step_counter > 0
        and env.common_step_counter % (env.max_episode_length) == 0
    ):
        reward_sums = env.reward_manager._episode_sums.get(reward_term_name)
        if reward_sums is not None:
            reward = torch.mean(reward_sums[env_ids]) / env.max_episode_length_s
            reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

            if reward > reward_term_cfg.weight * reward_threshold:

                #  -- update ee pos level 
                pos_delta = torch.tensor([-0.05, 0.05], device=env.device)

                # Update position ranges
                if limit_ranges is not None:
                    for axis in ["pos_x", "pos_y", "pos_z"]:
                        _safe_update_range(
                            ranges, limit_ranges, axis, pos_delta, env.device
                        )
            
    return torch.tensor(ranges.pos_x[1], device=env.device)


def truth_curve_levels(
    env,
    env_ids: Sequence[int],
    command_name: str = "ee_target",
    reward_term_names: tuple[str, ...] = (
        "track_lin_vel_xy",
        "track_ang_vel_z",
        "end_effector_position_tracking_exp",
    ),
    reward_thresholds: tuple[float, ...] = (0.60, 0.35, 0.50),
    straight_path_fractions: tuple[float, ...] = (0.80, 0.65, 0.50, 0.40),
    arc_radius_ranges_m: tuple[tuple[float, float], ...] = (
        (16.0, 24.0),
        (12.0, 20.0),
        (9.0, 16.0),
        (6.0, 12.0),
    ),
    speed_ranges_mps: tuple[tuple[float, float], ...] = (
        (0.15, 0.26),
        (0.16, 0.30),
        (0.18, 0.34),
        (0.20, 0.38),
    ),
    max_yaw_rates_rps: tuple[float, ...] = (0.10, 0.13, 0.16, 0.18),
    cross_track_gains: tuple[float, ...] = (0.35, 0.40, 0.45, 0.50),
    path_heading_gains: tuple[float, ...] = (0.75, 0.80, 0.85, 0.90),
    minimum_evaluation_episodes: int = 1024,
    successful_evaluations_required: int = 2,
    max_terminated_fraction: float | None = None,
    min_mean_episode_length_fraction: float | None = None,
    max_terminated_fractions: tuple[float, ...] | None = None,
    min_mean_episode_length_fractions: tuple[float, ...] | None = None,
    emit_diagnostics: bool = False,
) -> torch.Tensor | dict[str, float]:
    """Performance-gated curriculum for world-fixed truth curves.

    The original ``ee_drived`` curriculum expands the end-effector sampling
    box after velocity tracking passes a threshold.  The current rear-facing
    task already has a more deployment-like, world-fixed crack generator, so
    this adaptation increases the *curve* difficulty instead: fewer straight
    paths, tighter left/right arcs, a wider speed range and stronger steering.

    Curriculum updates happen only from completed episodes.  Requiring a
    large batch and consecutive successful evaluations prevents a few easy or
    prematurely terminated environments from advancing every other reset.
    """

    if len(reward_term_names) != len(reward_thresholds):
        raise ValueError("reward_term_names and reward_thresholds must have equal length")
    if max_terminated_fraction is not None and not 0.0 <= max_terminated_fraction <= 1.0:
        raise ValueError("max_terminated_fraction must be in [0, 1]")
    if min_mean_episode_length_fraction is not None and not 0.0 <= min_mean_episode_length_fraction <= 1.0:
        raise ValueError("min_mean_episode_length_fraction must be in [0, 1]")

    stage_count = len(straight_path_fractions)
    stage_sequences = (
        arc_radius_ranges_m,
        speed_ranges_mps,
        max_yaw_rates_rps,
        cross_track_gains,
        path_heading_gains,
    )
    if stage_count == 0 or any(len(values) != stage_count for values in stage_sequences):
        raise ValueError("all truth-curve curriculum stage sequences must have equal length")
    for values in (max_terminated_fractions, min_mean_episode_length_fractions):
        if values is not None and (len(values) != stage_count or any(not 0.0 <= value <= 1.0 for value in values)):
            raise ValueError("stage safety thresholds must match stage count and lie in [0, 1]")
    if minimum_evaluation_episodes < 1 or successful_evaluations_required < 1:
        raise ValueError("curriculum evaluation counts must be positive")

    command_term = env.command_manager.get_term(command_name)
    cfg = command_term.cfg
    if (
        getattr(cfg, "mode", None) != "synthetic"
        or not getattr(cfg, "synthetic_world_fixed_path", False)
        or getattr(cfg, "synthetic_path_family", None) != "straight_arc"
    ):
        return torch.tensor(-1.0, device=env.device)

    if not hasattr(command_term, "_truth_curve_curriculum_level"):
        command_term._truth_curve_curriculum_level = 0
        command_term._truth_curve_successful_evaluations = 0
        command_term._truth_curve_evaluation_episodes = 0
        command_term._truth_curve_reward_totals = {
            name: 0.0 for name in reward_term_names
        }
        command_term._truth_curve_terminated_count = 0.0
        command_term._truth_curve_episode_steps = 0.0
        command_term._truth_curve_diagnostics = {}

    level = command_term._truth_curve_curriculum_level
    if max_terminated_fractions is not None:
        max_terminated_fraction = max_terminated_fractions[level]
    if min_mean_episode_length_fractions is not None:
        min_mean_episode_length_fraction = min_mean_episode_length_fractions[level]

    # Environment construction performs an initial reset before any reward is
    # collected.  Do not let that all-zero batch count as a failed evaluation.
    if env.common_step_counter > 0 and len(env_ids) > 0:
        completed = len(env_ids)
        for reward_name in reward_term_names:
            reward_sums = env.reward_manager._episode_sums.get(reward_name)
            if reward_sums is None:
                raise KeyError(
                    f"truth-curve curriculum reward term {reward_name!r} is not active"
                )
            normalized_returns = reward_sums[env_ids] / env.max_episode_length_s
            command_term._truth_curve_reward_totals[reward_name] += float(
                normalized_returns.sum().item()
            )
        command_term._truth_curve_evaluation_episodes += completed
        if max_terminated_fraction is not None:
            command_term._truth_curve_terminated_count += float(
                env.termination_manager.terminated[env_ids].float().sum().item()
            )
        if min_mean_episode_length_fraction is not None:
            command_term._truth_curve_episode_steps += float(
                env.episode_length_buf[env_ids].float().sum().item()
            )

    if (
        command_term._truth_curve_evaluation_episodes
        >= minimum_evaluation_episodes
    ):
        episode_count = command_term._truth_curve_evaluation_episodes
        passed = True
        diagnostics = {"evaluated_level": float(level), "evaluated_episodes": float(episode_count)}
        blocked_rewards = []
        for reward_name, threshold_fraction in zip(
            reward_term_names, reward_thresholds
        ):
            mean_return = (
                command_term._truth_curve_reward_totals[reward_name]
                / episode_count
            )
            reward_cfg = env.reward_manager.get_term_cfg(reward_name)
            reward_passed = mean_return > reward_cfg.weight * threshold_fraction
            passed = passed and reward_passed
            diagnostics[f"margin_{reward_name}"] = (
                mean_return - reward_cfg.weight * threshold_fraction
            ) / max(abs(reward_cfg.weight), 1.0e-8)
            if not reward_passed:
                blocked_rewards.append(reward_name)

        if max_terminated_fraction is not None:
            failure_fraction = command_term._truth_curve_terminated_count / episode_count
            diagnostics["terminated_fraction"] = failure_fraction
            diagnostics["max_terminated_fraction"] = max_terminated_fraction
            diagnostics["survival_passed"] = float(failure_fraction <= max_terminated_fraction)
            passed = passed and failure_fraction <= max_terminated_fraction
        if min_mean_episode_length_fraction is not None:
            mean_length_fraction = (
                command_term._truth_curve_episode_steps
                / (episode_count * env.max_episode_length)
            )
            passed = passed and mean_length_fraction >= min_mean_episode_length_fraction
            diagnostics["mean_length_fraction"] = mean_length_fraction
            diagnostics["min_length_fraction"] = min_mean_episode_length_fraction
            diagnostics["length_passed"] = float(mean_length_fraction >= min_mean_episode_length_fraction)

        diagnostics["blocked_reward_count"] = float(len(blocked_rewards))
        diagnostics["batch_passed"] = float(passed)
        command_term._truth_curve_diagnostics = diagnostics

        if passed:
            command_term._truth_curve_successful_evaluations += 1
        else:
            command_term._truth_curve_successful_evaluations = 0

        current_level = command_term._truth_curve_curriculum_level
        if (
            command_term._truth_curve_successful_evaluations
            >= successful_evaluations_required
            and current_level < stage_count - 1
        ):
            current_level += 1
            command_term._truth_curve_curriculum_level = current_level
            command_term._truth_curve_successful_evaluations = 0
            cfg.synthetic_straight_path_fraction = straight_path_fractions[current_level]
            cfg.synthetic_arc_radius_range_m = arc_radius_ranges_m[current_level]
            cfg.synthetic_speed_range_mps = speed_ranges_mps[current_level]
            cfg.max_yaw_rate_rps = max_yaw_rates_rps[current_level]
            cfg.cross_track_gain = cross_track_gains[current_level]
            cfg.path_heading_gain = path_heading_gains[current_level]
            if emit_diagnostics:
                print(f"[Curriculum] truth curve promoted {level} -> {current_level}: radius={cfg.synthetic_arc_radius_range_m}, speed={cfg.synthetic_speed_range_mps}")

        if emit_diagnostics and not passed:
            print(
                f"[Curriculum] level={level} held; blocked_rewards={blocked_rewards}; "
                f"failure={diagnostics.get('terminated_fraction', 0.0):.3f}/{max_terminated_fraction}; "
                f"length={diagnostics.get('mean_length_fraction', 0.0):.3f}/{min_mean_episode_length_fraction}"
            )

        command_term._truth_curve_evaluation_episodes = 0
        command_term._truth_curve_reward_totals = {
            name: 0.0 for name in reward_term_names
        }
        command_term._truth_curve_terminated_count = 0.0
        command_term._truth_curve_episode_steps = 0.0

    if emit_diagnostics:
        return {
            **command_term._truth_curve_diagnostics,
            "level": float(command_term._truth_curve_curriculum_level),
            "pending_episodes": float(command_term._truth_curve_evaluation_episodes),
            "success_streak": float(command_term._truth_curve_successful_evaluations),
        }
    return torch.tensor(
        float(command_term._truth_curve_curriculum_level), device=env.device
    )
