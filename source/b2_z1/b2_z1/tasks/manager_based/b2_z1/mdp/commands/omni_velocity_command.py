"""Structured omnidirectional velocity commands for B2+Z1 locomotion."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

from .command_cfg import UniformVelocityCommandCfg


class OmniVelocityCommand(UniformVelocityCommand):
    """Sample explicit forward, lateral, yaw-only and mixed motion modes.

    A plain uniform sampler almost never presents the policy with a genuinely
    straight command because all three velocity components are sampled at the
    same time.  This command generator reserves a configurable fraction of
    environments for each primitive.  Pure translation modes additionally
    hold the heading measured when the command was sampled.  The resulting
    corrective yaw-rate remains part of the ordinary 3-D velocity command, so
    the actor observation size stays compatible with existing checkpoints.
    """

    cfg: OmniVelocityCommandCfg

    def __init__(self, cfg: OmniVelocityCommandCfg, env):
        super().__init__(cfg, env)
        probs = (
            cfg.longitudinal_fraction,
            cfg.lateral_fraction,
            cfg.yaw_fraction,
            cfg.mixed_fraction,
        )
        if any(probability < 0.0 for probability in probs):
            raise ValueError("Omnidirectional command fractions must be non-negative.")
        if abs(sum(probs) - 1.0) > 1.0e-6:
            raise ValueError(f"Omnidirectional command fractions must sum to 1.0, got {sum(probs):.6f}.")
        if cfg.longitudinal_forward_fraction is not None and not (
            0.0 <= cfg.longitudinal_forward_fraction <= 1.0
        ):
            raise ValueError("longitudinal_forward_fraction must lie in [0, 1].")

        self.is_longitudinal_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_lateral_env = torch.zeros_like(self.is_longitudinal_env)
        self.is_yaw_env = torch.zeros_like(self.is_longitudinal_env)
        self.is_mixed_env = torch.zeros_like(self.is_longitudinal_env)
        self.is_heading_hold_env = torch.zeros_like(self.is_longitudinal_env)
        self.is_forward_env = torch.zeros_like(self.is_longitudinal_env)
        self.is_backward_env = torch.zeros_like(self.is_longitudinal_env)
        self.straight_heading_target = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            super().__str__()
            + "\n\tMode fractions (longitudinal/lateral/yaw/mixed): "
            + f"{self.cfg.longitudinal_fraction:.2f}/"
            + f"{self.cfg.lateral_fraction:.2f}/"
            + f"{self.cfg.yaw_fraction:.2f}/"
            + f"{self.cfg.mixed_fraction:.2f}"
        )

    def _sample_nonzero(self, count: int, value_range: tuple[float, float], minimum: float) -> torch.Tensor:
        """Uniformly sample a component while avoiding an ambiguous near-zero band."""
        low, high = value_range
        values = torch.empty(count, device=self.device).uniform_(low, high)
        if minimum <= 0.0:
            return values

        small = torch.abs(values) < minimum
        if not torch.any(small):
            return values

        negative_available = low <= -minimum
        positive_available = high >= minimum
        if negative_available and positive_available:
            signs = torch.where(
                torch.rand(count, device=self.device) < 0.5,
                -torch.ones(count, device=self.device),
                torch.ones(count, device=self.device),
            )
            values[small] = signs[small] * minimum
        elif positive_available:
            values[small] = minimum
        elif negative_available:
            values[small] = -minimum
        return values.clamp(min=low, max=high)

    def _sample_longitudinal(self, count: int) -> torch.Tensor:
        """Sample longitudinal speed with an optional explicit sign ratio."""
        forward_fraction = self.cfg.longitudinal_forward_fraction
        if forward_fraction is None:
            return self._sample_nonzero(
                count, self.cfg.ranges.lin_vel_x, self.cfg.minimum_linear_speed
            )

        low, high = self.cfg.ranges.lin_vel_x
        minimum = self.cfg.minimum_linear_speed
        forward_available = high >= minimum
        backward_available = low <= -minimum
        if not (forward_available or backward_available):
            return torch.zeros(count, device=self.device)

        choose_forward = torch.rand(count, device=self.device) < forward_fraction
        if not forward_available:
            choose_forward[:] = False
        if not backward_available:
            choose_forward[:] = True

        values = torch.empty(count, device=self.device)
        forward_count = int(choose_forward.sum().item())
        backward_count = count - forward_count
        if forward_count:
            values[choose_forward] = torch.empty(forward_count, device=self.device).uniform_(minimum, high)
        if backward_count:
            values[~choose_forward] = torch.empty(backward_count, device=self.device).uniform_(low, -minimum)
        return values

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        count = len(env_ids)
        mode = torch.rand(count, device=self.device)
        longitudinal_end = self.cfg.longitudinal_fraction
        lateral_end = longitudinal_end + self.cfg.lateral_fraction
        yaw_end = lateral_end + self.cfg.yaw_fraction

        longitudinal = mode < longitudinal_end
        lateral = (mode >= longitudinal_end) & (mode < lateral_end)
        yaw = (mode >= lateral_end) & (mode < yaw_end)
        mixed = mode >= yaw_end

        self.is_longitudinal_env[env_ids] = longitudinal
        self.is_lateral_env[env_ids] = lateral
        self.is_yaw_env[env_ids] = yaw
        self.is_mixed_env[env_ids] = mixed

        self.vel_command_b[env_ids] = 0.0
        sampled_x = self._sample_longitudinal(count)
        sampled_y = self._sample_nonzero(count, self.cfg.ranges.lin_vel_y, self.cfg.minimum_linear_speed)
        sampled_yaw = self._sample_nonzero(count, self.cfg.ranges.ang_vel_z, self.cfg.minimum_yaw_rate)

        self.vel_command_b[env_ids[longitudinal], 0] = sampled_x[longitudinal]
        self.vel_command_b[env_ids[lateral], 1] = sampled_y[lateral]
        self.vel_command_b[env_ids[yaw], 2] = sampled_yaw[yaw]
        self.vel_command_b[env_ids[mixed], 0] = sampled_x[mixed]
        self.vel_command_b[env_ids[mixed], 1] = sampled_y[mixed]
        self.vel_command_b[env_ids[mixed], 2] = sampled_yaw[mixed]

        standing = torch.rand(count, device=self.device) <= self.cfg.rel_standing_envs
        self.is_standing_env[env_ids] = standing
        self.is_heading_hold_env[env_ids] = (longitudinal | lateral) & ~standing
        self.is_forward_env[env_ids] = longitudinal & (sampled_x > 0.0) & ~standing
        self.is_backward_env[env_ids] = longitudinal & (sampled_x < 0.0) & ~standing
        self.straight_heading_target[env_ids] = self.robot.data.heading_w[env_ids]

    def _update_command(self):
        heading_ids = self.is_heading_hold_env.nonzero(as_tuple=False).flatten()
        if len(heading_ids) > 0:
            heading_error = wrap_to_pi(
                self.straight_heading_target[heading_ids] - self.robot.data.heading_w[heading_ids]
            )
            self.vel_command_b[heading_ids, 2] = torch.clamp(
                self.cfg.heading_hold_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

        standing_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if len(standing_ids) > 0:
            self.vel_command_b[standing_ids] = 0.0


@configclass
class OmniVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`OmniVelocityCommand`."""

    class_type: type = OmniVelocityCommand

    longitudinal_fraction: float = 0.35
    lateral_fraction: float = 0.25
    yaw_fraction: float = 0.15
    mixed_fraction: float = 0.25
    minimum_linear_speed: float = 0.10
    minimum_yaw_rate: float = 0.10
    heading_hold_stiffness: float = 1.25
    longitudinal_forward_fraction: float | None = None
