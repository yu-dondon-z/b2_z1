"""Physics-rate exclusion-zone state, independent of Isaac/MuJoCo.

A negative conservative clearance is an exclusion-zone violation, not a
measurement of contact force or penetration. Four 5 ms samples are latched
before the original 20 ms episode termination boundary.
"""

from __future__ import annotations

import math

import torch


class GeometryWindow:
    def __init__(self, num_envs, *, device="cpu", dtype=torch.float32,
                 decimation=4, soft_margin_m=0.03, hard_threshold_m=0.0,
                 cost_cap=9.0):
        values = (soft_margin_m, hard_threshold_m, cost_cap)
        if (not all(math.isfinite(x) for x in values)
                or soft_margin_m <= 0 or not 0 <= hard_threshold_m < soft_margin_m
                or cost_cap <= 0 or decimation != 4 or num_envs < 1):
            raise ValueError("Invalid VD12 exclusion-zone configuration")
        self.decimation = decimation
        self.soft_margin_m = soft_margin_m
        self.hard_threshold_m = hard_threshold_m
        self.cost_cap = cost_cap
        self.minimum_m = torch.full((num_envs,), torch.inf, device=device, dtype=dtype)
        self.last_m = torch.full_like(self.minimum_m, torch.inf)
        self.total_cost = torch.zeros_like(self.minimum_m)
        self.violated = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.first_bad_physics_step = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.window_id = None
        self.last_physics_step = None
        self.sample_count = 0

    def reset(self, env_ids=None):
        """Reset only selected episodes; the shared four-sample clock survives."""
        ids = slice(None) if env_ids is None else env_ids
        self.minimum_m[ids] = torch.inf
        self.last_m[ids] = torch.inf
        self.total_cost[ids] = 0
        self.violated[ids] = False
        self.first_bad_physics_step[ids] = 0

    def sample(self, clearance_m, input_valid, *, window_id, physics_step):
        if clearance_m.shape != self.minimum_m.shape or input_valid.shape != self.minimum_m.shape:
            raise ValueError("Expected one measured minimum clearance/validity per environment")
        if window_id != self.window_id:
            self.reset()
            self.window_id = window_id
            self.sample_count = 0
        elif self.last_physics_step is not None and physics_step != self.last_physics_step + 1:
            raise RuntimeError("VD12 geometry samples must be consecutive physical steps")
        if self.sample_count >= self.decimation:
            raise RuntimeError("Too many geometry samples in one actor window")
        valid = input_valid & torch.isfinite(clearance_m)
        measured = torch.where(valid, clearance_m, torch.full_like(clearance_m, -torch.inf))
        bad = (~valid) | (measured < self.hard_threshold_m)
        new_bad = bad & ~self.violated
        self.first_bad_physics_step = torch.where(
            new_bad, torch.full_like(self.first_bad_physics_step, physics_step),
            self.first_bad_physics_step,
        )
        self.violated |= bad
        self.minimum_m = torch.minimum(self.minimum_m, measured)
        self.last_m = measured
        excess = ((self.soft_margin_m - measured).clamp_min(0) / self.soft_margin_m).square()
        self.total_cost += excess.clamp_max(self.cost_cap)
        self.last_physics_step = physics_step
        self.sample_count += 1

    def require_complete(self, physics_step):
        if self.sample_count != self.decimation or self.last_physics_step != physics_step:
            raise RuntimeError("VD12 requires all four current 200 Hz samples before reward/termination")

    def mean_cost(self, physics_step):
        self.require_complete(physics_step)
        return self.total_cost / self.decimation
