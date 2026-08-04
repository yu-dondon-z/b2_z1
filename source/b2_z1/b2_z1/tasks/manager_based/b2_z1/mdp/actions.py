from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass


class FilteredJointPositionAction(JointPositionAction):
    """Joint-position action with the same EMA used by the MuJoCo deployment."""

    cfg: FilteredJointPositionActionCfg

    def __init__(self, cfg: FilteredJointPositionActionCfg, env):
        super().__init__(cfg, env)
        if not 0.0 < cfg.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {cfg.alpha}")
        self._alpha = cfg.alpha

    def process_actions(self, actions: torch.Tensor):
        # Keep raw_actions equal to the action that is actually applied. This
        # also makes mdp.last_action match the deployment observation.
        self._raw_actions[:] = (
            self._alpha * actions + (1.0 - self._alpha) * self._raw_actions
        )
        self._processed_actions = self._raw_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


@configclass
class FilteredJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`FilteredJointPositionAction`."""

    class_type: type = FilteredJointPositionAction
    alpha: float = 0.20
