"""VB9: acquire a clear arm posture before widening hanging-hose loads."""
import torch
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, EventTermCfg, SceneEntityCfg
from .b2_z1_vb8_env_cfg import (B2Z1VB8QuietLoadAMPEnvCfg,
                               B2Z1VB8QuietLoadAMPEnvCfg_Play,
                               B2Z1VB8QuietLoadVisionAMPEnvCfg_Play,
                               VB8QuietLoadedToolRewardsCfg)
from .mdp.loaded_tool_stability import gun_connector_normal_velocity_cost


def reset_clear_arm_mixture(env, env_ids, asset_cfg, fraction=.25):
    """Write only reset state, never live actions or default normalization.

    75% starts retain the legacy pose to preserve takeover; 25% demonstrate
    a clear branch. Playback never uses this reset event.
    """
    robot = env.scene[asset_cfg.name]
    if not 0. <= fraction <= 1.:
        raise ValueError('Clear-arm reset fraction must be within [0,1]')
    if env_ids is None:
        env_ids = torch.arange(env.num_envs,device=env.device)
    selected = env_ids[torch.rand(len(env_ids),device=env.device) < fraction]
    if len(selected) == 0:
        return
    q = torch.tensor([0.,2.7673,-2.4606,-.3067,0.,0.],device=env.device).expand(len(selected),6)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=asset_cfg.joint_ids, env_ids=selected)


@configclass
class VB9PoseGuideRewardsCfg(VB8QuietLoadedToolRewardsCfg):
    gun_connector_normal_velocity = RewTerm(
        func=gun_connector_normal_velocity_cost, weight=-.50,
        params={"asset_cfg": SceneEntityCfg("robot",body_names="link_gun"),
                "vertical_scale": .12, "lateral_scale": .20})


@configclass
class B2Z1VB9PoseGuideAMPEnvCfg(B2Z1VB8QuietLoadAMPEnvCfg):
    rewards: VB9PoseGuideRewardsCfg = VB9PoseGuideRewardsCfg()
    vb9_pose_guide_enabled: bool = True
    vb9_arm_guide_weight: float = .40
    vb9_guide_warmup_iterations: int = 200

    def __post_init__(self):
        super().__post_init__()
        # Stage A is posture acquisition, not a claim of 6/10 kg qualification.
        self.hose_mass_range_kg = (0.,3.)
        # AMP still trains gait. Do not overwrite the new reset mixture with
        # reference-frame arms/roots that reintroduce the old posture basin.
        self.reference_state_initialization_prob = 0.
        self.events.vb9_pose_reset = EventTermCfg(
            func=reset_clear_arm_mixture,mode="reset",
            params={"asset_cfg":SceneEntityCfg("robot",joint_names=[f"joint{i}" for i in range(1,7)],preserve_order=True),
                    "fraction": .25})
        self.rewards.base_world_heave.weight = -.80
        self.rewards.gun_world_tilt_rate.weight = -.35
        self.rewards.gun_vertical_axis.weight = -.60
        self.rewards.arm_limit_buffer.weight = -.80


@configclass
class B2Z1VB9LoadStableAMPEnvCfg(B2Z1VB9PoseGuideAMPEnvCfg):
    """Stage B: only resume a screened stage-A policy; guide gently remains."""
    vb9_arm_guide_weight: float = .08

    def __post_init__(self):
        super().__post_init__()
        self.hose_mass_range_kg = (0.,6.)


@configclass
class B2Z1VB9PoseGuideAMPEnvCfg_Play(B2Z1VB8QuietLoadAMPEnvCfg_Play):
    """Actor-only, unchanged nominal start; no reference controller/mixture."""
    pass


@configclass
class B2Z1VB9PoseGuideVisionAMPEnvCfg_Play(B2Z1VB8QuietLoadVisionAMPEnvCfg_Play):
    """Actor-only real vision validation of the stage-A/B trained checkpoint."""
    pass
