"""VB10: real cylinder clearance, no ground support, measured vision startup."""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, SceneEntityCfg
from b2_z1.assets.vb10_gun_collision import flat_gun_urdf
from .b2_z1_vb9_env_cfg import (B2Z1VB9PoseGuideAMPEnvCfg, VB9PoseGuideRewardsCfg,
                               B2Z1VB9PoseGuideAMPEnvCfg_Play,
                               B2Z1VB9PoseGuideVisionAMPEnvCfg_Play)
from .mdp.gun_clearance import gun_support_force_cost, sustained_gun_contact


@configclass
class VB10ClearanceRewardsCfg(VB9PoseGuideRewardsCfg):
    gun_support_force = RewTerm(func=gun_support_force_cost,weight=-.75,
        params={'sensor_cfg':SceneEntityCfg('contact_forces',body_names='link_gun'),
                'force_scale_n':20.})


def configure_clearance(cfg):
    # Only change the gun collider representation, not joint normalization,
    # actuator gains, gun inertia/weight, leg colliders or tool TCP.
    cfg.scene.robot.spawn.asset_path = flat_gun_urdf(cfg.scene.robot.spawn.asset_path)
    cfg.commands.ee_target.tool_height_m = .01
    cfg.commands.ee_target.tool_height_range_m = (.01,.01)
    cfg.commands.ee_target.base_motion_delay_s = 1.5
    cfg.vb10_contact_diagnostics = True
    cfg.terminations.gun_sustained_contact = DoneTerm(
        func=sustained_gun_contact,
        params={'sensor_cfg':SceneEntityCfg('contact_forces',body_names='link_gun'),
                'threshold_n':10.,'duration_s':.8,'grace_s':4.})


@configclass
class B2Z1VB10ClearanceAMPEnvCfg(B2Z1VB9PoseGuideAMPEnvCfg):
    rewards: VB10ClearanceRewardsCfg = VB10ClearanceRewardsCfg()
    vb9_arm_guide_weight: float = .08

    def __post_init__(self):
        super().__post_init__()
        configure_clearance(self)
        # Keep the acquired clear branch, but allow load/position compensation.
        self.rewards.arm_deviation.weight = -.08
        self.rewards.end_effector_position_tracking_fine_grained.weight = 2.5
        self.rewards.end_effector_position_tracking_fine_grained.params['std'] = .05
        self.rewards.undesired_contacts.weight = -3.
        self.commands.ee_target.synthetic_initial_lateral_offset_range_m = (-.06,.06)
        self.commands.ee_target.synthetic_initial_heading_offset_range_rad = (-.06,.06)


@configclass
class B2Z1VB10ClearanceAMPEnvCfg_Play(B2Z1VB9PoseGuideAMPEnvCfg_Play):
    def __post_init__(self):
        super().__post_init__()
        configure_clearance(self)


@configclass
class B2Z1VB10ClearanceVisionAMPEnvCfg_Play(B2Z1VB9PoseGuideVisionAMPEnvCfg_Play):
    def __post_init__(self):
        super().__post_init__()
        configure_clearance(self)
        self.commands.ee_target.file_standby_before_motion = True
