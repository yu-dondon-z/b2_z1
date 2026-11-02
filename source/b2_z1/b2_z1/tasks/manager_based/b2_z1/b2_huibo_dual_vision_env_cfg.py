"""Fresh B2 + seven-axis Huibo tasks with the dual-camera command contract.

The actor has 73 observations and 19 actions, rather than the Z1 tasks'
70/18. Old VA15/VA16 policies and their Z1 action teacher cannot be loaded
into this morphology. The B2-only 43-D AMP motion prior remains applicable.
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from b2_z1.assets.b2_huibo import (
    B2_HUIBO_CFG, HUIBO_ARM_JOINT_NAMES, HUIBO_GUN_LENGTH_M, HUIBO_TCP_ORIENTATION_RPY,
)

from . import mdp
from .b2_z1_va16_dual_vision_env_cfg import (
    B2Z1VA16DualVisionAMPEnvCfg,
    B2Z1VA16DualVisionAMPEnvCfg_Play,
    VA16DualVisionSceneCfg,
    VA16SyntheticSceneCfg,
)
from .b2_z1_va_env_cfg import VA13LowBaseTerminationsCfg


B2_AMP_JOINT_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
]
B2_AMP_FOOT_NAMES = [f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")]


@configclass
class B2HuiboSyntheticSceneCfg(VA16SyntheticSceneCfg):
    robot = B2_HUIBO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class B2HuiboDualVisionSceneCfg(VA16DualVisionSceneCfg):
    robot = B2_HUIBO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class B2HuiboTerminationsCfg(VA13LowBaseTerminationsCfg):
    arm_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="right_link[1-7]"),
            "threshold": 1.0,
        },
    )


def _configure_huibo(env_cfg):
    """Select joints by name and retain only morphology-compatible priors."""
    arm_names = list(HUIBO_ARM_JOINT_NAMES)
    command = env_cfg.commands.ee_target
    # Commands stay in base_link coordinates, including the optical
    # feedback bridge; the Huibo mounting frame is a different orientation.
    command.link_name = "base_link"
    command.arm_joint_names = tuple(arm_names)
    command.tool_orientation_rpy = HUIBO_TCP_ORIENTATION_RPY
    command.file_standby_before_motion = False
    command.show_rear_camera_frustum = False
    # This is a Huibo-only cylinder edit. Original Z1 tasks retain 1.10 m.
    # Its lower face still coincides with the existing optical TCP offset.
    gun = env_cfg.scene.welding_gun
    gun.spawn.height = HUIBO_GUN_LENGTH_M
    gun.init_state.pos = (0.0, -0.03, -0.62 + 0.5 * HUIBO_GUN_LENGTH_M)
    command.rear_work_x_range_m = (-0.80, -0.75)
    command.synthetic_lateral_amplitude_range_m = (-0.08, 0.08)
    for ranges in (command.ranges, command.limit_ranges):
        ranges.pos_x = command.rear_work_x_range_m
        ranges.pos_y = (-0.08, 0.08)
    for term_name in ("arm_deviation", "arm_action_rate_l2"):
        getattr(env_cfg.rewards, term_name).params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=arm_names, preserve_order=True,
        )
    # The original reward explicitly requires six Z1 joints. Huibo's
    # redundant seventh axis needs its own validated nullspace objective.
    env_cfg.rewards.arm_nullspace_velocity = None

    # Reference-pose resets in the legacy environment write the first 12
    # articulation rows. A different arm can interleave that joint order.
    # Ordinary named default-pose resets avoid writing a leg pose to an arm.
    env_cfg.reference_state_initialization = False
    env_cfg.reference_state_initialization_prob = 0.0

    # AMP continues to observe just the B2 legs. Explicit dataset order
    # avoids the legacy assumption about the importer's grouped joint rows.
    amp = env_cfg.observations.amp_obs
    for term_name, function in (("joint_pos", mdp.joint_pos), ("joint_vel", mdp.joint_vel)):
        term = getattr(amp, term_name)
        term.func = function
        term.params = {
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=B2_AMP_JOINT_NAMES, preserve_order=True,
            )
        }
    amp.foot_pos.params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=B2_AMP_FOOT_NAMES, preserve_order=True,
    )


@configclass
class B2HuiboDualVisionAMPEnvCfg(B2Z1VA16DualVisionAMPEnvCfg):
    """Efficient synthetic-command acquisition for a new 19-action actor."""

    scene: B2HuiboSyntheticSceneCfg = B2HuiboSyntheticSceneCfg(num_envs=1024, env_spacing=4.0)
    terminations: B2HuiboTerminationsCfg = B2HuiboTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure_huibo(self)


@configclass
class B2HuiboDualVisionAMPEnvCfg_Play(B2Z1VA16DualVisionAMPEnvCfg_Play):
    """Image-feedback playback requiring a separately trained Huibo actor."""

    scene: B2HuiboDualVisionSceneCfg = B2HuiboDualVisionSceneCfg(num_envs=1, env_spacing=4.0)
    terminations: B2HuiboTerminationsCfg = B2HuiboTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure_huibo(self)
