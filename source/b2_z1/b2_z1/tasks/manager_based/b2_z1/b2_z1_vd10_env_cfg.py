"""Isolated 2.93 kg flange-aligned tube, with VD9's 200 Hz gait/TCP rewards."""

from isaaclab.utils import configclass

from b2_z1.assets.unitree import B2_Z1_HOLLOW_GUN_CFG

from .b2_z1_vd9_env_cfg import B2Z1VD9EnvCfg, B2Z1VD9PlayEnvCfg
from .mdp.commands.vd10_safe_seam_command import SAFE_PITCH_RAD, VD10SafeSeamCommand


def _configure_hollow_gun(cfg, *, play=False):
    robot = B2_Z1_HOLLOW_GUN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (*robot.init_state.pos[:2], cfg.vd5_nominal_base_height_m)
    # Fixed-base offline FK: high-clearance, within-limits spawn, TCP near
    # (-0.95, 0, 0.85) m at base height 0.50 m. Even a 0.45 m TCP start
    # struck the ground under zero action with actual arm/base sag.
    robot.init_state.joint_pos.update({
        "joint1": 0.0, "joint2": 0.66647356, "joint3": -1.25178768,
        "joint4": 0.88531411, "joint5": 0.0, "joint6": 0.0,
    })
    cfg.scene.robot = robot
    # The near-start temporary command matches a reachable, raised pose; only
    # VD10's command interpolates its pitch back to the original seam angle.
    # The VD5 world seam and gun-ground termination remain fully active.
    goal = cfg.commands.ee_target
    goal.class_type = VD10SafeSeamCommand
    goal.ranges.pos_x = (-0.96, -0.94)
    goal.ranges.pos_y = (-0.01, 0.01)
    goal.ranges.pos_z = (0.34, 0.36)
    goal.ranges.pitch = (SAFE_PITCH_RAD, SAFE_PITCH_RAD)
    # VD5 has curriculum_enabled=False, so its constructor replaces ranges
    # with limit_ranges; both must encode the safe temporary workspace.
    goal.limit_ranges.pos_x = (-0.96, -0.94)
    goal.limit_ranges.pos_y = (-0.01, 0.01)
    goal.limit_ranges.pos_z = (0.34, 0.36)
    goal.limit_ranges.pitch = (SAFE_PITCH_RAD, SAFE_PITCH_RAD)
    # At 50 Hz actor control, 20k actor steps correspond to ~400 s of training
    # time per environment / 833 PPO iterations (24 steps per iteration).
    # Start above the floor, then gradually introduce the real ground seam.
    goal.vd5_seam_trailing_distance = 0.80
    goal.vd5_ground_transition_steps = 0 if play else 20_000


@configclass
class B2Z1VD10EnvCfg(B2Z1VD9EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_hollow_gun(self)


@configclass
class B2Z1VD10PlayEnvCfg(B2Z1VD9PlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_hollow_gun(self, play=True)
