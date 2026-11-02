"""B2 with the seven-axis Huibo right arm supplied in the local archive.

The archive records zero effort and velocity limits for every moving arm
joint. The values below are explicit simulation assumptions for importing
and training this model; they are not Huibo hardware specifications.
"""

from copy import deepcopy

from isaaclab.actuators import ImplicitActuatorCfg

from .unitree import B2Z1_ASSETS_DIR, B2_Z1_NO_STICK_CFG, UnitreeUrdfFileCfg


HUIBO_ARM_JOINT_NAMES = tuple(f"right_joint{index}" for index in range(1, 8))
# Huibo-only tool geometry. The old 1.10 m placeholder extended 0.48 m
# above the adapter; an axial grip would put that portion through the wrist.
# Keep the TCP/fiducial offsets but end the cylinder 0.02 m below gun Z=0.
HUIBO_GUN_LENGTH_M = 0.60
HUIBO_INITIAL_TCP_B = (-0.775, 0.0, -0.48)
HUIBO_INITIAL_MAX_TILT_DEG = 13.3
HUIBO_INITIAL_MIN_LIMIT_MARGIN_RAD = 0.099
# dummy_target orientation for the same FK-checked axial working pose.
HUIBO_TCP_ORIENTATION_RPY = (-0.672152093357085, 1.34057420646849, -3.1415826441195884)
# Flange +X and gun -Z are coaxial and point down, 13.19 degrees rearward
# from vertical. This working solution keeps the original wrist limits;
# every joint retains at least 0.10 rad of limit margin. It is not vertical.
HUIBO_INITIAL_JOINT_POS = dict(zip(HUIBO_ARM_JOINT_NAMES, (
    1.57079226334755,
    0.9351052713220801,
    -1.5751082043382922,
    1.0912415788479128,
    -0.8986226998003438,
    -0.650491566,
    -0.650491566,
)))

B2_HUIBO_CFG = deepcopy(B2_Z1_NO_STICK_CFG)
B2_HUIBO_CFG.spawn = UnitreeUrdfFileCfg(
    asset_path=f"{B2Z1_ASSETS_DIR}/b2_huibo.urdf",
    self_collision=False,
)
B2_HUIBO_CFG.init_state.joint_pos = {
    name: value
    for name, value in B2_HUIBO_CFG.init_state.joint_pos.items()
    if name not in {f"joint{index}" for index in range(1, 7)}
}
B2_HUIBO_CFG.init_state.joint_pos.update(HUIBO_INITIAL_JOINT_POS)
B2_HUIBO_CFG.actuators = {
    name: deepcopy(B2_Z1_NO_STICK_CFG.actuators[name])
    for name in ("M107-24-2", "2")
}
B2_HUIBO_CFG.actuators["huibo_arm_simulation"] = ImplicitActuatorCfg(
    joint_names_expr=list(HUIBO_ARM_JOINT_NAMES),
    effort_limit_sim=30.0,
    velocity_limit_sim=1.5,
    stiffness=40.0,
    damping=2.0,
    friction=0.01,
)
