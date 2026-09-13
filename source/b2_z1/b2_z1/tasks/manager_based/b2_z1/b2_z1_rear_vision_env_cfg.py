"""Rear-facing Z1 training and RGB-D visual-trajectory playback configs."""

from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from . import mdp
from .b2_z1_amp_env_cfg import (
    B2Z1AMPEnvCfg,
    B2Z1SceneCfg,
    GROUND_MATERIAL_CFG,
    StabilityFineTuneRewardsCfg,
)


RGB_WIDTH = 640
RGB_HEIGHT = 480
ROAD_HALF_LENGTH_M = 95.0
ROAD_HALF_WIDTH_M = 95.0
CRACK_WIDTH_M = 0.02
CRACK_DEPTH_M = 0.05
CRACK_CURVE_HALF_LENGTH_M = 20.0
# The cubic sine has zero slope at x=0, so the robot starts on a centered,
# forward-facing crack instead of immediately needing a 14-degree turn.  It
# then reaches +/-0.40 m lateral displacement with a minimum radius of about
# 8.4 m, inside the 6--12 m arc range used by V5 training.
CRACK_CURVE_AMPLITUDE_M = 0.40
CRACK_CURVE_WAVELENGTH_M = 20.0
CRACK_CURVE_MESH_STEP_M = 0.05
# Same cylinder dimensions as b2_z1_gun.urdf, with its lower face centred on
# the existing dummy_target joint origin in link_gun coordinates.
WELDING_GUN_LENGTH_M = 1.10
WELDING_GUN_RADIUS_M = 0.028
WELDING_GUN_TIP_POSITION = (0.0, -0.03, -0.62)


def _camera_model(fx: float, fy: float, clipping_range=(0.07, 6.0)):
    return sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=[fx, 0.0, 319.5, 0.0, fy, 239.5, 0.0, 0.0, 1.0],
        width=RGB_WIDTH,
        height=RGB_HEIGHT,
        clipping_range=clipping_range,
    )


@sim_utils.clone
def _spawn_seamless_road_half(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn one continuous road half with a curved inner trench wall.

    A single mesh is important here: the former 0.25 m cuboids left transverse
    render seams, and RHACrackNet occasionally followed those seams instead of
    the real crack.  The top, bottom and side walls below form one watertight
    static triangle mesh with no cross-road joints.
    """
    import isaacsim.core.utils.prims as prim_utils
    import numpy as np
    import trimesh

    from isaaclab.terrains.utils import create_prim_from_mesh

    curve_x = np.linspace(
        -cfg.curve_half_length,
        cfg.curve_half_length,
        round(2.0 * cfg.curve_half_length / cfg.mesh_step) + 1,
        dtype=np.float64,
    )
    x = np.concatenate(([-cfg.half_length], curve_x, [cfg.half_length]))
    center_y = np.zeros_like(x)
    curved = np.abs(x) <= cfg.curve_half_length + 1.0e-9
    phase = 2.0 * np.pi * x[curved] / cfg.wavelength
    center_y[curved] = cfg.amplitude * np.sin(phase) ** 3

    side_sign = 1.0 if cfg.left_side else -1.0
    inner_y = center_y + side_sign * 0.5 * cfg.crack_width
    outer_y = np.full_like(x, side_sign * cfg.half_width)
    top_z = np.zeros_like(x)
    bottom_z = np.full_like(x, -cfg.crack_depth)

    # Per station: inner-top, outer-top, inner-bottom, outer-bottom.
    vertices = np.stack(
        (
            np.column_stack((x, inner_y, top_z)),
            np.column_stack((x, outer_y, top_z)),
            np.column_stack((x, inner_y, bottom_z)),
            np.column_stack((x, outer_y, bottom_z)),
        ),
        axis=1,
    ).reshape(-1, 3)
    faces: list[tuple[int, int, int]] = []
    for station in range(len(x) - 1):
        a = 4 * station
        b = a + 4
        # top, bottom, inner trench wall and remote outer wall
        faces.extend(
            [
                (a, a + 1, b + 1), (a, b + 1, b),
                (a + 2, b + 3, a + 3), (a + 2, b + 2, b + 3),
                (a, b, b + 2), (a, b + 2, a + 2),
                (a + 1, a + 3, b + 3), (a + 1, b + 3, b + 1),
            ]
        )
    last = 4 * (len(x) - 1)
    faces.extend(
        [
            (0, 2, 3), (0, 3, 1),
            (last, last + 1, last + 3), (last, last + 3, last + 2),
        ]
    )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    # The per-station inner/outer ordering is mirrored between the two road
    # halves.  Consequently one generated shell has a negative signed volume
    # unless its winding is reversed. PhysX triangle-mesh collision is
    # one-sided: leaving that shell inverted lets the corresponding feet pass
    # through the road and makes the robot roll immediately onto its side.
    if mesh.volume < 0.0:
        mesh.invert()
    create_prim_from_mesh(
        prim_path,
        mesh,
        translation=translation,
        orientation=orientation,
        visual_material=cfg.visual_material,
        physics_material=cfg.physics_material,
    )
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class SeamlessCrackedRoadHalfCfg(sim_utils.SpawnerCfg):
    """Parameters for one side of the seamless cracked road."""

    func = _spawn_seamless_road_half
    left_side: bool = True
    half_length: float = ROAD_HALF_LENGTH_M
    half_width: float = ROAD_HALF_WIDTH_M
    crack_width: float = CRACK_WIDTH_M
    crack_depth: float = CRACK_DEPTH_M
    curve_half_length: float = CRACK_CURVE_HALF_LENGTH_M
    amplitude: float = CRACK_CURVE_AMPLITUDE_M
    wavelength: float = CRACK_CURVE_WAVELENGTH_M
    mesh_step: float = CRACK_CURVE_MESH_STEP_M
    visual_material: sim_utils.PreviewSurfaceCfg = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.42, 0.42, 0.42), roughness=0.95
    )
    physics_material: sim_utils.RigidBodyMaterialCfg = GROUND_MATERIAL_CFG


@configclass
class B2Z1RearVisionSceneCfg(B2Z1SceneCfg):
    """B2 built-in front/rear RGB-D and optional front-down D435i."""

    # A compound collider belonging to the existing link_gun rigid body.
    # Do not add a separate rigid body or mass: the trained URDF's 1 g tool
    # link and fixed joints are retained, including the dummy_target TCP.
    welding_gun = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link_gun/welding_gun",
        spawn=sim_utils.CylinderCfg(
            radius=WELDING_GUN_RADIUS_M,
            height=WELDING_GUN_LENGTH_M,
            axis="Z",
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.002, rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.65, 0.68, 0.72), metallic=0.8, roughness=0.3,
            ),
            physics_material=GROUND_MATERIAL_CFG,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                WELDING_GUN_TIP_POSITION[0],
                WELDING_GUN_TIP_POSITION[1],
                WELDING_GUN_TIP_POSITION[2] + 0.5 * WELDING_GUN_LENGTH_M,
            ),
        ),
    )

    # Uniform illumination avoids the directional sun in the inherited HDR
    # sky, whose robot silhouette was segmented as part of the crack.
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            color=(1.0, 1.0, 1.0),
            texture_file=None,
        ),
    )

    # A real 2 cm-wide, 5 cm-deep trench. The inherited base ground is lowered
    # so its top is the trench bottom at z=-0.05 m. Two continuous meshes form
    # the road surface and curved trench walls; this avoids visual seams that
    # can be mistaken for a crack by the segmentation network.
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.CuboidCfg(
            size=(200.0, 200.0, 0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.003, 0.003, 0.003), roughness=1.0
            ),
            physics_material=GROUND_MATERIAL_CFG,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.10)),
        collision_group=-1,
    )

    road_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RoadLeft",
        spawn=SeamlessCrackedRoadHalfCfg(left_side=True),
        collision_group=-1,
    )
    road_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RoadRight",
        spawn=SeamlessCrackedRoadHalfCfg(left_side=False),
        collision_group=-1,
    )

    # Built-in optical cameras. Their transforms come from joint_f_oc and
    # joint_r_oc in the B2 URDF, expressed directly relative to base_link so
    # fixed-joint collapsing in the URDF converter cannot remove the parent.
    front_rgb = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_rgb",
        update_period=1.0 / 15.0,
        width=RGB_WIDTH,
        height=RGB_HEIGHT,
        data_types=["rgb"],
        spawn=_camera_model(460.0, 460.0),
        offset=CameraCfg.OffsetCfg(
            pos=(0.39079, 0.0, -0.013433),
            rot=(0.49999816, -0.5, 0.50000184, -0.5),
            convention="ros",
        ),
    )
    rear_rgb = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/rear_rgb",
        update_period=1.0 / 15.0,
        width=RGB_WIDTH,
        height=RGB_HEIGHT,
        data_types=["rgb"],
        spawn=_camera_model(460.0, 460.0),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.38451, 0.0, -0.023249),
            rot=(0.49999816, -0.5, -0.50000184, 0.5),
            convention="ros",
        ),
    )

    # Built-in depth cameras are pitched down approximately 45 degrees. The
    # forward unit also exposes co-located RGB in simulation so RHACrackNet
    # receives pixel-aligned RGB-D from the built-in mounting position.
    front_depth = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_depth",
        update_period=1.0 / 15.0,
        width=RGB_WIDTH,
        height=RGB_HEIGHT,
        data_types=["rgb", "distance_to_image_plane"],
        depth_clipping_behavior="zero",
        spawn=_camera_model(430.0, 430.0),
        offset=CameraCfg.OffsetCfg(
            pos=(0.41251, 0.024997, 0.04765),
            rot=(0.27059575, -0.65328103, 0.65328343, -0.27059675),
            convention="ros",
        ),
    )
    rear_depth = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/rear_depth",
        update_period=1.0 / 15.0,
        width=RGB_WIDTH,
        height=RGB_HEIGHT,
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="zero",
        spawn=_camera_model(430.0, 430.0),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.4081, -0.025, 0.024861),
            rot=(0.27075907, -0.65321336, -0.65321576, 0.27076006),
            convention="ros",
        ),
    )

    # Optional D435i (the requested "D35i" is treated as D435i until the
    # exact model is confirmed). Mount it ahead of the B2 head and above the
    # chassis, looking about 45 degrees down.  The forward offset is
    # intentional: a mount near x=0.30 m is inside the arm/chassis silhouette
    # and loses most of the crack image to self-occlusion.
    d435i = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/d435i",
        update_period=1.0 / 15.0,
        width=RGB_WIDTH,
        height=RGB_HEIGHT,
        data_types=["rgb", "distance_to_image_plane"],
        depth_clipping_behavior="zero",
        # Nominal square-pixel intrinsics for the 640x480 simulation stream.
        # Replace with the calibrated matrix for the physical D435i.
        spawn=_camera_model(392.5, 392.5),
        offset=CameraCfg.OffsetCfg(
            # Requires a short front/top bracket.  At this height the arm and
            # front shell stay outside most of the road-looking field of view.
            pos=(0.48, 0.0, 0.34),
            rot=(0.27059805, -0.65328148, 0.65328148, -0.27059805),
            convention="ros",
        ),
    )


def _configure_vision_playback_cameras(env_cfg):
    """Render the selected RGB-D stream at its export cadence during playback."""
    command = env_cfg.commands.ee_target
    selected = command.vision_camera_name
    camera = getattr(env_cfg.scene, selected, None)
    if camera is None or not {"rgb", "distance_to_image_plane"}.issubset(camera.data_types):
        raise ValueError(f"Vision playback requires an RGB-D camera, got {selected!r}")
    if os.environ.get("B2_Z1_VISION_ALL_CAMERAS", "0") != "1":
        for name in ("front_rgb", "rear_rgb", "front_depth", "rear_depth", "d435i"):
            if name != selected:
                setattr(env_cfg.scene, name, None)
    # Physics and policy stepping retain their trained rates. RTX only needs
    # to render when the bridge publishes a frame; extra renders slow wall-
    # clock motion without supplying any additional vision to the controller.
    env_cfg.sim.render_interval = env_cfg.decimation * command.camera_export_period_steps
    camera.update_period = env_cfg.sim.dt * env_cfg.sim.render_interval


def _rear_command(mode: str, debug_vis: bool):
    exchange_dir = os.environ.get("B2_Z1_VISION_EXCHANGE", "/tmp/b2_z1_vision_bridge")
    vision_camera_name = os.environ.get("B2_Z1_VISION_CAMERA", "front_depth")
    ranges = mdp.command_cfg.UniformPoseWBCCommandCfg.Ranges(
        pos_x=(-1.0, -0.55),
        pos_y=(-0.15, 0.15),
        pos_z=(-0.38, -0.30),
        roll=(0.0, 0.0),
        pitch=(math.pi / 2.0, math.pi / 2.0),
        yaw=(math.pi, math.pi),
    )
    return mdp.RearVisionTrajectoryCommandCfg(
        asset_name="robot",
        body_name="dummy_target",
        link_name="base_link",
        base_name="base_link",
        # Synthetic training deliberately resamples trajectories. A live file
        # stream must not: the previous 8--10 s timer periodically erased the
        # cached visual path and produced a sudden stop/TCP takeover cycle.
        # Environment reset still invokes _resample_command normally.
        resampling_time_range=(
            (8.0, 10.0) if mode == "synthetic" else (1.0e6, 1.0e6)
        ),
        debug_vis=debug_vis,
        mode=mode,
        trajectory_file=os.environ.get(
            "B2_Z1_VISION_TRAJECTORY",
            os.path.join(exchange_dir, "latest_trajectory.json"),
        ),
        vision_camera_name=vision_camera_name,
        vision_exchange_directory=exchange_dir,
        ranges=ranges,
        limit_ranges=ranges,
        curriculum_enabled=False,
        rear_work_x_range_m=(-1.0, -0.55),
        tool_height_m=0.10,
        synthetic_ground_z_b_m=-0.45,
        path_speed_mps=0.25,
        synthetic_speed_range_mps=(0.10, 0.60),
    )


@configclass
class B2Z1RearSyntheticAMPEnvCfg(B2Z1AMPEnvCfg):
    """Efficient rear-arm training with synthetic crack tangents (no RTX)."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_target = _rear_command("synthetic", debug_vis=False)
        # EEDerivedVelocityCommand delegates to the trajectory tangent exposed
        # by RearVisionTrajectoryCommand, so it no longer points toward a rear
        # target and accidentally commands backward walking.
        self.commands.base_velocity.curriculum_enabled = False


@configclass
class B2Z1RearDeploymentFineTuneAMPEnvCfg(B2Z1RearSyntheticAMPEnvCfg):
    """Rear-path fine-tuning for stand/start transitions and sim-to-sim stability.

    Unlike the older deployment fine-tune task, this task deliberately keeps
    the filtered action interface used to train ``model_7999.pt``.  The
    ordinary standing resets and periodically sampled standing commands fill
    the state-distribution gap between reference-state Isaac playback and a
    controller taking over a stationary robot in MuJoCo or on hardware.
    """

    rewards: StabilityFineTuneRewardsCfg = StabilityFineTuneRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Long enough to expose delayed oscillation while still generating
        # several stop/start command transitions within every episode.
        self.episode_length_s = 60.0

        # Keep some motion-reference resets to preserve the learned trot, but
        # train primarily from the nominal static pose used at deployment.
        self.reference_state_initialization_prob = 0.25
        self.events.reset_base.params["pose_range"].update(
            {"roll": (-0.04, 0.04), "pitch": (-0.04, 0.04)}
        )
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.04, 0.04),
            "y": (-0.04, 0.04),
            "z": (-0.02, 0.02),
            "roll": (-0.08, 0.08),
            "pitch": (-0.08, 0.08),
            "yaw": (-0.06, 0.06),
        }
        self.events.reset_robot_joints.params["velocity_range"] = (-0.05, 0.05)

        # EEDerivedVelocityCommand masks the synthetic path velocity for these
        # environments. Re-sampling every few seconds teaches both directions
        # of the transition without changing the rear TCP command geometry.
        self.commands.base_velocity.rel_standing_envs = 0.25
        self.commands.base_velocity.resampling_time_range = (4.0, 7.0)
        self.commands.base_velocity.curriculum_enabled = False

        # Match realistic deployment disturbances. The base task's stronger,
        # very frequent pushes obscure the specific takeover behavior.
        self.events.push_robot.interval_range_s = (10.0, 15.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.20, 0.20),
            "y": (-0.25, 0.25),
            "roll": (-0.15, 0.15),
            "pitch": (-0.12, 0.12),
            "yaw": (-0.12, 0.12),
        }

        # StabilityFineTuneRewardsCfg adds world-frame TCP acceleration and
        # diagonal-contact symmetry. These weights retain enough leg bandwidth
        # for the AMP trot while discouraging the observed arm/body shaking.
        self.rewards.feet_air_time.weight = 0.50
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.action_smoothness.weight = -0.02
        self.rewards.dof_acc_l2.weight = -1.0e-6
        self.rewards.ee_stability.weight = -0.20
        self.rewards.base_angular_velocity.weight = -1.0


@configclass
class RearVisionV2FineTuneRewardsCfg(StabilityFineTuneRewardsCfg):
    """Rewards for crack-centred walking and safe foot placement."""

    crack_centerline_tracking = RewTerm(
        func=mdp.crack_centerline_tracking_exp,
        weight=0.50,
        params={
            "command_name": "ee_target",
            # Preserve a useful gradient during recoverable deviations instead
            # of collapsing to almost zero beyond ten centimetres.
            "std": 0.20,
            "velocity_threshold": 0.08,
        },
    )
    crack_heading_tracking = RewTerm(
        func=mdp.crack_heading_tracking_exp,
        weight=0.20,
        params={
            "command_name": "ee_target",
            "std": 0.35,
            "velocity_threshold": 0.08,
        },
    )
    stance_feet_crack_clearance = RewTerm(
        func=mdp.stance_feet_crack_clearance_penalty,
        weight=-0.20,
        params={
            "command_name": "ee_target",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            # 1 cm crack half-width + approximately 3 cm sole radius + margin.
            "safe_distance": 0.07,
            "contact_force_threshold": 5.0,
            "velocity_threshold": 0.08,
        },
    )


@configclass
class B2Z1RearVisionV2FineTuneAMPEnvCfg(B2Z1RearDeploymentFineTuneAMPEnvCfg):
    """Fine-tune V2 for lower TCP, smooth takeover and crack-centred gait.

    The policy observation remains exactly 70-dimensional. Synthetic crack
    geometry is privileged training information used only by the command and
    reward terms, so existing rear-vision checkpoints remain load-compatible.
    """

    rewards: RearVisionV2FineTuneRewardsCfg = RearVisionV2FineTuneRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # A stationary world-frame curve allows physical base drift to become
        # a genuine cross-track error. This is the condition that was absent
        # from the original moving base-frame synthetic path.
        self.commands.ee_target.synthetic_world_fixed_path = True
        self.commands.ee_target.synthetic_lateral_amplitude_range_m = (-0.12, 0.12)
        self.commands.ee_target.synthetic_wavelength_range_m = (16.0, 28.0)
        self.commands.ee_target.synthetic_speed_range_mps = (0.20, 0.35)
        self.commands.ee_target.path_lookahead_m = 1.00
        self.commands.ee_target.cross_track_gain = 0.45
        self.commands.ee_target.path_heading_gain = 0.80
        self.commands.ee_target.max_yaw_rate_rps = 0.15

        # Train down from the old 10 cm standoff instead of applying a sudden
        # deployment-only z offset. The target is reconstructed from world
        # road height and current base pose every step.
        self.commands.ee_target.tool_height_range_m = (0.08, 0.10)
        self.commands.ee_target.synthetic_ground_height_w_m = 0.0
        self.commands.ee_target.command_transition_time_s = 4.0
        self.commands.ee_target.velocity_filter_time_constant_s = 0.45

        # World-fixed paths require a path-relative reset distribution. The
        # former +/-0.5 m lateral and +/-pi yaw reset inherited from generic
        # locomotion placed many environments outside the arm workspace or
        # facing away from the crack before the first policy action.
        self.events.reset_base.params["pose_range"].update(
            {
                "x": (-0.20, 0.20),
                "y": (-0.05, 0.05),
                "yaw": (-0.10, 0.10),
            }
        )
        self.events.push_robot.params["velocity_range"].update(
            {
                "y": (-0.10, 0.10),
                "roll": (-0.08, 0.08),
                "pitch": (-0.08, 0.08),
                "yaw": (-0.06, 0.06),
            }
        )

        # Repeated smooth stand/start changes teach the same transition that
        # pressing T produces without switching targets every few seconds.
        self.commands.ee_target.resampling_time_range = (10.0, 14.0)
        self.commands.base_velocity.rel_standing_envs = 0.25
        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)
        self.rewards.track_lin_vel_xy.weight = 2.5
        self.rewards.track_ang_vel_z.weight = 1.5


@configclass
class B2Z1RearVisionV2FineTuneAMPEnvCfg_Visual(
    B2Z1RearVisionV2FineTuneAMPEnvCfg
):
    """Small-batch viewer configuration for inspecting V2.1 while training."""

    def __post_init__(self):
        super().__post_init__()
        # Debug markers are intentionally limited to this viewer task. Keeping
        # them out of the 1024-environment task avoids slowing formal runs.
        self.scene.num_envs = 32
        self.commands.ee_target.debug_vis = True


@configclass
class B2Z1RearVisionV2FineTuneAMPEnvCfg_Play(B2Z1RearVisionV2FineTuneAMPEnvCfg):
    """Deterministic playback for selecting the V2.1 fine-tune checkpoint."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        # Playback is operator-controlled. The former 120 s timeout reset the
        # robot in the middle of a walk and looked like a sudden crouch every
        # two minutes. Keep a one-hour safety horizon instead.
        self.episode_length_s = 3600.0
        self.reference_state_initialization_prob = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.ee_target.synthetic_speed_range_mps = (0.30, 0.30)
        self.commands.ee_target.synthetic_lateral_amplitude_range_m = (0.12, 0.12)
        self.commands.ee_target.synthetic_wavelength_range_m = (20.0, 20.0)
        self.commands.ee_target.tool_height_range_m = (0.09, 0.09)

        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_dynamics = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class RearVisionV3GaitFineTuneRewardsCfg(RearVisionV2FineTuneRewardsCfg):
    """Conservative gait-refinement rewards built on the stable V2.2 task."""


@configclass
class B2Z1RearVisionV3GaitFineTuneAMPEnvCfg(B2Z1RearVisionV2FineTuneAMPEnvCfg):
    """Short continuation that targets the remaining limp and speed lag.

    This deliberately preserves the 70-D observation, action filtering, rear
    TCP geometry and visual path command used by ``model_9600.pt``.  Only the
    moving-state sampling and a few locomotion reward weights change, so the
    continuation does not have to relearn the arm/vision task.
    """

    rewards: RearVisionV3GaitFineTuneRewardsCfg = RearVisionV3GaitFineTuneRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Spend more of this short run on the deployed 0.30 m/s walking state,
        # while retaining enough stand/start samples to preserve smooth policy
        # takeover.  Keep the range inside the data-supported V2.2 envelope.
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.resampling_time_range = (12.0, 18.0)
        self.commands.ee_target.synthetic_speed_range_mps = (0.25, 0.35)

        # model_9600 already tracks commands well in Isaac Sim, but its
        # diagonal gait reward reaches only about 64% of the available score.
        # These are small changes rather than hard kinematic constraints: AMP
        # remains responsible for natural motion and the legs retain clearance.
        self.rewards.track_lin_vel_xy.weight = 3.0
        self.rewards.trot_diagonal_sync.weight = 0.50
        self.rewards.air_time_variance.weight = -0.40
        self.rewards.feet_height.weight = 0.35


@configclass
class B2Z1RearVisionV3GaitFineTuneAMPEnvCfg_Visual(
    B2Z1RearVisionV3GaitFineTuneAMPEnvCfg
):
    """Small-batch viewer for checking the V3 gait continuation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.ee_target.debug_vis = True


@configclass
class B2Z1RearVisionV3GaitFineTuneAMPEnvCfg_Play(
    B2Z1RearVisionV3GaitFineTuneAMPEnvCfg
):
    """Deterministic playback used to select a V3 gait checkpoint."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 3600.0
        self.reference_state_initialization_prob = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.ee_target.synthetic_speed_range_mps = (0.30, 0.30)
        self.commands.ee_target.synthetic_lateral_amplitude_range_m = (0.12, 0.12)
        self.commands.ee_target.synthetic_wavelength_range_m = (20.0, 20.0)
        self.commands.ee_target.tool_height_range_m = (0.09, 0.09)

        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_dynamics = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class RearCurveGaitHeadingV5RewardsCfg(RearVisionV3GaitFineTuneRewardsCfg):
    """V5 rewards for straight heading retention and symmetric curved trot."""

    straight_lateral_velocity = RewTerm(
        func=mdp.path_straight_lateral_velocity_l2,
        weight=-1.50,
        params={
            "path_command_name": "ee_target",
            "velocity_command_name": "base_velocity",
            "velocity_threshold": 0.10,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    straight_yaw_rate = RewTerm(
        func=mdp.path_straight_yaw_rate_l2,
        weight=-1.00,
        params={
            "path_command_name": "ee_target",
            "velocity_command_name": "base_velocity",
            "velocity_threshold": 0.10,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    straight_heading_drift = RewTerm(
        func=mdp.path_straight_heading_drift_l2,
        weight=-1.25,
        params={
            "path_command_name": "ee_target",
            "velocity_command_name": "base_velocity",
            "velocity_threshold": 0.10,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class B2Z1RearCurveGaitHeadingV5FineTuneAMPEnvCfg(
    B2Z1RearVisionV3GaitFineTuneAMPEnvCfg
):
    """Mix straight paths and exact left/right arcs for deployment fine-tuning."""

    rewards: RearCurveGaitHeadingV5RewardsCfg = RearCurveGaitHeadingV5RewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Each environment receives either an exact straight path or a
        # constant-curvature arc. The 40% straight share gives heading hold a
        # strong, unambiguous training signal; the remaining paths split
        # evenly between left and right turns with deployment-relevant radii.
        self.commands.ee_target.synthetic_path_family = "straight_arc"
        self.commands.ee_target.synthetic_straight_path_fraction = 0.40
        self.commands.ee_target.synthetic_arc_radius_range_m = (6.0, 12.0)
        self.commands.ee_target.synthetic_speed_range_mps = (0.22, 0.38)
        self.commands.ee_target.resampling_time_range = (14.0, 20.0)

        # Keep steering smooth enough for the physical platform while still
        # covering the angular rate needed by a 6 m arc at 0.38 m/s.
        self.commands.ee_target.path_lookahead_m = 0.90
        self.commands.ee_target.cross_track_gain = 0.50
        self.commands.ee_target.path_heading_gain = 0.90
        self.commands.ee_target.max_yaw_rate_rps = 0.18
        self.commands.ee_target.minimum_forward_tangent = 0.35
        self.commands.ee_target.command_transition_time_s = 4.0
        self.commands.ee_target.velocity_filter_time_constant_s = 0.40

        # Improve the intermittent limp without forcing a rigid foot phase.
        # AMP remains the dominant motion prior; these terms only bias the two
        # diagonal pairs toward matching contact/air timing.
        self.rewards.trot_diagonal_sync.weight = 0.60
        self.rewards.air_time_variance.weight = -0.45
        self.rewards.feet_height.weight = 0.30
        self.rewards.track_lin_vel_xy.weight = 3.0
        self.rewards.track_ang_vel_z.weight = 1.75


@configclass
class B2Z1RearCurveGaitHeadingV5FineTuneAMPEnvCfg_Visual(
    B2Z1RearCurveGaitHeadingV5FineTuneAMPEnvCfg
):
    """32-environment visual validation of the V5 path distribution."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.ee_target.debug_vis = True


@configclass
class B2Z1RearCurveGaitHeadingV5FineTuneAMPEnvCfg_Play(
    B2Z1RearCurveGaitHeadingV5FineTuneAMPEnvCfg
):
    """V5 policy playback on a physical curved crack using live RGB-D vision."""

    scene: B2Z1RearVisionSceneCfg = B2Z1RearVisionSceneCfg(
        num_envs=1, env_spacing=4.0
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 3600.0
        self.reference_state_initialization_prob = 0.0
        # File mode exports the built-in front RGB-D stream and consumes the
        # RHACrackNet trajectory JSON. Therefore neither the synthetic arc nor
        # privileged curve geometry can influence playback.
        self.commands.ee_target = _rear_command("file", debug_vis=True)
        # Match the command dynamics seen in V5 synthetic training. The old
        # visual defaults allowed 0.5 rad/s yaw and reached walking speed in
        # less than a second, which can fold the rear legs at policy startup.
        # Cruise inside the 0.22--0.38 m/s range seen by the V5 checkpoint.
        self.commands.ee_target.path_speed_mps = 0.35
        self.commands.ee_target.path_lookahead_m = 0.90
        self.commands.ee_target.cross_track_gain = 0.50
        self.commands.ee_target.path_heading_gain = 0.90
        self.commands.ee_target.max_yaw_rate_rps = 0.18
        self.commands.ee_target.command_transition_time_s = 4.0
        self.commands.ee_target.velocity_filter_time_constant_s = 0.40
        self.commands.base_velocity.curriculum_enabled = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_dynamics = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "roll", "pitch", "yaw")
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "DLAA"
        self.sim.render.enable_dl_denoiser = True
        self.sim.render.enable_shadows = False

        _configure_vision_playback_cameras(self)


@configclass
class B2Z1RearDeploymentFineTuneAMPEnvCfg_Play(B2Z1RearDeploymentFineTuneAMPEnvCfg):
    """Deterministic nominal-stand playback for deployment checkpoint selection."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 120.0

        # Exercise the deployment gap explicitly: begin from the nominal
        # stationary pose instead of dropping the robot into a motion frame.
        self.reference_state_initialization_prob = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.ee_target = _rear_command("synthetic", debug_vis=True)
        self.commands.ee_target.synthetic_speed_range_mps = (0.30, 0.30)
        self.commands.ee_target.synthetic_lateral_amplitude_range_m = (0.12, 0.12)
        self.commands.ee_target.synthetic_wavelength_range_m = (6.0, 6.0)

        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_dynamics = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class B2Z1RearSyntheticAMPEnvCfg_Play(B2Z1RearSyntheticAMPEnvCfg):
    """Deterministic playback using the synthetic path command from training."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 120.0
        self.commands.ee_target = _rear_command("synthetic", debug_vis=True)
        self.commands.base_velocity.curriculum_enabled = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_dynamics = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.reference_state_initialization_prob = 1.0
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class B2Z1RearVisionEnvCfg_Play(B2Z1RearSyntheticAMPEnvCfg):
    """Single-environment RGB-D playback fed by RHACrackNet trajectory JSON."""

    scene: B2Z1RearVisionSceneCfg = B2Z1RearVisionSceneCfg(num_envs=1, env_spacing=4.0)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        # Playback is operator-controlled. The former 120 s timeout reset the
        # robot in the middle of a walk and looked like a sudden crouch every
        # two minutes. Keep a one-hour safety horizon instead.
        self.episode_length_s = 3600.0
        self.commands.ee_target = _rear_command("file", debug_vis=True)
        self.commands.base_velocity.curriculum_enabled = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_dynamics = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # The simulated crack is aligned with world +X. Deterministic playback
        # must therefore start B2 at the origin with zero yaw and velocity.
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        # Start and restart from the nominal standing configuration. Random
        # AMP-frame initialization can select a low leg phase that looks like
        # an unexplained squat during continuous playback.
        self.reference_state_initialization_prob = 0.0
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None
        self.rerender_on_reset = True
        # The previous OFF/performance combination exposed stochastic RTX
        # lighting samples as white speckles. DLAA plus the renderer denoiser
        # produces a stable image for segmentation without changing physics.
        self.sim.render.antialiasing_mode = "DLAA"
        self.sim.render.enable_dl_denoiser = True
        self.sim.render.samples_per_pixel = 4
        self.sim.render.enable_shadows = False
        _configure_vision_playback_cameras(self)
