"""VA15 policy with a physical crack and external RGB-D segmentation."""
import math
import os

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg

from b2_z1.assets.unitree import B2_Z1_GUN_CFG

from .b2_z1_rear_vision_env_cfg import B2Z1RearVisionSceneCfg, CRACK_DEPTH_M, _configure_vision_playback_cameras
from .b2_z1_va_env_cfg import B2Z1VA15CurriculumAMPEnvCfg_Play


@configclass
class VA15VisionSceneCfg(B2Z1RearVisionSceneCfg):
    # Retain the trained robot/tool; no additional tool collider is introduced.
    welding_gun = None


@configclass
class B2Z1VA15VisionAMPEnvCfg_Play(B2Z1VA15CurriculumAMPEnvCfg_Play):
    scene: VA15VisionSceneCfg = VA15VisionSceneCfg(num_envs=1, env_spacing=4.0)

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.ee_target
        # Preserve trained workspace, TCP standoff and command smoothing.
        # Only replace the source of path points with validated camera output.
        command.mode = "file"
        command.use_capture_pose = True
        # Simulated depth samples the trench bottom; standoff is above the road.
        command.file_surface_offset_m = CRACK_DEPTH_M
        command.synthetic_world_fixed_path = False
        command.resampling_time_range = (1.0e6, 1.0e6)
        command.vision_exchange_directory = os.environ.get(
            "B2_Z1_VISION_EXCHANGE", "/tmp/b2_z1_va15_vision"
        )
        command.trajectory_file = os.path.join(command.vision_exchange_directory, "latest_trajectory.json")
        command.vision_camera_name = "front_depth"
        self.scene.front_depth.update_latest_camera_pose = True
        command.path_speed_mps = 0.22
        command.path_lookahead_m = 1.0
        command.cross_track_gain = 0.50
        command.path_heading_gain = 0.90
        command.max_yaw_rate_rps = 0.18
        command.minimum_forward_tangent = 0.35
        command.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "DLAA"
        self.sim.render.enable_dl_denoiser = True
        self.sim.render.enable_shadows = False
        _configure_vision_playback_cameras(self)


@configclass
class VA15LoadedGunSceneCfg(VA15VisionSceneCfg):
    """Gun mesh, inertia and contact from the existing physical gun URDF."""

    robot: ArticulationCfg = B2_Z1_GUN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class B2Z1VA15LoadedGunVisionAMPEnvCfg_Play(B2Z1VA15VisionAMPEnvCfg_Play):
    """Isolated loaded-gun playback, leaving the original VA15 task intact."""

    scene: VA15LoadedGunSceneCfg = VA15LoadedGunSceneCfg(num_envs=1, env_spacing=4.0)

    # Effective unsupported hose portion; replace these estimates after the
    # physical hose mass and routing are measured. Units are explicit.
    hose_effective_mass_kg: float = 3.0
    hose_attachment_gun_frame_m: tuple[float, float, float] = (0.0, -0.03, 0.48)
    hose_truck_anchor_env_m: tuple[float, float, float] = (0.0, -2.0, 2.8)
    hose_slack_length_m: float = 4.0
    hose_tension_stiffness_n_per_m: float = 45.0
    hose_tension_damping_ns_per_m: float = 8.0
    hose_linear_drag_ns_per_m: float = 4.0
    hose_acceleration_filter_s: float = 0.08
    hose_max_acceleration_mps2: float = 8.0
    hose_max_force_n: float = 250.0

    def __post_init__(self):
        super().__post_init__()
        self.hose_effective_mass_kg = float(os.environ.get("B2_Z1_HOSE_MASS_KG", self.hose_effective_mass_kg))
        if not math.isfinite(self.hose_effective_mass_kg) or not 1.0 <= self.hose_effective_mass_kg <= 10.0:
            raise ValueError("B2_Z1_HOSE_MASS_KG must be within [1, 10] kg")
        anchor = os.environ.get("B2_Z1_HOSE_TRUCK_ANCHOR_XYZ_M")
        if anchor:
            values = tuple(float(v.strip()) for v in anchor.split(","))
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError("B2_Z1_HOSE_TRUCK_ANCHOR_XYZ_M must be x,y,z in meters")
            self.hose_truck_anchor_env_m = values
        self.hose_slack_length_m = float(os.environ.get("B2_Z1_HOSE_SLACK_M", self.hose_slack_length_m))
        if not math.isfinite(self.hose_slack_length_m) or self.hose_slack_length_m <= 0.0:
            raise ValueError("B2_Z1_HOSE_SLACK_M must be positive")
