"""VA15 policy with a physical crack and external RGB-D segmentation."""
import os

from isaaclab.utils import configclass

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
