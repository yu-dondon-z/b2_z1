# Rear-facing Z1 and visual trajectory WBC

This version rotates the Z1 installation 180 degrees and keeps crack
perception outside the policy network. The policy observation remains 70-D:
the existing 7-D end-effector command is now generated from a visual path.

## Control flow

```text
front RGB-D / front-down D435i
  -> bounded RGB-D file bridge (image + depth + calibration metadata)
  -> RHACrackNet segmentation and 3-D crack waypoints in b2_base
  -> bridge/latest_trajectory.json
  -> transform new points into world and cache them
  -> follow fresh forward path tangent with B2
  -> wait until cached crack enters x=[-1.0, -0.55] m behind B2
  -> command rear-facing Z1 TCP to that point (+0.10 m tool height)
  -> rear RGB-D checks the tool/crack area
```

The velocity command is derived from the forward path tangent. It is not
derived from the vector to the rear tool target, which would make B2 walk
backward.

The add-on D435i is provisionally mounted at `(x, y, z) = (0.48, 0.0, 0.34)`
metres in `base_link` and pitched about 45 degrees toward the road. This needs
a short front/top bracket; its final transform must come from hand-eye
calibration rather than these nominal CAD values.

## Tasks

- `Unitree-b2z1-AMP-Rear-Synthetic`: no RTX cameras; generates curved rear
  targets and forward tangents for efficient policy training.
- `Unitree-b2z1-AMP-Rear-Vision-Play`: one environment with front/rear RGB,
  front/rear depth, and optional D435i RGB-D. It contains a light simulated
  190 x 190 m visual road and a 5 cm-wide dark crack, then reads RHACrackNet
  JSON. The underlying physical ground remains 200 x 200 m.

The visual task exchanges data in this directory by default:

```text
/tmp/b2_z1_vision_bridge
```

Override it without editing code:

```bash
export B2_Z1_VISION_EXCHANGE=/absolute/path/to/shared_directory
# Optional if the result JSON is stored somewhere else:
export B2_Z1_VISION_TRAJECTORY=/absolute/path/to/latest_trajectory.json
```

During visual playback the WBC side publishes `latest_frame.json` as the
atomic commit marker. It names one of four RGB/depth ring-buffer slots and
contains the 3x3 intrinsics plus the ROS optical camera pose in `base_link`.
The external RHACrackNet process should read that metadata, segment the named
RGB image, use the named uint16 millimetre depth image to back-project pixels,
transform the points with `camera_position_b` and
`camera_quaternion_b_ros_wxyz`, and atomically replace
`latest_trajectory.json`.

That external process is included and uses the already trained stage-1
checkpoint by default:

```bash
source /home/dong/anaconda3/etc/profile.d/conda.sh
conda activate crack_vision
cd /home/dong/b2_z1

python scripts/vision/run_rhacracknet_bridge.py \
  --exchange-dir /tmp/b2_z1_vision_bridge \
  --save-overlay \
  --display
```

The live OpenCV panel shows, from left to right, the D435i RGB image,
RHACrackNet mask, and extracted centerline. Press `q` or `Esc` to close it.
The latest annotated frame is also saved as
`/tmp/b2_z1_vision_bridge/latest_overlay.png`.

In the Isaac Sim window:

- the large red sphere is the commanded extended tool/TCP target;
- the small red spheres are the accepted RGB-D crack points in world space;
- the coordinate frames show commanded and actual tool poses.

No red path points means RHACrackNet has not produced a fresh, geometrically
valid trajectory; in that state the WBC intentionally commands zero velocity.
For a text-level check, inspect
`/tmp/b2_z1_vision_bridge/latest_controller_status.json`: its
`source_sequence_consumed`, `trajectory_fresh`, velocity command, cached point
count, and tool target prove whether the controller has consumed the latest
visual result.

Run it alongside the Isaac Sim visual-play process. If no crack or fewer than
three valid depth waypoints are found, it deliberately stops refreshing the
trajectory; the WBC stale timeout then commands zero base velocity.
Short detections are also rejected unless the 3-D path is at least 0.20 m long
and spans at least 0.15 m in B2's forward direction. The WBC independently
rejects a path whose normalized tangent has less than 0.65 forward component.

For the current simulated road, the default segmentation threshold is `0.90`
(the measured true-crack probability is above `0.99`, while uniform road stays
below `0.69`). A result must also pass mask-area, waypoint-confidence,
near-field visibility, image-span, valid-depth, path-straightness and forward
direction checks for three consecutive frames. A rejected/missing result is
removed immediately; the controller's 0.25 s stale timeout then commands zero
velocity.

Locomotion uses pure pursuit rather than blindly copying the crack tangent.
It combines tangent heading with lateral error at a 1 m look-ahead point,
commands no lateral velocity, rotates in place for heading errors above 0.45
rad, and only translates once the crack is back inside the recoverable field
of view.

The Isaac Sim process (run in a second terminal) is:

```bash
source /home/dong/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/dong/b2_z1

python scripts/rsl_rl/play_amp.py \
  --task Unitree-b2z1-AMP-Rear-Vision-Play \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/b2_z1_amp/2026-09-01_09-10-14_rear_vision_resume_from_5500/model_7999.pt \
  --enable_cameras \
  --real-time
```

VS Code can start both processes together with the Run and Debug selection
`后向机械臂+视觉：完整闭环播放（一键启动）`.

Expected JSON fields are:

```json
{
  "frame_id": "b2_base",
  "control_waypoints_robot_xyz": [[0.5, 0.0, -0.45], [0.6, 0.0, -0.45]]
}
```

Atomic replacement of the JSON file is recommended. If it is absent, invalid,
or older than 0.25 s, the visual command sets base velocity to zero and holds
the last tool target.

## Validate camera output

```bash
source /home/dong/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/dong/b2_z1

python scripts/vision/check_rear_vision_scene.py \
  --headless \
  --device cuda:0 \
  --rendering_mode performance \
  --output outputs/rear_vision_camera_check
```

It saves front/rear RGB PNGs, front/rear uint16 millimetre depth PNGs, and the
D435i RGB/depth pair.

## Training

The old forward-facing policy is not valid for the changed Z1 kinematics and
negative-x rear workspace. Start by training or fine-tuning with the synthetic
task; do not use the five RTX cameras across parallel training environments.

```bash
source /home/dong/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/dong/b2_z1

python scripts/rsl_rl/train_amp.py \
  --task Unitree-b2z1-AMP-Rear-Synthetic \
  --headless
```

Before physical deployment, replace all simulated camera intrinsics/extrinsics
with calibration results and confirm whether the add-on device is D435i or a
different model called D35i.
