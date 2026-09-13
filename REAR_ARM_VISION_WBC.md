# Rear-facing Z1 and visual trajectory WBC

## v4 snapshot (2026-09-13)

Git tag `v4` / branch `release/v4` saves the current rear-vision playback,
startup target-hold fix, diffuse shadow-free lighting, reduced camera/render
cost, and lightweight cylindrical welding-gun collider. The training task's
internal V5 name is retained; the published repository version is `v4`.

Playback task: `Unitree-b2z1-AMP-Rear-Curve-Gait-Heading-V5-Finetune-Play`.
Validated local policy:
`logs/rsl_rl/b2_z1_amp/2026-09-13_14-31-05_rear_curve_gait_heading_v5_from10000/model_10300.pt`.
SHA-256: `a132e55b3e23b1e92ba05b1706b4be95a5e294658f89b0e9ea9541f2fad49136`.
Policy checkpoints, RHACrackNet weights, training logs, local diagnostic
scratch files and backup datasets are not included in this source snapshot.
The playback and physical-contact validation results are recorded below.

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
  -> optional rear-camera diagnostics check the tool/crack area
```

The velocity command is derived from the forward path tangent. It is not
derived from the vector to the rear tool target, which would make B2 walk
backward.

Visual playback includes a silver cylindrical welding gun attached to
`Robot/link_gun`: length 1.10 m and radius 0.028 m, matching the old gun URDF's
collision dimensions. Its lower face centre is aligned with the existing
`dummy_target` TCP at `(0, -0.03, -0.62)` in the gun frame. The visible cylinder
is also its physical collider (2 mm contact offset, zero rest offset).
It belongs to the existing tool rigid body and adds no separate body or mass;
the trained model's 0.001 kg tool-link mass and joint transforms are retained.
It collides with the road and other environment colliders; robot self-collision
follows the existing disabled setting. Synthetic training and the legacy gun
asset are unchanged. Restart visual playback to load the cylinder.

Validation with `model_10300`, front RGB-D and the CPU RHACrackNet bridge:
20 simulated seconds (73.53 wall-clock seconds, headless) advanced 3.69 m
without resets. After the first second, base height was 0.496--0.524 m and
maximum tilt was 2.96 degrees. A separate CPU PhysX contact check measured
zero gun contact before adding a test obstacle and 78.58 N peak contact after
intersecting the tip with it. It also verified the 0.001 kg link mass and TCP
alignment. Reproduce that isolated check without running the vision bridge:

```bash
conda activate env_isaaclab
python scripts/vision/check_welding_gun.py --headless --device cpu
```

The add-on D435i is provisionally mounted at `(x, y, z) = (0.48, 0.0, 0.34)`
metres in `base_link` and pitched about 45 degrees toward the road. This needs
a short front/top bracket; its final transform must come from hand-eye
calibration rather than these nominal CAD values.

## Tasks

- `Unitree-b2z1-AMP-Rear-Synthetic`: no RTX cameras; generates curved rear
  targets and forward tangents for efficient policy training.
- `Unitree-b2z1-AMP-Rear-Vision-Play`: one environment with the selected RGB-D
  stream (front depth camera by default). It contains a light simulated
  two 190 m-long collision road slabs separated by a real 2 cm-wide,
  5 cm-deep trench, then reads RHACrackNet JSON. The dark 200 x 200 m lower
  ground forms the trench bottom.

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

Visual playback creates only the camera selected by `B2_Z1_VISION_CAMERA`
(`front_depth` by default; `d435i` is also supported). It renders every three
policy steps: approximately 16.7 Hz, matching RGB-D export. Physics remains
200 Hz and policy inference remains 50 Hz. Set `B2_Z1_VISION_ALL_CAMERAS=1`
to create all five camera sensors for diagnostics; this increases rendering
and GPU memory costs. Restart Isaac playback to apply these settings.

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

Status also reports `base_linear_velocity_b` (measured velocity per simulated
second), `simulation_time_s`, and `real_time_factor` (simulated seconds divided
by wall-clock seconds between status updates; initially null). A factor of
0.16 means roughly 6.25 real seconds per simulated second. `--real-time` caps
fast playback; it cannot accelerate an overloaded simulator. The 0.35 m/s
setting is a command, not a measurement: the previous 20-s simulation averaged
about 0.218 m/s forward body velocity after startup. Before camera optimization,
live GUI playback advanced only 2.34 simulated seconds in 15 wall-clock seconds,
with observed wall-clock progress around 0.037 m/s. A post-change GUI speed
measurement is still needed; a second validation instance exhausted GPU memory
while the existing playback was running.

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
direction checks for three consecutive frames. Visual playback uses a uniform
white dome light and disables cast shadows, so the robot silhouette does not
need to be removed by changing the centerline used by the policy.

Depth-based shadow filtering is experimental and **off by default**. The default
uses the original centerline extraction and raw-mask area checks. Use
`--depth-shadow-filter` only for an explicit comparison. In that
mode, the RGB mask is intersected with aligned-depth depressions and at least
60% of centerline probes must be 15 mm deeper than symmetric road samples
6 px to either side. `--no-depth-shadow-filter` explicitly selects the default
behavior. Rejected results are removed; the controller uses its configured
stale timeout and velocity filter to stop. Restart both the bridge and Isaac
playback for the default filtering and lighting changes to take effect.

While the observed crack has not yet reached the rear workspace, the tool
holds the body-relative target captured at reset, bounded by the trained arm
workspace. On a later loss of tracking it holds the last commanded target.
The waiting target must not be refreshed from the measured TCP every step:
that removed the position error needed to correct arm drift. With continuous
valid path input, a `model_10300` physics diagnostic reproduced a lateral target
drift above 0.5 m and a collapse after approximately 0.3 m of travel. Holding a
fixed target kept the same checkpoint standing through a 15-second comparison,
including takeover of the cached rear path. A 20-second headless run (simulation
time, before the camera/rendering optimization) at the
current 0.35 m/s cruise command with the actual front RGB-D camera, RHACrackNet
bridge (CPU inference), and diffuse lighting advanced 3.70 m with no reset.
After the first second, base height stayed within 0.495--0.527 m and maximum
tilt was 2.98 degrees; cached rear-path tracking started at 8.08 s. This checks startup and initial takeover,
not long-duration tracking. Controller status also records `base_position_w`
and `base_projected_gravity_b` to make startup failure visible.

Locomotion uses pure pursuit rather than blindly copying the crack tangent.
V5 visual playback commands a 0.35 m/s cruise speed, within its trained
0.22--0.38 m/s range, with a four-second smooth startup transition.
It combines tangent heading with lateral error at a forward look-ahead point,
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

It explicitly enables all five cameras and saves front/rear RGB PNGs,
front/rear uint16 millimetre depth PNGs, and the D435i RGB/depth pair.

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
