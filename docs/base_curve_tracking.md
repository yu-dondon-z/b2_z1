# Base 地面曲线跟踪

`base_velocity` 使用 `CurveVelocityCommandCfg`。独立地面路径复用现有 EE 曲线的三次 Bézier 插值形式，控制点在重采样时固定到世界坐标系；EE 的位置、姿态和跟踪误差不参与底盘指令计算。

## 指令计算

每个控制步执行：

1. 将世界坐标系下的地面曲线点转换到安装在 base 上的理想相机坐标系。
2. 投影到像素，筛选图像边界和深度范围内的点。
3. 根据当前相机姿态、高度和地面平面，将像素射线重新投影到地面。
4. 在水平本体坐标系内，使用最近可见线段估计曲线切线、横向偏差和局部曲率。
5. 输出 `[forward_speed, 0, yaw_rate]`，保持现有 3 维速度观测和速度跟踪奖励接口。

定义横向误差 `e_y` 在机器人位于曲线左侧时为正，切线相对机身前向的角度为 `e_psi`：

```text
heading_error = wrap_to_pi(e_psi - atan2(e_y, lookahead_distance))
yaw_rate = clip(heading_control_stiffness * heading_error + forward_speed * curvature,
                ranges.ang_vel_z[0], ranges.ang_vel_z[1])
```

前向相机通常看不到机身正下方，横向误差采用最近可见线段的切线向机身方向外推估计。此近似适用于平滑曲线；较急弯道、较远的近端盲区会增加误差。

## 参数

在 `mdp/commands/command_cfg.py` 的 `CurveVelocityCommandCfg` 配置默认值，在环境的 `commands.base_velocity` 覆盖：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `forward_speed` | `0.4` | 机身前向速度指令，m/s |
| `lookahead_distance` | `0.6` | 横向纠偏距离尺度，m；减小会加快回归 |
| `heading_control_stiffness` | `1.5` | 角度误差到角速度的增益，1/s |
| `ranges.ang_vel_z` | 环境中 `(-0.8, 0.8)` | yaw 角速度限幅，rad/s |
| `trajectory_num_points` | `96` | 曲线采样点数 |
| `trajectory_curve_offset_scale` | `0.1` | 横向控制点偏移占路径长度的比例；设为 `0` 可验证直线 |
| `resampling_time_range` | 环境中 `(10, 10)` | 独立路径重采样周期，s |
| `camera_position_b` | `(0.25, 0, 0.05)` | 相机相对 base 的安装位置，m |
| `camera_rpy_b` | `(0, 35°, 0)`，以弧度存储 | 相机安装姿态；正 pitch 向下看 |
| `camera_intrinsics` | `(320, 320, 320, 240)` | `fx, fy, cx, cy`，像素 |
| `camera_image_size` | `(640, 480)` | 图像宽、高 |
| `camera_depth_range` | `(0.05, 5)` | 相机前向深度范围，m |
| `camera_update_period` | `0.1` | Play 的 RGB 数据更新周期，s；不影响几何控制更新频率 |
| `ground_height` | `0` | 相对于各环境原点的地面高度，m |
| `ground_curve_width` | `0.04` | 地面黄色线带宽度，m |
| `ground_curve_thickness` | `0.002` | 线带厚度，m；仅用于渲染，没有碰撞 |
| `ground_curve_z_offset` | `0.002` | 线带中心高于地面的距离，m，避免闪烁 |

相机安装坐标使用 x 前、y 左、z 上。对应的光学坐标为 x 右、y 下、z 前，即 `(-y, -z, x)`。投影使用 base 的完整姿态，控制误差使用仅包含 yaw 的水平坐标系，因此会补偿 roll、pitch 和机身高度变化。

控制器仍采用理想曲线像素观测，不包含遮挡、畸变和图像检测误差。安装参数是可调的示例值，应替换为实际标定结果。没有连续可见线段时，yaw 角速度指令为零，前向速度指令保持不变；`path_visible` 指标记录这种情况。

## 训练与 Play 的实际相机

场景中的 `base_camera` 安装在 `{ENV_REGEX_NS}/Robot/base_link/BaseCamera`。相机生成与几何控制共用 `commands.base_velocity.camera_*` 参数，环境初始化时会在 Hydra/命令行覆盖后重新同步相机配置。

| 模式 | 场景配置类型 | 相机行为 |
| --- | --- | --- |
| 训练 | `AssetBaseCfg` + `PinholeCameraCfg` | 添加实际 USD Camera；不创建 RGB 传感器、render product 或图像 annotator，也不因该相机启用 RTX 传感器渲染 |
| Play | `CameraCfg` + 同一 `PinholeCameraCfg` | 创建 RGB 传感器；GUI 下显示环境 0 的实际相机图像 |

只把 `CameraCfg.update_period` 设为很大并不能关闭相机渲染，因此训练使用普通 USD 相机配置。训练时由 base 位姿和固定安装外参计算当前相机位姿，直接进行几何投影；两种模式都不读取 RGB 来生成 yaw，策略观测维度不变。

当前版本的 Isaac Lab 渲染相机使用方形像素和居中主点。为了保持渲染与几何投影一致，相机生成器要求 `fx == fy`、`cx == width / 2`、`cy == height / 2`，不支持时会直接报错。

```bash
# 训练：无需 --enable_cameras。
python scripts/rsl_rl/train.py --task Unitree-b2z1-Velocity --num_envs 4096 --headless

# Play：脚本会自动启用相机；使用 1 个环境可避免多路 RGB 渲染开销。
python scripts/rsl_rl/play.py --task Unitree-b2z1-Velocity-Play --num_envs 1 --checkpoint /path/to/model.pt
```

Play 窗口显示的是实际渲染图像，会受到机器人和场景物体遮挡；几何控制当前不模拟遮挡。Play 默认开启 `base_velocity.debug_vis`，以宽 4 cm 的连续黄色线带显示地面目标曲线，主视口和本体相机都能看到。线带固定在世界坐标系，随曲线重采样更新，不随机器人移动；它没有碰撞，不改变地面接触。训练默认关闭 base 曲线可视化。Headless Play 不创建图像窗口，RGB 数据仍可从 `env.scene.sensors["base_camera"].data.output["rgb"]` 获取。

## 查看与验证

开启 `base_velocity.debug_vis` 可显示地面曲线和目标 heading 箭头。数据也可通过 `env.command_manager.get_term("base_velocity")` 获取：

- `curve_points_w`：世界坐标系下固定的路径点。
- `camera_pixels`、`camera_visible`：投影像素及可见性掩码。
- `perceived_curve_points_w`：由可见像素恢复的地面点，仅掩码内的点有效。
- `lateral_error`、`heading_error`、`curvature`：控制器使用的曲线误差。
- `heading_target`：世界坐标系下的目标 yaw，rad。

训练和 Play 共用固定速度与曲线 yaw 设置，原有 base 速度范围课程已关闭。固定的是速度指令，实际速度仍由训练得到的策略跟踪。

CPU 数值测试（需要 PyTorch，无需启动 Isaac Sim）：

```bash
python -m unittest discover -s tests -p test_curve_tracking.py -v
```

测试覆盖投影回算、视野筛选、转向符号、曲率、坐标变换、部分环境重采样，以及简化运动学闭环；不代替 Isaac Sim 动力学验证。
