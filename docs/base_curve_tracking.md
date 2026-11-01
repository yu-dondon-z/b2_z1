# Base 地面曲线跟踪

`base_velocity` 使用 `CurveVelocityCommandCfg`。环境默认生成含两到三段连续转折的地面路径，训练逐步增加转折数和角度，Play 使用三段 `90°` 转折的最高难度；也可选择原有三次 Bézier 采样。路径在重采样时固定到世界坐标系，EE 的位置、姿态和跟踪误差不参与底盘指令计算。

## AMP 步态

`Unitree-b2z1-Velocity` 和 `Unitree-b2z1-Velocity-Play` 现在使用从 `origin/amp_loco`（`3d279d8`）移植的 AMP-PPO。原有训练与 Play 脚本会根据 `agent_cfg.amp_cfg` 自动选择 AMP runner 和 wrapper，底盘曲线指令、机械臂观测和 18 维动作接口保持原有形式。

`datasets/b2_motion/` 包含该分支的 19 份 B2 动作数据。判别器使用相邻两帧的 43 维观测：腿关节位置 12、足端本体位置 12、本体线速度 3、角速度 3、腿关节速度 12、离地高度 1。关节和足端通过名字显式排列为 `FL, FR, RL, RR`，每条腿关节依次为 `hip, thigh, calf`；机械臂不进入动作先验。

已移除 `feet_gait`、`feet_height`、`air_time_variance`、三项 `stand_still_*`、`hip_deviation`、`joint_deviation`、固定 `base_height_l2` 和 `flat_orientation_l2`。保留速度跟踪、机械臂姿态和末端稳定、打滑、关节限位、接触及动作平滑等奖励。步态配对、摆腿高度和本体姿态由参考动作先验学习。

AMP 参数在 `agents/rsl_rl_ppo_cfg.py` 中设置：`amp_reward_coef=0.2`、`amp_task_reward_lerp=0.8`、判别器隐藏层 `[1024, 512]`。总奖励为 `0.8 * task_reward + 0.2 * amp_reward`，其中 `amp_reward` 已乘 `amp_reward_coef`；任务奖励仍由 Isaac Lab 乘控制步长计算。

训练默认以概率 1.0 从参考动作初始化本体和腿关节，机械臂沿用默认重置。写入参考状态并刷新运动学后，底盘和机械臂两条世界坐标曲线都会重新采样。Play 默认使用正常站立初始化。终止前的 AMP 状态单独保存，避免跨 episode 的跳变进入判别器；检查点同时保存策略、判别器、优化器和 AMP 归一化统计。

新实验目录为 `logs/rsl_rl/b2_z1_curve_amp`。需要重新训练以获得 AMP 步态；旧 PPO 检查点可以使用 Play 加载策略，但不会因为奖励配置改变而自动产生新步态。训练续跑需要包含判别器状态的新 AMP 检查点。

## 后向机械臂与 EE 指令

`assets/b2_z1/b2_z1_gun.urdf` 中，`base_link -> link00` 的固定安装关节 `joint0` 绕 Z 轴旋转 `pi`，使整套 Z1 机械臂及工具朝向 B2 后方。安装位置仍为 `(0, 0, 0.09)` 米，六个运动关节沿用现有初始角度；正常重置和 AMP 参考动作重置都使用这一安装方向。

`ee_target` 直接在 `base_name="base_link"` 中采样 XYZ 和 RPY，Bézier 起点、终点、控制点及实际轨迹记录也统一使用完整 base 坐标系。Z 是相对 base 原点的高度；位置指令直接在 base 中插值，世界坐标仅用于误差计算和可视化。底盘平移、升降和 roll/pitch/yaw 变化时，整条 EE 目标曲线随 base 变换。命令和策略观测仍为原有的 7 维 base 相对位姿。

机械臂后向安装，因此工作区设置在 base 的负 X 方向。采样范围如下，单位为米：

| 参数 | 训练初始 `ranges` | 最终 `limit_ranges` |
| --- | --- | --- |
| `pos_x` | `(-1.03, -1.01)` | `(-1.08, -0.96)` |
| `pos_y` | `(-0.02, 0.02)` | `(-0.15, 0.15)` |
| `pos_z` | `(-0.31, -0.29)` | `(-0.34, -0.26)` |

姿态固定为 base RPY `(0, pi/2 - 0.45, pi)`，约 `(0°, 64.2°, 180°)`，接近默认机械臂姿态，让课程主要学习位置曲线跟踪。这里的负 Z 表示末端低于 base 原点；例如 base 离地 `0.7 m` 且保持水平时，完整范围对应世界高度约 `0.36–0.44 m`。采样不再经过 `link00` 的安装旋转；原有 `link_name` 采样参数已移除。

范围依据仓库 URDF 和关节限位设置，并在固定目标姿态下检查了 125 个工作区网格点的数值逆运动学解，最大位置残差小于 `0.1 mm`。该检查用于筛选工作区，动力学跟踪表现仍需 Isaac Sim 验证。每次重采样从当前真实 EE 的 base 相对位置连续起步；控制点限制在当前采样范围内，曲线位于起点与控制点、终点的凸包中，避免随机弯曲把控制点推到工作区之外。

训练已启用 `curriculum.pos_cmd_levels`：末端位置跟踪奖励的 episode 均值超过满分的 `80%` 后，XYZ 范围每侧扩大 `0.01 m`，并分别钳制到 `limit_ranges`；姿态扩展步长为零。检查间隔至少一个最大 episode 时长，支持异步重置，升级只影响之后重采样的曲线。新增 `end_effector_position_tracking_exp` 奖励，权重 `5.0`、位置核宽度 `0.1 m`；另以权重 `-0.2` 的姿态误差维持固定朝向。Play 关闭 EE 课程，从首次采样即使用完整 `limit_ranges`。

底盘依旧沿自身正 X 方向前进，前置地面相机和独立底盘曲线保持现有方向。机械臂默认姿态惩罚继续以相同关节角为参考，位置/姿态误差和策略观测继续使用底盘坐标系；新的安装方向需要重新训练策略。

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

## 大角度转折路径

环境默认 `commands.base_velocity.path_type="turn"`，路径由入弯直线、两到三段圆弧转弯、弯间直线及出弯直线组成。默认入弯直线 `0.8 m`，弯间直线 `turn_gap_distance=1.0 m`，圆弧半径 `1 m`。直线与圆弧的切线连续，连接处曲率在 `0` 与圆弧曲率之间切换。

首个转弯随机选择左右方向，之后交替转向，并在每对转弯后回到初始航向，形成连续折线路径。训练完整难度下每个侧向目标从 `turn_angle_range=(pi/4, pi/2)` 采样；返回初始航向的转弯与前一弯角度相反。Play 固定使用三段最大角度转折，例如 `+90°、-90°、+90°`，或其左右镜像。

路径长度按 `forward_speed * time_left + 2 * lookahead_distance` 计算。环境重采样周期增加到 `25 s`，因此默认总长由 `5.2 m` 增加到 `11.2 m`；采样点增至 192，仍按弧长等间距采样。Episode 时长增加到 `30 s`。默认三段 90° 转弯会在约 `7.51 m` 的弧长位置全部完成，留有出弯直线，能够在下一次路径重采样前走过三段转折。

转向角和转弯半径可分别调整：`turn_angle_range` 控制转多少度，`turn_curvature` 控制转弯有多急。默认 `turn_curvature=1.0 m⁻¹`，在 `0.4 m/s` 时圆弧前馈角速度为 `0.4 rad/s`，低于 `0.8 rad/s` 的 yaw 限幅。路径起始切线与机身前向一致，大角度变化发生在前方圆弧中。

路径较短或曲率上限较低时，采样器先缩短弯间直线，再按相同比例缩小各段转向角，保证全部转折共用的弧长预算不超过路径长度，并为出弯直线预留 `lookahead_distance`；入弯直线最多占总长四分之一。路径不会为了容纳转弯而无限延长。`curve_turn_angles` 记录每段实际角度，`curve_turn_count` 记录实际非零转折数。

可通过 `env.commands.base_velocity.path_type=bezier` 使用原有控制点采样。默认路径分布已经改变，旧策略面对大角度转向的动力学表现需验证，并按需要继续训练。

## 曲率课程

训练默认启用 `curve_curvature_cmd_levels`，沿用 `lin_vel_cmd_levels` 的奖励门槛。`turn` 模式将现有尺度转换为难度比例 `trajectory_curve_offset_scale / limit_trajectory_curve_offset_scale`，再乘到采样转向角上，并按难度增加转折数。默认阶段如下：

| 难度比例 | 实际转折数 | 每段转向角范围 |
| --- | --- | --- |
| `0` | `0`，直线 | `0°` |
| `0 < 难度 < 0.5` | `2` | `45°–90°` 乘当前难度比例 |
| `0.5 <= 难度 < 1` | `3` | `45°–90°` 乘当前难度比例 |
| `1`，训练上限 | `3` | `45°–90°` |
| Play 最高难度 | `3` | 每段固定 `90°` |

`turn_count_range=(2, 3)` 定义初始和最大转折数，配置支持 1 到 3 段；`turn_angle_range` 的上界定义最大单段角度。课程尺度被限制在 `limit_trajectory_curve_offset_scale`，达到上限后不再增加转折数或角度。曲率上限独立于课程，不会随等级突破配置值。

- `commands.base_velocity.trajectory_curve_offset_scale=0.0`：初始难度为零，生成直线。
- `commands.base_velocity.limit_trajectory_curve_offset_scale=0.1`：最终难度为 100%。
- `curriculum.curve_curvature_cmd_levels.params.offset_scale_step=0.01`：每次达标增加的尺度。
- `curriculum.curve_curvature_cmd_levels.params.reward_threshold=0.8`：线速度与转向跟踪奖励都必须超过各自权重的 80%。

奖励取本次重置环境的 episode 累积值，除以最大 episode 时长，与原有速度课程一致；提前终止会降低该分数。两次检查至少间隔一个最大 episode 时长（默认 30 s），在间隔结束后的第一次非空重置时检查，因此支持异步重置；未达标时维持当前难度。日志 `Curriculum/curve_curvature_cmd_levels` 记录当前课程尺度。

尺度是无量纲的弯曲难度参数，不是以 m⁻¹ 表示的曲率上限。`bezier` 模式中它仍代表横向控制点偏移与路径长度的比例，实际曲率取决于随机控制点和路径长度。升级只影响之后正常重采样的路径，当前世界坐标路径不会立即改变；前向速度指令为 `0.4 m/s`，yaw 限幅为 `(-0.8, 0.8) rad/s`。

Play 默认设置 `commands.base_velocity.curriculum_enabled=False`、`turn_count_range=(3, 3)`、`turn_angle_range=(pi/2, pi/2)`，命令初始化时即使用最终尺度 `0.1`，从第一次采样起生成最高难度。训练时关闭课程开关也会直接使用最终尺度，角度仍按其配置范围采样；要固定测试直线，将初始尺度和最终尺度都设为 `0.0`。仅移除 curriculum term 会停留在配置的初始尺度。当前检查点沿用原有行为，不保存课程进度，续训会从配置的初始尺度重新开始。

## 几何曲率上限

训练设置 `max_curvature=1.0 m⁻¹`，Play 的硬上限为 `1.5 m⁻¹`。`turn` 模式的实际圆弧曲率为 `min(turn_curvature, max_curvature)`，所以在默认 `turn_curvature=1.0` 时两者都生成半径 `1 m` 的圆弧；直线段曲率为零。`max_curvature=0.0` 生成直线，`None` 关闭额外硬上限，此时转弯仍由 `turn_curvature` 约束。

`bezier` 模式先按偏移尺度生成控制点，再按几何曲率上限缩小横向偏移。上限是约束，不要求每条曲线都达到该值。局部 Bézier 控制点的 X 等间隔，满足 `x'(t)=L`、`x''(t)=0`，因此整条曲线有保守上界：

```text
|κ(t)| = |L * y''(t)| / (L² + y'(t)²)^(3/2)
       <= max(|y''(0)|, |y''(1)|) / L²
```

采样器根据此解析上界缩小横向偏移，覆盖端点和采样点之间的曲线；实际曲率可能低于配置值。世界坐标系中的平移与旋转不改变曲率。参数必须非负且有限，并满足 `forward_speed * max_curvature <= min(-yaw_min, yaw_max)`；该检查只保证曲率前馈处于双向转向限幅内，完整纠偏指令仍按原有方式限幅。

例如固定为左右 `90°` 转向，并把圆弧曲率设为 `1.2 m⁻¹`（半径约 `0.83 m`）：

```bash
python scripts/rsl_rl/play.py --task Unitree-b2z1-Velocity-Play --num_envs 1 \
    --checkpoint /path/to/model.pt \
    'env.commands.base_velocity.turn_angle_range=[1.57079632679,1.57079632679]' \
    env.commands.base_velocity.turn_curvature=1.2
```

## 参数

在 `mdp/commands/command_cfg.py` 的 `CurveVelocityCommandCfg` 配置默认值，在环境的 `commands.base_velocity` 覆盖：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `forward_speed` | `0.4` | 机身前向速度指令，m/s |
| `lookahead_distance` | `0.6` | 横向纠偏距离尺度，m；减小会加快回归 |
| `heading_control_stiffness` | `1.5` | 角度误差到角速度的增益，1/s |
| `ranges.ang_vel_z` | 环境中 `(-0.8, 0.8)` | yaw 角速度限幅，rad/s |
| `trajectory_num_points` | 配置类 `96`；环境 `192` | 曲线采样点数 |
| `path_type` | 配置类 `bezier`；环境 `turn` | 选择原有 Bézier 或直线—圆弧—直线路径 |
| `turn_count_range` | 训练 `(2, 3)`；Play `(3, 3)` | 初始和最大转折数，随课程增加，最多 3 段 |
| `turn_angle_range` | 训练 `(pi/4, pi/2)`；Play `(pi/2, pi/2)` | 完整难度下转向角绝对值范围，rad；转向交替 |
| `turn_curvature` | `1.0` | 圆弧转弯曲率，m⁻¹，受硬上限约束 |
| `turn_start_distance` | `0.8` | 入弯前直线长度，m，短路径中最多为总长的四分之一 |
| `turn_gap_distance` | `1.0` | 转折之间的直线长度，m；短路径中会缩短 |
| `trajectory_curve_offset_scale` | 配置类 `0.1`；训练初始 `0.0` | 课程尺度；`turn` 模式按当前/最终尺度缩放转向角 |
| `limit_trajectory_curve_offset_scale` | `0.1` | 最终课程尺度；关闭课程时直接使用 |
| `max_curvature` | 训练 `1.0`，Play `1.5` | 整条路径的绝对几何曲率上限，m⁻¹；`0` 为直线 |
| `curriculum_enabled` | 训练 `True`，Play `False` | 是否逐级增加曲线弯曲难度 |
| `resampling_time_range` | 环境中 `(25, 25)` | 独立路径重采样周期，s；默认总长 `11.2 m` |
| `camera_position_b` | `(0.41251, 0.024997, 0.04765)` | B2 前深度相机相对 base 的安装位置，m |
| `camera_rpy_b` | `(0, 45°, 0)`，以弧度存储 | 相机安装姿态；正 pitch 向下看 |
| `camera_intrinsics` | `(320, 320, 320, 240)` | `fx, fy, cx, cy`，像素 |
| `camera_image_size` | `(640, 480)` | 图像宽、高 |
| `camera_depth_range` | `(0.05, 5)` | 相机前向深度范围，m |
| `camera_update_period` | `0.1` | Play 的 RGB 数据更新周期，s；不影响几何控制更新频率 |
| `ground_height` | `0` | 相对于各环境原点的地面高度，m |
| `ground_curve_width` | `0.04` | 地面黄色线带宽度，m |
| `ground_curve_thickness` | `0.002` | 线带厚度，m；仅用于渲染，没有碰撞 |
| `ground_curve_z_offset` | `0.002` | 线带中心高于地面的距离，m，避免闪烁 |

相机安装坐标使用 x 前、y 左、z 上。对应的光学坐标为 x 右、y 下、z 前，即 `(-y, -z, x)`。投影使用 base 的完整姿态，控制误差使用仅包含 yaw 的水平坐标系，因此会补偿 roll、pitch 和机身高度变化。

安装位置参考仓库 `assets/b2_z1/b2_z1_gun.urdf` 和[宇树官方 B2 URDF 的 `joint_f_dc`](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/b2_description/urdf/b2_description.urdf)：光学坐标 RPY 约为 `(-135°, 0°, -90°)`，换算到上述相机坐标约定后约为向下 45°，位于头部雷达下方。内参仍为理想针孔参数；上真机时应使用实际标定结果。

控制器仍采用理想曲线像素观测，不包含遮挡、畸变和图像检测误差。没有连续可见线段时，yaw 角速度指令为零，前向速度指令保持不变；`path_visible` 指标记录这种情况。

## 训练与 Play 的实际相机

场景包含前向地面相机和后向 EE 相机，均固定安装在 `base_link` 下：

| 场景名称 | USD 路径 | 标定参数 | 用途 |
| --- | --- | --- | --- |
| `base_camera` | `{ENV_REGEX_NS}/Robot/base_link/BaseCamera` | `commands.base_velocity.camera_*` | 观察前方地面曲线，与几何控制共用标定 |
| `rear_camera` | `{ENV_REGEX_NS}/Robot/base_link/RearCamera` | `commands.ee_target.camera_*` | 观察后方 EE 目标、插值曲线与实际末端轨迹 |

后向安装位置沿用 URDF 的 `joint_r_dc`：`(-0.4081, -0.025, 0.024861) m`；安装 RPY 为 `(0, pi/4, pi)`，即朝 base 负 X 方向并向下看 45°。默认 `640 × 480`、内参 `(320, 320, 320, 240)`、深度范围 `(0.05, 5.0) m`，RGB 更新周期 `0.1 s`。几何视野覆盖当前完整 EE `limit_ranges`、默认末端起点和插值曲线；实际画面仍受机器人与场景遮挡影响。两台相机都跟随 base 的完整位姿，环境初始化时会在 Hydra/命令行覆盖后重新同步配置。

| 模式 | 场景配置类型 | 相机行为 |
| --- | --- | --- |
| 训练 | `AssetBaseCfg` + `PinholeCameraCfg` | 添加实际 USD Camera；不创建 RGB 传感器、render product 或图像 annotator，也不因该相机启用 RTX 传感器渲染 |
| Play | `TiledCameraCfg` + 同一 `PinholeCameraCfg` | 每个相机组分别将各环境的 RGB 相机合并到一个 tiled render product；GUI 下分别显示环境 0 的前向和后向图像 |

`TiledCamera` 通过合并多路相机的渲染，减少多环境 Play 中逐相机创建 render product 和读取图像的开销。输出仍通过 `env.scene.sensors["base_camera"].data.output["rgb"]` 读取，形状为 `[num_envs, height, width, 3]`，数据保留在传感器设备上；GUI 仅把环境 0 的图像复制到 CPU 显示。环境初始化时重新同步标定仍会保留 tiled 传感器类型。

后向图像使用 `env.scene.sensors["rear_camera"].data.output["rgb"]`，格式相同。Play 自动打开 `Rear camera (EE target) — environment 0` 窗口，并启用 `ee_target.debug_vis` 显示目标和轨迹。两台相机的分辨率、内参、安装位姿及更新周期可分别配置；例如降低后向图像开销：

```bash
python scripts/rsl_rl/play.py --task Unitree-b2z1-Velocity-Play --num_envs 1 \
    --checkpoint /path/to/model.pt \
    'env.commands.ee_target.camera_image_size=[320,240]' \
    'env.commands.ee_target.camera_intrinsics=[160,160,160,120]' \
    env.commands.ee_target.camera_update_period=0.2
```

只把渲染相机的 `update_period` 设为很大并不能关闭相机渲染，因此训练时两台相机都使用普通 USD 相机配置。训练时由 base 位姿和固定安装外参计算当前前向相机位姿，直接进行几何投影；两种模式都不读取 RGB 来生成指令，策略观测维度不变。TiledCamera 主要优化多环境 RGB 渲染，具体加速比例需在 Isaac Sim 中实测；增加后向相机仍会增加 Play 的渲染开销。

当前版本的 Isaac Lab 渲染相机使用方形像素和居中主点。为了保持渲染与几何投影一致，相机生成器要求 `fx == fy`、`cx == width / 2`、`cy == height / 2`，不支持时会直接报错。

```bash
# 训练：无需 --enable_cameras。
python scripts/rsl_rl/train.py --task Unitree-b2z1-Velocity --num_envs 4096 --headless

# Play：脚本会自动启用 tiled 相机；可按需增大 --num_envs 批量渲染。
python scripts/rsl_rl/play.py --task Unitree-b2z1-Velocity-Play --num_envs 1 --checkpoint /path/to/model.pt
```

Play 窗口显示的是实际渲染图像，会受到机器人和场景物体遮挡；几何控制当前不模拟遮挡。Play 默认开启 `base_velocity.debug_vis`，以宽 4 cm 的连续黄色线带显示地面目标曲线，主视口和本体相机都能看到。线带固定在世界坐标系，随曲线重采样更新，不随机器人移动；它没有碰撞，不改变地面接触。训练默认关闭 base 曲线可视化。Headless Play 不创建图像窗口，RGB 数据仍可从 `env.scene.sensors["base_camera"].data.output["rgb"]` 获取。

## 查看与验证

开启 `base_velocity.debug_vis` 可显示地面曲线和目标 heading 箭头。数据也可通过 `env.command_manager.get_term("base_velocity")` 获取：

- `curve_points_w`：世界坐标系下固定的路径点。
- `curve_turn_angles`：各段实际转向角，形状 `[num_envs, max_turns]`，未使用的列为零，rad。
- `curve_turn_count`：实际非零转折数；`curve_turn_angle` 为各段角度的带符号总和，rad。
- `curve_path_length`：目标路径总弧长，m。日志 `turn_count`、`turn_angle`、`path_length` 分别记录转折数、最大单段角度绝对值及路径长度。
- `curve_control_points_w`：仅 `bezier` 模式提供控制点张量，`turn` 模式为 `None`。
- `camera_pixels`、`camera_visible`：投影像素及可见性掩码。
- `perceived_curve_points_w`：由可见像素恢复的地面点，仅掩码内的点有效。
- `lateral_error`、`heading_error`、`curvature`：控制器使用的曲线误差。
- `heading_target`：世界坐标系下的目标 yaw，rad。

训练和 Play 共用固定速度与曲线 yaw 设置，原有 base 速度范围课程已关闭。训练逐渐放大转向角并从两段增加到三段转折，Play 固定三段最大角度，两者均施加几何曲率约束。固定的是速度指令，实际速度仍由训练得到的策略跟踪。

CPU 数值测试（需要 PyTorch，无需启动 Isaac Sim）：

```bash
python -m unittest discover -s tests -p test_curve_tracking.py -v

# AMP 数据、插值、训练更新、参考状态重置及检查点测试；还需要 RSL-RL 2.3.x。
python -m unittest discover -s tests -p test_amp.py -v

# 后向安装的 URDF 正运动学、EE 目标/轨迹变换和部分环境重采样测试。
python -m unittest discover -s tests -p test_rear_arm.py -v
```

测试覆盖 tiled 相机配置与标定重建、投影回算、视野筛选、转向符号、曲率、多段路径的总长和曲率约束、课程的门槛/异步更新/次数与角度封顶、坐标变换、部分环境重采样、单段及三段 90° 转向的简化运动学闭环；不代替 Isaac Sim 动力学和渲染性能验证。
