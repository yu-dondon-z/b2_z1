# B2 + Huibo 模型替换交接（2026-11-02）

## 后续更新：正向同轴持枪

用户在同日明确确认要“从上方朝下正向持枪”。当前代码已由下文的侧装
初版改为同轴持枪；下文的早先 GUI、执行器、训练日志均保留为历史，
不能算作新姿态的动态验证。原厂关节轴/限位、B2 底座、执行器增益未改。

- 工具安装现为 `huibo_tool_mount xyz=(0.075,0.03,0), rpy=(0,-π/2,0)`。
  法兰 `+X` 与枪 `-Z` 同向同轴，不再相差 90°。
- `assets/b2_huibo.py` 的新七轴默认角为
  `[1.57079226334755,0.9351052713220801,-1.5751082043382922,1.0912415788479128,-0.8986226998003438,-0.650491566,-0.650491566]`。
  TCP 保持 `(-0.775,0,-0.48) m`；共同轴 `(-0.228191,0,-0.973616)`，
  离竖直 13.1906°。这是向下略向后的正向持枪，不应写成严格竖直。
  各轴距原限位至少 0.10 rad，不硬改腕部限位凑姿态。
- 新 `HUIBO_TCP_ORIENTATION_RPY` 同步更新命令方向，与实际默认 TCP
  旋转一致，避免新关节姿态仍沿用旧 Z1 的方向目标。
- Huibo 专用圆柱长 0.60 m，本地端面 z=-0.62/-0.02；删除会穿腕的
  原占位枪上半段，保留 TCP 和 marker 本地偏置以及后相机安装。
  生成的 `link_gun` 另含 95 mm 长、22 mm 半径的同轴转接件 visual/collision，
  连接法兰与枪体上端；无新刚体、质量或自由度，不代表实际支架设计。
  旧 Z1 枪长 1.10 m 不变，源 ZIP/供应方模型也不改。
- `scripts/assets/test_huibo_upright.py`：真实 URDF 独立 NumPy FK 检查
  TCP 保持、同轴关系、原限位余量、实际倾角、圆柱端面与源模型不变，
  以及方向命令/FK 一致、转接件几何连续性，9 项通过。
  `check_va16_dual_scene.py` 也增加了实际 PhysX 法兰/枪轴方向报告和检查。

用户已确认关闭旧预览，检查旧进程退出后串行启动新渲染，未擅自终止用户进程。
`outputs/b2_huibo_axial_grip_20261102_01/` 首轮因诊断函数条件分支内重复
导入 `quat_apply` 导致未绑定局部变量而失败，日志保留；移除重复局部导入后
`..._02/scene_report.json` 实际验证通过：同轴差 0°、下倾偏竖直 13.1906°、
RGB-D 静态枪尖误差 1.998 mm，前/后感知均 valid，枪碰撞启用、视锥关闭。
02 是补可视转接件前的报告。03 因系统盘写满导致 USD 文件写入失败，
04 尝试 `TMPDIR=/dev/shm` 仍失败：当前 Isaac 的 USD 缓存目录硬编码为
`/tmp/IsaacLab`，不读取该变量。这两次均不能算作模型验证通过。
新增诊断参数 `--usd-cache-dir` 后，最终含转接件的
`outputs/b2_huibo_axial_grip_20261102_05/scene_report.json` **pass=true**，
GUI 截图 `viewport_robot.png` 已人工复核，转接件连续、无视锥。
实际 PhysX 法兰/枪轴差 0°，偏竖直 13.19063°，静态 RGB-D 枪尖误差
0.001997943 m，前/后感知均 valid，TCP `(-0.775001,-0.000001,-0.479999) m`，
枪质量约 .001 kg、碰撞 true，73 观测/19 动作。六个地面采样投影在图像内，
仅说明投影范围，不证明各点 IK、遮挡和动态闭环。该诊断使用 classical
裂缝检测，不是 RHANet 跟踪性能报告。
本轮仅把已结束的 14:24/14:26 转换缓存及 14:28 失败缓存移到 `/dev/shm`
缓解空间不足，没有删除代码、权重、日志或用户数据。内存缓存重启消失，
不影响项目内源模型与报告。05 使用缓存 `/dev/shm/b2_huibo_axial_usd_20261102_05`。
验证后已重新打开保持预览，输出目录 `outputs/b2_huibo_axial_preview_20261102_06/`，
启动日志 `/tmp/b2_huibo_axial_preview_20261102_06.log`，同样验证通过后进入
`--hold`。用户可关闭窗口结束；不要在其运行时再并发启动第二个 Isaac。
默认预览是零重力保持，不等于带重力稳定或可用策略。下文历史报告均保留，
当前参数以本段及 `assets/b2_huibo.py`、生成脚本、任务配置为准。

## 当前任务和完成边界

用户最新要求：不再显示深度相机视野范围；解压本地 `huibo_description_right_hand.zip` 并将 Z1 换成该机械臂。前一阶段固定前进速度 0.20 m/s、前视 RGB 纠偏、后置 D435i 朝后斜下 45°测枪尖和裂缝的需求继续保留。

本轮已完成本地资产替换、七轴控制与训练接线、无视锥 GUI 预览、真实 Isaac 导入检查、逐轴仿真动作响应及 64 环境训练烟测。**尚无可用的 Huibo 行走/跟踪策略。** 不能把旧 Z1 VA15/VA16 的可走表现或闭环精度算作新臂结果。未同步板载、未连接硬件、未启动机器人控制或实机运动。

用户此前将精度定义为 TCP 在地面的投影到裂缝中央的误差，目标至少 5 cm 以内，不应重新写成任意三维 TCP 10 mm 精度。

## 文件与模型来源

仓库：`/home/dong/b2_z1`。Isaac Python：`/home/dong/anaconda3/envs/env_isaaclab/bin/python`；视觉 Python：`/home/dong/anaconda3/envs/crack_vision/bin/python`。当前 GPU 内存有限，应一次只运行一个 Isaac 进程。

原始 ZIP 为 `source/b2_z1/b2_z1/assets/b2_z1/huibo_description_right_hand.zip`，保留。解压目录为同级 `huibo_description/`，供应方原始 URDF 为 `huibo_description/urdf/single_right_hand.urdf`，也保留未改。旧 `b2_z1_no_stick.urdf`、Z1 任务、权重和历史日志均保留。

新增组合资产 `source/b2_z1/b2_z1/assets/b2_z1/b2_huibo.urdf`，由 `scripts/assets/build_b2_huibo.py` 合成。脚本拒绝覆盖源描述；需要重新生成现有组合资产时使用 `--replace-generated`，只替换生成物。

Huibo 有七轴 `right_joint1..7`；原始轴向、位置限位、STL、CAD 质量和惯量沿用供应方描述。CAD 总质量约 1.000429 kg，未称量实物。原始每轴 effort/velocity 均为 0，因此生成的仿真 URDF 和执行器配置显式替换为 30 Nm / 1.5 rad/s；刚度 40、阻尼 2 均为仿真起点，不是厂家参数或实机增益。

## 安装、工具和相机约定

| 变换/设置 | 当前值 |
| --- | --- |
| B2 `base_link` → Huibo `right_arm_base_link` | 位置 `(-0.10,0,-0.014) m`，RPY `(-π/2,0,π)` |
| 底座安装面底部 | B2 坐标系 `z=0.09 m` |
| `right_arm_end_link` → `link_gun` | 位置 `(0.075,0.03,0) m`，RPY `(0,-π/2,0)` |
| 旧 Z1 下落支架 | 去掉原先约 0.20 m 下落偏置 |
| TCP 在 `link_gun` 中 | `(0,-0.03,-0.62) m`，旧定义保留 |
| 后置 D435i 光心 | B2 坐标系 `(-0.445,0,0.005) m`，朝后斜下 45° |
| 相机视锥 | 关闭；相机外壳继续显示 |
| 工具与机械臂碰撞 | 外部碰撞启用；`self_collision=False` 沿用旧设置 |

枪体 0.001 kg，视觉标记与相机外壳不增加质量、刚体或碰撞。独立 `dummy_target` 帧从旧 0 质量修正为 `1e-6 kg`，惯量对角 `1e-9`，用于避免 PhysX 自动估计零质量未合并帧；这并非给真实枪建了完整质量模型。

默认七轴姿态见 `assets/b2_huibo.py` 的 `HUIBO_INITIAL_JOINT_POS`：

```text
[1.57079226334755, 0.9351052713220801, -1.5751082043382922,
 1.0912415788479128, -0.8986226998003438, -0.650491566,
 -0.650491566]
```

该姿态 FK 的 TCP 约为 `(-0.775,0,-0.48) m`，枪与法兰同轴向下、偏竖直 13.19°。spawn 的 B2 高度为 0.7 m，因此静态预览时枪尖世界高度约 0.22 m。不要用此预览声称已经在 9 cm 工作高度稳定行走。继承机身高度奖励目标实际为 0.52 m，动态可达性和重力响应尚需评估。

## 代码接线

- `assets/b2_huibo.py`：独立新资产配置，保留 B2 腿执行器，仅替换臂和七轴默认姿态。
- `b2_huibo_dual_vision_env_cfg.py`：新 train/play scene，基于 VA16 双相机逻辑，覆盖机械臂奖励的 joint names，禁用原六轴 nullspace 奖励，增加臂部接触终止。
- `agents/rsl_rl_amp_cfg.py`：新增 `AMPB2HuiboDualVisionRunnerCfg`，`resume=False`，19 维统一噪声上下限，独立 `b2_huibo_amp` 日志目录。
- 任务注册：`Unitree-b2huibo-AMP-DualVision`、`Unitree-b2huibo-AMP-DualVision-Play`；旧 Z1 任务继续存在。
- `dual_vision_kinematics.py`、`dual_vision_command.py`、父视觉命令：七轴名称/FK 和相关数组长度泛化，使用真实 manager 维度；视觉 TCP 与编码器增量的来源仍分开，不能将 FK 当相机测量。
- `scripts/rsl_rl/play_amp.py`：动作过滤与日志的机械臂索引按任务配置解析，支持七轴。
- `scripts/vision/play_va16_dual_vision.py`：新增 `--robot huibo`，Huibo 强制指定权重；旧 VA15 70→18 不能兼容新臂。
- `scripts/preview_b2_huibo.py`：无策略、零重力模型 GUI 预览；`scripts/assets/check_huibo_actuators.py`：逐轴实际仿真响应诊断。

实际 Isaac 接口：**actor 73、critic 105、action 19、AMP 43**。AMP 的 43 维为 `12+12+3+3+12+1`；先前交谈中出现的 49 维是错误数字，不应延续。新任务明确按数据集腿序选择关节与足部，避免旧固定索引重排。实际 articulation 中 `right_joint1` 为索引 8，后续臂轴为 13..18，腿不是连续前 12 行。

旧 AMP 参考姿态重置依赖前 12 行，因此 Huibo 关闭 `reference_state_initialization`；仍使用 B2 的 43 维腿部 AMP 判别器和运动数据。不得加载旧 VA7 六轴动作教师。工作命令继续以 `base_link` 表达，不能改用已旋转的 Huibo 底座坐标系。

新工作窗 `x∈[-0.80,-0.75] m`，侧向 ±0.08 m，工具目标离地 0.09 m。课程不再覆盖旧 `[-1.0,-0.55] m` 工作窗，课程速度各阶段均为 0.20 m/s；课程仍会调整路径曲率与转向能力。高吞吐训练用合成路径，图像闭环 Play 才启用实际渲染和独立视觉进程。

## 验证证据：保留成功与失败记录

1. `outputs/b2_huibo_scene_20261102_01/`：首个导入检查，使用旧零质量 dummy 帧。保留，不能当作当前最终资产报告。
2. `outputs/b2_huibo_scene_20261102_02/scene_report.json`：修正 dummy 质量后的当前模型，`pass=true`。已导入 19 动作、73 策略观测；后置相机约 45°；视锥 false；枪体碰撞 true；静态图像 TCP 误差约 0.0020757 m。报告六个地面工作窗投影采样均可见，包含 x 两端与 y=-0.15/0/+0.15。这只是投影可见性，不是这六点均有无碰撞 IK。`viewport_robot.png` 为 GUI 截图。
3. `outputs/b2_huibo_actuators_20261102_01/`：首轮返回码 0，但报告为空、日志只有 4096 字节，不计通过，记录保留。具体原因未定位；不能将后续有效重跑描述为已修复首轮空报告的根因。
4. `outputs/b2_huibo_actuators_20261102_02/report.json`：增加错误路径 traceback 保留、避免异常时快速关闭 Kit，并以 `-u` 无缓冲输出重跑后得到完整报告，七轴运动 `passed=true`。此版 USD 碰撞枚举未进入 instance proxies，碰撞计数不完整；不能据其只枚举到枪体而判断机械臂没有碰撞。
5. **最终执行器报告** `outputs/b2_huibo_actuators_20261102_03/report.json`，退出码 0，七轴全部 `passed=true`。固定基座、零重力、无相机、无策略，每轴通过实际 action term 单独施加 +0.08 rad，再返回。最终实际增量约 `[.080035,.079986,.079407,.079715,.075290,.074099,.073426] rad`，最大目标误差 .006573932 rad，其他关节最大瞬态漂移约 .00593 rad。包含 USD instance proxies 的遍历确认八个 Huibo link（底座及 1..7）的 mesh collider 启用，整机共 48 个启用 collider（含枪）。PhysX 实际 dummy 质量约 `9.99999997e-7 kg`，枪体约 `.001 kg`。验证的是模拟执行器确实运动，不代表带重力/硬件响应合格。力矩字段已区分 PhysX 轴向反力和隐式 PD 估计。
6. 训练烟测：`logs/rsl_rl/b2_huibo_amp/2026-11-02_10-43-27_huibo_integration_smoke/model_2.pt`，64 环境、3 迭代、4608 个环境步。92 个模型张量均有限；84 个标量标签、252 个值均有限。新策略低机身高度终止较多，尚无有效步态，不应作为可部署策略推荐。
7. `outputs/va16_dual_vision/run_20261102_104852_bo477d_i/`：使用烟测 `model_2.pt` 运行 5 步新臂 RHANet + 双相机 + policy trace + export 入口，退出码 0。CSV 实际五行，含 `right_joint1..7` 列，解析出的动作行是 `[8,13,14,15,16,17,18]`；源码变更清单为空。控制状态按 25 步周期写出，五步运行没有状态文件属于正常情况。只证明入口连通，不是跟踪性能验证。

静态场景检查使用 classical 裂缝诊断，并非 RHANet 长时间视觉闭环。旧 Z1 的 20/40 秒毫米级平均误差结果仍属于旧形态，不能改名后归给 Huibo。本次没有进行 Huibo 5 cm 跟踪达标评估。

## 可操作命令

查看当前新臂与相机外壳（GUI 默认保持，零重力，不载策略）：

```bash
cd /home/dong/b2_z1
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/preview_b2_huibo.py
```

正式新策略训练入口（已配置，本轮未启动长训练）：

```bash
cd /home/dong/b2_z1
bash scripts/rsl_rl/train_amp_b2_huibo_dual_vision.sh
```

取得经过筛选的新 Huibo 权重后，运行本地双相机视觉闭环：

```bash
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/play_va16_dual_vision.py \
  --robot huibo --checkpoint /绝对路径/Huibo训练目录/model_N.pt
```

不要把烟测 `model_2.pt` 当可走策略。若需复查逐轴诊断，使用 `scripts/assets/check_huibo_actuators.py --output` 指定一个尚不存在的新目录，保留既有记录。用户操作说明另见 `scripts/vision/HUIBO_INTEGRATION.md`。

## 下一步与尚未完成事项

1. 确认新臂的实际型号、安装支架、质量/惯量、关节硬件额定值与驱动接口。当前 30 Nm、1.5 rad/s、40/2 无硬件依据，不能直接用于板载。
2. 先验证带重力的静态保持、工作高度和整个窄窗可达性；区分碰撞、执行器饱和和策略未学会的问题。当前 self-collision 关闭，不能据外部碰撞测试宣称无自碰。
3. 正式从头训练新的全身 actor，筛选站立/启停、机身高度、足底避缝、后腿抬升、左右弯航向与 TCP 跟踪。旧 Z1 教师不兼容，不通过改维度强行灌旧权重。
4. 新策略稳定后再启用真实 RHANet 双相机闭环；测量 TCP 地面投影到裂缝中心误差、拒帧/陈旧帧行为及脚底与裂缝的最小距离。当前二维投影目标为 5 cm，脚中心距离不能冒充完整足底净空。
5. 硬件路线需实际后置 D435i RGB-D 标定、视觉标记/TCP 标定及 Huibo 驱动接线。当前彩色球标记为仿真可见特征，尚未落实对应实物方案。未获得新的实机状态确认与明确操作授权时，继续只做本地工作。

仓库原先有用户删除的若干旧训练脚本及其他未提交变更，均应保留，不要清理 dirty worktree 或改写历史日志。此前 VA15 实机故障及其交接文档仍保留，不属于这次模型替换已修复的范围。
