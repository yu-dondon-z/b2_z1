# VA16：固定速度与前后双相机基线

2026-11-02 后续修改：相机视野线框默认关闭，D435i 外壳模型仍显示。
新增七轴 B2＋Huibo 模型与训练入口，见
[`HUIBO_INTEGRATION.md`](HUIBO_INTEGRATION.md)。下文 VA15/VA16 的
70 维观测、18 动作及精度记录均属于原 Z1，不能套用到 Huibo。

本分支回到 `/home/dong/b2_z1` 的 Isaac 训练工程，保留原 VA15 入口。
新任务为 `Unitree-b2z1-AMP-VA16-DualVision-Play`，第一阶段复用 VA15
`model_5000.pt` 的 70 维观测、18 维动作，验证新的视觉外环。
默认播放仍是未做正式 VA16 训练的 VA15 视觉外环基线；另有10轮代理续训 smoke
权重用于验证训练链。下面的 20/40 秒精度记录对应原 VA15 加上一版侧置后相机，
不代表当前中央安装、smoke 权重或实机验收。

## 运行

```bash
cd /home/dong/b2_z1
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/play_va16_dual_vision.py
```

默认启动 Isaac 窗口和三联视觉预览：前视 RGB/裂缝中心线、后视
D435i RGB/实测枪尖、后视对齐深度。Ctrl+C 只关闭本次创建的子进程。
不连接机器人、不启动 DDS、不同步板载程序。

## 当前后相机安装与显示

后置 D435i 现固定在机身后部中间，光心相对 `base_link` 为
`(-0.445, 0, 0.005) m`，朝后（yaw 180°）、向下俯视 45°。
安装参考 URDF 自带 `r_dc_link`，从尾壳稍向外伸出，沿外壳斜向下；
枪体位置和标记安装位置未随相机调整。之前的 `(-0.48, 0.30, 0.34) m`、
yaw 225°/俯视 55° 是上一版偏左高位方案，仅保留作历史对照。

仿真中显示银色外壳、黑色前面板与三个镜头的 D435i 示意模型，
没有新增刚体质量或物理碰撞。按最新要求，视锥线框默认关闭。
可选显示功能仍只作用于主视口，不进入 RGB-D 图像；其近端 0.10 m、
远端 1.50 m 只是显示范围。实际深度裁剪仍为 0.10–6.00 m。
主视口采用机身后方斜视并跟随机器人，
可同时检查安装位置、朝向和地面覆盖范围。

中央安装的首次静态记录为
`outputs/va16_rear_center45_20261102_01/scene_report.json`：
此姿态后视感知有效，RGB-D TCP 对独立真值误差约 2.38 mm；
但初始机身高度约 0.70 m 时，原完整工作域的近端 `x=-0.55 m`
投影超出画面下边界，因此旧全工作域检查仍为 `pass=false`。
这一完整旧工作域限制仍保留；当前安装的动态和显示检查另见下文，
不能用单个可见 TCP 的静态测量代替全工作域验收。

已实际打开 GUI 检查相机模型与视锥，屏幕截图为
`outputs/va16_rear_center45_20261102_03/viewport_camera_frustum.png`。
该次报告确认俯角 44.99999°、相机壳体无刚体/质量/有效碰撞 API、
当前可见 TCP 的测量误差 2.3765 mm；`pass=true` 的范围仅为
`rear-centre 45-degree mount and current visible TCP`，
`full_workspace_visible=false` 仍明确保留。
03/04 检查器虽生成报告和截图，但 `fast_shutdown=False` 的完整插件卸载阶段
返回 139，显式清理 overlay 后仍可复现。最终检查器改用与默认播放相同的标准
Kit shutdown 模式；05 已保存报告和截图并正常退出 0，没有覆盖异常或退出码。
最终记录为 `outputs/va16_rear_center45_20261102_05/scene_report.json`，
截图为同目录 `viewport_camera_frustum.png`；安装/当前 TCP 检查通过，
原完整工作域可见性仍为 false。未声称定位或修复底层插件完整卸载故障。

有界检查：

```bash
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/play_va16_dual_vision.py \
  --headless --max-steps 500
```

`--dry-run` 只生成本次运行目录并打印启动参数。
默认视觉模式 `rhanet` 加载既有 RHANet_dep 权重。显式
`--vision-mode classical` 仅用于相机场景、几何和标记可见性诊断，
其分数是像素对比度，不代表神经网络置信度或已验证影子鲁棒性。

## 固定速度代理阶段续训

训练入口已准备好，并完成了一次 128 环境、10 轮的有界 smoke：
共 30,720 个采样步，新日志为
`logs/rsl_rl/b2_z1_amp/2026-11-02_09-51-54_va16_fixed020_smoke/`，
保存 `model_5009.pt`。这证明续训初始化和更新链能够执行，不代表新策略效果已经验收。

```bash
cd /home/dong/b2_z1
bash scripts/rsl_rl/train_amp_va16_dual_vision.sh
```

此入口使用 `Unitree-b2z1-AMP-VA16-DualVision`，默认 1024 个环境，
从 VA15 的 `2026-09-28_15-07-31_va15_curriculum_from2500/model_5000.pt`
继续训练 2000 轮；保留 actor/critic 和 AMP 判别器/归一化状态，显式
`--reset_optimizer` 重新建立优化器，输出到新的 `*_va16_fixed020_proxy`
日志目录。独立 VA7 Z1 教师只参与训练损失，不替换学生初始化。
可在末尾加 `--num_envs 256 --max_iterations 100` 覆盖环境数和追加轮数。

这是无 RTX 相机的高并行代理阶段，训练固定 `0.20 m/s` 下的步态、
转向与机械臂目标跟踪。其曲线来自特权几何命令，地面沿用平路；
它不训练 RHANet，也不能替代后续真实渲染双视觉验收和续训。
VA16 关闭独立随机 standing mask，起停统一由轨迹命令管理，避免腿部
随机置零而末端仍按运动状态推进。原 VA15 的 standing 样本设置保留。

可选 `Unitree-b2z1-AMP-VA16-DualVision-Visual-Train` 仅支持一个渲染环境和
外部视觉桥接；当前不提供已验收的一键图像闭环训练入口。
足底避线奖励已改接物理裂缝 mesh 的最近线段距离，避免 file 模式
旧 synthetic 距离返回无穷大而令惩罚失效。`dual_vision_rewards.py`
专门用于该奖励/独立评估；物理真值不得进入视觉数据包或控制目标。

## 数据与控制关系

- 工作段速度指令为 `vx=0.20 m/s, vy=0`；前视 RGB 裂缝几何生成角速度 `w`。
  起步、观测尚未就绪和结束不作为恒速工作段。
- 前相机只导出 RGB、内外参和拍摄时间；没有使用前部深度。
- 后部 D435i 导出 RGB 与对齐深度。地面由后视深度 RANSAC 拟合，
  前视将裂缝像素射线与这个同帧实测平面求交。当前只适用于局部近似平面地面。
- 后相机通过枪体青色/品红色球形标记的 RGB 像素与深度表面拟合球心，
  使用已知标记尺寸、间距和到 TCP 的安装偏移恢复实际枪尖；没有将关节正运动学、
  USD 中枪尖位置或仿真地面高度作为视觉测量。
- 两次视觉测量之间，使用关节编码器正运动学的位移增量传播枪尖位置，
  补偿识别延迟期间的机械臂运动；绝对位置仍由 RGB-D 测量锚定。
  仿真真值 TCP 仅用于独立日志比较，不进入这项传播。
- 后视同时识别焊缝中心线，将 TCP 投影到实测地面平面，输出到线段中心的误差。
  当前标记是仿真感知方案；实机需要实体标记和独立安装偏移标定。
- 感知帧按拍摄时的基座坐标表达；基座拍摄姿态只用于时序补偿。
  任何真值 TCP 或焊缝位置仅可用于独立评估。
- 策略为 50 Hz，相机导出设置为每两个策略步一次、目标 25 Hz。
  实际可处理频率由 Isaac 渲染和视觉耗时决定，不能把配置频率当作实测频率。

## 文件协议和日志

每次运行建立独立 `outputs/va16_dual_vision/run_*` 目录，保留：

| 文件 | 内容 |
|---|---|
| `session.json` | 任务、完整启动参数、策略 SHA256、基线阶段说明 |
| `play.log` / `vision.log` | 本次仿真与视觉日志 |
| `front/latest_frame.json` | 前 RGB 元数据、序号与原始拍摄时间 |
| `rear/latest_frame.json` | 后 RGB-D 元数据、内外参与原始拍摄时间 |
| `latest_dual_trajectory.json` | 最新联合测量，包含有效性和拒绝原因 |
| `dual_vision_packets.jsonl` | 每次实际处理的联合测量历史 |
| `latest_dual_overlay.png` | 前 RGB、后 RGB/枪尖、后深度的三联预览 |
| `motion_trace.csv` | 策略执行和真值 TCP/目标日志，用于独立评估 |
| `run_report.json` | 运行后执行评估脚本生成的跟踪、时效、运动诊断 |
| `policy_export/` | 本次 JIT/ONNX 导出，不覆盖原 checkpoint/exported |

元数据中 `capture_timestamp` 为同主机 `time.monotonic()` 秒；
`capture_generation` 隔离环境重置；`sequence` 标识真实导出的采集。
桥接只处理同 generation、同 sequence 的相机对，读图后再次核对元数据，
不将旧图文件的 mtime、发布时刻或推理结束时刻改写为拍摄时刻。

联合包 schema 为 `b2_z1_dual_vision_v1`：顶层 `timestamp` 是 wall-clock 发布秒，
`front/rear` 各自保留原始 `capture_timestamp`、`sequence`、`capture_generation`、
`valid/reason` 和 `points_b`；rear 另含 `ground_plane_b`、`tcp_measured_b`、
`tcp_error_b`、`tcp_error_m`、标记深度拟合诊断。
感知失败仍发布明确无效包，不向包中补入真值、不伪造置信度。

## 独立评估一次运行

```bash
/home/dong/anaconda3/envs/crack_vision/bin/python scripts/vision/evaluate_va16_dual_run.py \
  outputs/va16_dual_vision/run_具体目录
```

脚本读取已保留的日志，不启动仿真。TCP 精度按实际进入跟踪的样本计算，
默认最大误差标准为 5 cm；无跟踪、缺失独立误差或最大误差超限均报告
`pass=false`，没有样本时输出 JSON `null`，不会用 NaN 或准备期数据冒充跟踪结果。
全局最大误差、机身倾斜及重置仍另行保留。

新 VA16 trace 以 50 Hz 直接记录 `visual_target_active`、重置 generation、
实际基座速度、四脚世界位置和接触力。脚体与接触传感器分别按
`FL_foot/FR_foot/RL_foot/RR_foot` 名称核对映射，不假设两者索引顺序一致。
评估优先使用这个逐步 active 标记，并利用 status 记录的真实 mesh 参数独立
计算 TCP 投影误差和支撑脚中心到裂缝的距离。旧 trace 没有 active 列时，
明确降级为低频 status 对齐并报告时间不确定性。

足部报告包含接触比例、抬脚次数及摆动脚中心高度。脚中心距线不等于
足底边缘净距或碰撞检测；7 cm 只标记为现有训练奖励的参考距离，不作为新增
验收门槛。视觉实际处理频率来自原始 capture 时间和已处理序号，不使用配置的
25 Hz 冒充实测吞吐。`pass` 仅代表本次记录样本的基线条件，不是实机或完整场景验收。

## 本阶段验证与未完成事项

已加入场景/命令接口、独立启动入口、真实权重视觉桥接、配对帧龄检查、
三联图像预览及日志。感知几何和文件协议有独立 CPU 回归测试：

```bash
/home/dong/anaconda3/envs/crack_vision/bin/python scripts/vision/test_dual_camera_perception.py
/home/dong/anaconda3/envs/crack_vision/bin/python scripts/vision/test_dual_camera_bridge.py
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/test_dual_vision_rewards.py
/home/dong/anaconda3/envs/crack_vision/bin/python scripts/vision/test_evaluate_va16_dual_run.py
```

文件协议测试覆盖前相机不需要深度、同帧后视实测平面供前视投影、
4 cm 已知 TCP 偏差重建以及旧帧不更新时间。测试图像为确定性合成图；
它们不代表实际 RHANet 在仿真渲染图像上的分割精度。

当前中央 45° 安装已完成独立新验证：

| 运行目录（均在 `outputs/va16_dual_vision/`） | 本轮结果 |
|---|---|
| `run_20261102_101653_bp0ynucm` | 20 秒真实 RHANet/VA15 闭环正常退出；945/1000 个 50 Hz 样本跟踪激活，独立 TCP 投影误差平均 7.99 mm、P95 17.83 mm、最大 30.65 mm，`pass=true` |
| `run_20261102_101959_nqq0vksk` | 默认 GUI launcher 有界 150 步正常退出；相机模型、主视口视锥和三联视觉预览正常，启动/结束源码哈希无变更；本次为显示与退出链检查，不作长程精度验收 |

新中央安装的 20 秒结果满足本次平地场景的 5 cm 条件；尚未证明完整工作域、
长期遮挡、所有曲线/光照或实机条件。它仍复用原 VA15 策略，并非新策略训练成绩。

历史上，2026-11-02 还完成以下真实渲染、真实 RHANet 和 VA15 策略的本地运行。
以下四次均使用上一版偏左高位后相机 `(-0.48, 0.30, 0.34) m`，
不能将这些误差结果迁移为当前中央 45° 安装的验收成绩：

| 运行目录（均在 `outputs/va16_dual_vision/`） | 独立结果 |
|---|---|
| `run_20261102_094258_6jcnctvq` | 无实际跟踪接入，`pass=false`；拒绝原因为前视路径缺少有效前进切线 |
| `run_20261102_094557__rookw_h` | 后视中心线误选枪体；实际跟踪段 TCP 误差均值15.37 cm、P95 39.55 cm、最大41.56 cm，`pass=false` |
| `run_20261102_095233_07sq83iw` | 修复地面深度 ROI 后20秒运行：940/1000个50 Hz样本跟踪激活；实际跟踪段误差均值8.99 mm、P95 21.72 mm、最大31.14 mm，`pass=true`，无重置 |
| `run_20261102_095512_iobalqyn` | 同逻辑延长40秒：1912/2000个50 Hz样本跟踪激活；实际跟踪段误差均值9.57 mm、P95 23.51 mm、最大40.22 mm，`pass=true`，无重置 |

run2 前视/后视 valid 比例为100%/92.5%，但枪体被误选为线，说明神经网络
置信度和几何有效性通过不能替代独立误差验收。原始失败日志与图像完整保留，
没有重写为修复后的结果。run3 活跃支撑脚中心距线最小为
FL26.16 cm、FR10.92 cm、RL22.42 cm、RR18.02 cm，后脚记录的抬脚事件为
RL31次、RR32次；后脚摆动中心高度最高5.71/6.28 cm，相对约3.2 cm的站立脚中心高度，
抬腿幅度仍偏低。5 N 接触阈值可能产生短时反复，需要逐周期检查，不能把事件数等同于完整步数。

run4 实际处理双相机对约4.41对/墙钟秒，图像导出序号推得约5.51帧/墙钟秒；
处理耗时均值142.85 ms、P95 201.13 ms，发布时帧龄均值179.64 ms、
P95 246.04 ms。控制消费时活跃样本的帧龄P95为430.76 ms。
仿真渲染慢于实时，所以这组墙钟频率不能代替板载性能测试。前/后有效测量比例为
100%/99.625%；这个比例仍只是筛选通过率。

run4 稳定速度命令段（active且 `command_vx>=0.198 m/s`，1217个样本）
实际 `vx` 均值0.1887 m/s、P05–P95为0.1362–0.2458 m/s；
恒定的是0.20 m/s工作速度指令，实际速度仍有步态波动和小幅欠速。
活跃支撑脚中心距线最小为FL7.17 cm、FR9.87 cm、RL11.26 cm、RR17.35 cm，
比20秒短程出现更小余量；后脚抬脚事件RL69次、RR73次，摆动脚中心最高约6.30/6.34 cm。
相对约3.2 cm的站立脚中心高度，后腿抬脚仍约3 cm量级，需要后续专项改进。
活跃横滚最大1.96°、总倾角最大2.54°。这些记录没有测量足底边缘净距或完整碰撞形状，
尤其FL中心仅7.17 cm，不能据此直接宣称全程足底不会踩线。

run3 在启动、结束均记录29个相关源文件 SHA256，标记 launcher 和 evaluator
在运行期间发生变更；控制和感知文件哈希一致。run4 列出的源文件启动/结束哈希均一致，
结束后再将编码器运动补偿 `dual_vision_kinematics.py` 加入清单，后续清单共30项。
run1/2 启动于哈希记录加入前，不为旧运行补写不存在的启动哈希。

仍需完成更长曲线、照明变化、相机/枪尖全工作域遮挡和起停恢复检查。
之后推进双视觉策略续训和噪声/延迟课程；新增观测的正式策略需要另外迁移、训练和导出，
不能称原 VA15 权重已经学会新版视觉任务。

实机部署仍需 D435i RGB/深度对齐与内外参标定、标记到 TCP 的测量、
真实执行器响应与日志回放验证。本次没有改动 MuJoCo 或实机部署程序。

深度地面 ROI 排除高于地面的枪体和腿部，但不能排除与地面共面的影子，
影子鲁棒性仍依赖 RHANet 和后续实拍训练。两色球需要真实安装件及其尺寸、间距、
到 TCP 偏移标定。前视把后视拟合的单个平面延伸到前方约3 m，仅在本次局部平地
条件验证，不覆盖台阶、坡折或前后地面不共面的情况。原始感知按地面法向投影计算的
误差，与正式独立验收的重力方向/世界 XY 投影在平地近似一致；斜地需要统一定义后重新验收。
