# VD3：零枪重与机械臂法兰稳定性训练，2026-11-04

## 追加修订：禁止借枪杆支撑

原长训练已由用户停止，最新检查点为 `model_18000.pt`。真实回放确认旧策略用
仍有碰撞的零质量杆承重；“零枪重”不代表“接触不产生力”。当前代码版本为
`VD3 / support_guard_v1`，继续复用本目录，没有新建项目副本。

本修订增加：

- `contact_forces.history_length=20`，200 Hz 下覆盖 0.1 s。
- 枪杆（合并到 `link06`）向上力超过 **20 N 持续 0.1 s**，或峰值超过
  **150 N**，按失败终止。5 N 内轻触不受新增承重惩罚；更大的力按
  `-2 * max(Fz_peak - 5, 0) / 100` 连续加罚，不设上界，峰值窗口为 0.1 s。
  原有 `undesired_contacts` 的 >1 N 二值项仍保留，因此轻触不是完全无惩罚。
- 底盘总倾角（不含偏航）超过 0.1 rad 后按
  `-3 * (max(tilt - 0.1, 0) / 0.2)^2` 惩罚；失稳终止从 1.0 rad 改为
  0.5 rad，约 28.6°。这是当前平地任务的初始设置。
- 禁止底盘、髋部、大腿、小腿及 `link00`–`link05` 触地；脚正常触地，枪使用
  独立承重阈值。Play 保留同样终止保护，不再隐藏底盘／小腿触地失败。
- TensorBoard 新增 `Metrics/ee_target/tool_upward_force_peak`、
  `tool_upward_force_mean`（N）、`tool_support_active`、`base_inclination_rad`。
  命令指标仍为 reset 时的汇总，连续时间统计用 probe CSV；
  `Episode_Termination/tool_support` 表示框架的终止归因。

默认长训练入口仍是下方的 `train_vd3.sh`，运行名改为 `vd3_no_tool_support`。
**建议从脚本默认的 VD2 model_2050 热启动新训练**，重置优化器和课程，不把已学会
借杆支撑的 18000 轮模型直接原样续训。本次仅执行短诊断和启动检查，没有启动长训练。
当前旧模型在新保护下频繁结束回合是检测生效，不能当成步态已经修好。

诊断报告：
`/home/dong/rl_sar_b2_z1/outputs/VD3_SUPPORT_REVIEW_20261104/review.md`。

VD3 当前源码已整合到统一训练目录 `/home/dong/b2_z1_VD2`，使用独立实验名
`b2_z1_vd3_curve_amp`。目录名沿用旧名，当前代码版本见 `CODE_VERSION.json`。
VD2 未提交源码以小型快照保留，原始资产、检查点与日志继续保留。
后续版本复用同一目录；目录整合及快照说明见 `TRAINING_WORKSPACE.md`。
本版本调整枪体动力学，并针对步态期间多余的机械臂法兰摆动增加训练约束；
保持 actor 70 维观测、18 维动作以及原 PPO＋AMP 网络结构。
枪体资产和新增稳定性项的具体定义以本版本环境配置及其奖励函数为准。

按用户补充，重点是**法兰来回摆动而 TCP 位置变化不大**。当前奖励已从初版
TCP 速度约束调整为 `link06` 法兰位置／速度约束，TCP 位置与姿态跟踪继续保留。
法兰到 TCP 的刚性偏移约 0.827 m，绕 TCP 小幅转动也可能引起较大法兰位移；
这是一种几何上允许的情况，尚未证明就是录制中摆动的唯一来源。

## 这版具体改了什么

1. **枪体质量和惯量贡献为零。**PhysX 对独立的零质量刚体会自动估算或回退，
   因此没有只在旧 URDF 填一个零。专用资产 `b2_z1_vd3_zero_gun.urdf` 将枪的外形和
   碰撞变换到原 `link06`，移除枪、安装固定坐标体及 `dummy_target` 的独立刚体，
   保持 `link06` 本身 0.2887581 kg 及原惯性。TCP 用相同固定变换计算，仍是原来的
   枪尖坐标。名义总质量 78.9990202 kg，不包含任何额外枪体或 TCP 负载。
   原 `b2_z1_gun.urdf` 与 VD2 资产保留。启动质量审计同时检查实际刚体、惯量及
   原有 base 质量随机化；审计不通过时停止训练。
2. **修正动作二阶平滑。**旧式误写成 `a[t] - 2*a[t] + a[t-2]`，会漏罚某些
   两步交替摆动；现在使用 `a[t] - 2*a[t-1] + a[t-2]`，权重仍为 −0.01。
3. **直接抑制法兰的多余摆动。**用同一条 TCP 目标及枪的固定变换，反算法兰
   应有的位置、姿态和速度。位置误差惩罚权重 −0.25，尺度 0.05 m，死区
   0.005 m，压制摆幅及缓慢偏移；线速度误差权重 −0.3，尺度 0.12 m/s，
   死区 0.01 m/s；角速度误差权重 −0.1，尺度 0.5 rad/s，死区 0.05 rad/s。
   三项都采用有上界的归一化平方惩罚，原始惩罚上限各为 4。实际速度取法兰
   `link06` 原点，正确减去底盘平移及旋转速度。法兰目标随 TCP 目标一起移动，
   不把法兰锁在一个位置。原 TCP 线／角速度奖励项不再启用，相关函数保留用于
   对照诊断；刚性连接下法兰与 TCP 角速度相同，避免重复计算角速度惩罚。
4. **补充温和约束。**六个机械臂关节的 `sum(qdot²)` 权重 −0.02，按名称解析，
   不假设它们是动作最后六维；TCP 位置跟踪权重仍为 5，宽度从 0.10 m 收紧到
   0.07 m，姿态误差权重从 −0.2 调到 −0.4。六臂关节重置初速度改为 0，腿部
   AMP 参考状态初始化和原来的 PD、动作缩放、50 Hz 策略／200 Hz 物理控制保留。

目标反算及相对速度计算为（F=法兰，T=TCP，FT=固定安装变换）：

```text
q_F_desired = q_T_desired * inverse(q_FT)
offset_b = R(q_F_desired) * offset_FT
p_F_desired = p_T_desired - offset_b
v_F_desired = v_T_desired - omega_T_desired × offset_b
v_F_relative_b = R_base^T * (v_F_w - v_base_w - omega_base_w × (p_F_w - p_base_w))
velocity_error = v_F_relative_b - v_F_desired
omega_F_relative_b = R_base^T * (omega_F_w - omega_base_w)
```

当前目标姿态固定，因此目标相对角速度为零，法兰目标线速度等于 TCP 的 Bézier
解析速度。位置项以反算出的法兰位置为准，避免与原 TCP 任务提出不同的目标。
这里使用 link 原点速度，避免混用质心速度。
约束持续生效，不按“哪只腿抬起”开关，以免产生奖励跳变或诱导改变步态逃避处罚。
这约束的是**法兰相对机身且超出目标要求的运动**，保留跟随末端目标的能力。
如果法兰在世界坐标中的运动完全来自底盘俯仰，这部分不会被当成额外摆臂处罚；
日志同时记录世界系速度和相对机身速度供区分。不能等同于枪尖在地面坐标系完全
固定。以上权重是本版训练初值，尚无长训练结果证明其最优。

本次没有改变 VD2 的命令采样：行走指令 0.2 m/s，重采样时约 20% 为停车，
周期 3–8 秒。指令不代表实测速度。课程重新从初始范围开始，曲线路径参数
`trajectory_curve_offset_scale` 不等于此前 MuJoCo 调试的 0.30/m 曲率。

## 长训练命令

```bash
bash /home/dong/b2_z1_VD2/scripts/train_vd3.sh
```

此命令会实际开始新的长训练。旧 VD3 目录已于 2026-11-05 归档删除，
历史日志和检查点已迁入本工作区，路径与版本入口见 `TRAINING_WORKSPACE.md`。
迁移没有自动开启新的长训练。
脚本默认使用以下设置：

- Python：`/home/dong/anaconda3/envs/env_isaaclab/bin/python`。
- 任务：`Unitree-b2z1-Velocity`，无窗口，256 个并行环境，20,000 次新 PPO 更新。
- 热启动：VD2 的
  `/home/dong/b2_z1_VD2/logs/rsl_rl/b2_z1_vd2_curve_amp/2026-11-03_10-33-21_vd2_curriculum_resume/model_2050.pt`。
- 加载 actor、critic、AMP 判别器和 AMP 归一化状态；优化器重新建立，迭代从 0 开始，
  课程采用 VD3 配置的初始难度，不继承 VD2 检查点已经升级的课程。
- AMP 预载 200,000 对转换；每 250 次更新及正常退出时保存检查点。
- 本次新运行只保留最近 5 个由当前 runner 成功保存的新检查点。

显式展开的等效命令：

```bash
cd /home/dong/b2_z1_VD2
PYTHONPATH=/home/dong/b2_z1_VD2/source/b2_z1 \
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VD3_ASSET_CACHE=/dev/shm/vd3_train_assets \
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Unitree-b2z1-Velocity --headless --num_envs 256 \
  --experiment_name b2_z1_vd3_curve_amp --run_name vd3_no_tool_support \
  --vd3-warm-start /home/dong/b2_z1_VD2/logs/rsl_rl/b2_z1_vd2_curve_amp/2026-11-03_10-33-21_vd2_curriculum_resume/model_2050.pt \
  --max_iterations 20000 --keep-last-checkpoints 5 \
  agent.amp_cfg.amp_num_preload_transitions=200000 agent.save_interval=250
```

建议使用封装脚本，它还会把临时文件、资产转换和 CUDA／图形缓存定向到
`/dev/shm/vd3_train_cache_<用户ID>/`，减少根分区占用。它明确设置本地
`PYTHONPATH`，避免系统 editable 安装错误导入 `/home/dong/b2_z1` 的旧版本。
缓存可用 `VD3_CACHE_ROOT` 环境变量指定；不要把检查点唯一副本放在 RAM 文件系统。

脚本末尾可追加参数覆盖默认值，例如先进行小规模启动检查：

```bash
bash /home/dong/b2_z1_VD2/scripts/train_vd3.sh \
  --num_envs 16 --max_iterations 2 --run_name vd3_smoke \
  agent.amp_cfg.amp_num_preload_transitions=4096
```

替换热启动检查点可追加 `--vd3-warm-start /绝对路径/model_x.pt`，
也可设置 `VD3_WARM_START`。显存不足时可减小 `--num_envs`。

## 日志、磁盘与恢复

默认日志路径：

```text
/home/dong/b2_z1_VD2/logs/rsl_rl/b2_z1_vd3_curve_amp/<时间>_<run_name>/
```

其中保存 TensorBoard、配置、源码快照与检查点。日志目录必须新建，遇到已有同名目录会停止，
避免覆盖旧运行。可用 `--log-root /持久磁盘/rl_logs` 替换实验目录的父目录；
程序拒绝将日志根目录放在 tmpfs、ramfs 等 RAM 文件系统。

检查点保留策略只登记当前 runner 新建并成功完整保存的 `model_<数字>.pt`，
且只能删除当前运行目录的直接子文件，不扫描其他目录，不删除预先存在的文件，
不跟随符号链接。文件先写临时文件、完成后原子替换，再清理超出数量的旧检查点；
写盘失败不会触发清理。通常 5 个完整检查点约占 60 MB，保存过程中短暂增加一个文件。
不同运行目录的旧记录、VD2 热启动源以及恢复来源都保留。
`--keep-last-checkpoints 0` 关闭清理；在当前小容量根分区上不建议关闭。

从 VD3 检查点继续训练：

```bash
bash /home/dong/b2_z1_VD2/scripts/train_vd3.sh \
  --resume --load_run 2026-11-04_XX-XX-XX_vd3_zero_gun_flange_stability \
  --checkpoint model_5000.pt --max_iterations 20000 --run_name vd3_continue
```

把示例运行名和检查点换成实际存在的文件。若原训练指定了自定义 `--log-root`，恢复时也要指定。
旧 VD3 的运行结果现已迁入本工作区，续训统一使用
`--log-root /home/dong/b2_z1_VD2/logs/rsl_rl`；脚本默认已使用此路径。
脚本遇到 `--resume` 会取消默认热启动参数。恢复加载网络、AMP、优化器、学习率标量、
VD3 课程范围及相对课程计时，并重置环境使初始命令采用恢复后的难度；
不恢复模拟器的物理状态、随机数流与进行中的 rollout。
`--max_iterations` 指本次新增更新次数，沿用现有 runner 的迭代标签行为，
会从保存的标签开始执行，而不是把参数当累计结束轮次。

VD3 恢复必须使用保存了 VD3 实验标识与 `vd3_curriculum_state` 的检查点。
VD1／VD2 检查点只能用热启动进入新动力学，不能伪装成 VD3 原样恢复。
保留 VD2 原有 `vd2_curriculum_state` 和显式 legacy 课程重建接口。

TensorBoard：

```bash
/home/dong/anaconda3/envs/env_isaaclab/bin/tensorboard \
  --logdir /home/dong/b2_z1_VD2/logs/rsl_rl/b2_z1_vd3_curve_amp --port 6006
```

旧 VD3 历史运行与本工作区的新运行均可通过上面的目录监控。

## 已有依据与验证范围

下列 2026-11-04 启动检查在原 `/home/dong/b2_z1_VD3` 目录完成，文中相对
日志路径以当时目录为准。原验证报告保留历史路径，不因源码整合改写。

源 VD2 的最新日志记录到 2064 次更新，最后完整检查点为 model2050。
其末 50 条训练记录平均末端位置误差约 0.01564 m、平均 episode 长度约
1483.94／1500 步。该检查点没有已核实的 MuJoCo 部署验证记录；这些数字
不能作为 VD3 效果、实机精度或新奖励有效性的证明。

原训练使用理想几何前视和仿真真值辅助的后视命令；VD3 仍是带特权信息的
策略训练，不是直接从双相机图像训练 actor。实际 RGBD 链路及部署动力学需另行评估。

CPU 检查覆盖热启动／恢复参数、VD3 与 VD2 课程隔离、优化器与学习率恢复、
日志存储类型、脚本额外参数传递和限定当前运行的检查点清理。

2026-11-04 初版 TCP 速度约束的历史检查（已保留，不代表当前法兰版）：

- 全部 CPU 测试 **113 passed，111 subtests passed**。包括原有命令／课程测试、
  枪体几何与惯量、虚拟 TCP、运动坐标变换、解析目标导数及动作平滑回归。
- 真实 Isaac Sim：**256 环境、AMP 预载 200,000、2 次 PPO 更新**完成，正常保存
  `model_0.pt` 和 `model_1.pt`。运行目录为
  `logs/rsl_rl/b2_z1_vd3_curve_amp/2026-11-04_09-25-22_vd3_smoke_20261104/`。
  其中 `vd3_runtime_asset.json` 确认 24 个实际刚体、18 个可动关节，名义总质量
  78.999015808 kg（float32），`link06` 质量 0.288758099 kg、原惯量保留，
  没有独立枪或 TCP 刚体。实时总质量约 78.021–81.985 kg，差额仅来自原有
  base 的 [−1, 3] kg 随机化，并非枪质量回退。
- 从上述 `model_1.pt` 恢复，**16 环境、再做 1 次 PPO 更新**通过。目录为
  `logs/rsl_rl/b2_z1_vd3_curve_amp/2026-11-04_09-27-12_vd3_resume_smoke_20261104/`；
  优化器与 VD3 课程加载成功，课程计时从 48 增加到 72 步。以磁盘上的实际目录为准。
- 检查点全部权重有限，actor 仍是 70→18；初始课程为 curve_level=0，EE 为
  X=[−1.03,−1.01]、Y=[−0.02,0.02]、Z=[−0.31,−0.29] m。
  初次运行的源码快照与结束时源码一致，VD2 基线文件哈希均未改变。

2026-11-04 当前法兰版检查：

- 全部 CPU 测试 **128 passed，111 subtests passed**。新增回归确认：TCP 位置和
  线速度为零时，绕 TCP 转动产生的法兰摆幅／速度仍受罚；绕枪杆轴转动由角速度项
  捕获；全机刚体运动及正常目标平移不产生伪惩罚。还检查了目标变换、重采样、部分
  重置和当前配置只启用法兰三项。
- 真实 Isaac Sim **256 环境、AMP 预载 200,000、2 次 PPO 更新**完成，三项法兰
  奖励和诊断指标正常输出，权重、优化器与课程保存成功。目录为
  `logs/rsl_rl/b2_z1_vd3_curve_amp/2026-11-04_09-41-33_vd3_flange_smoke_20261104/`，
  `console.log` 保存输出；实际零枪重质量审计通过。当前脚本默认运行名为
  `vd3_zero_gun_flange_stability`，与初版日志分开。

这些只是训练入口、动力学和保存／恢复的启动检查，**不是稳定行走或摆臂改善验收**。
刚更换动力学的短运行中仍有较大 TCP 速度误差与环境提前重置；应通过长训练适应，
之后同时检查实际行走速度、TCP 跟踪误差与摆动幅度，避免把“走得更慢”误判成
稳定性提升。上述启动检查阶段没有启动 20,000 次长训练；此后用户已自行启动，
本次目录整合保留了该进程及其原运行目录。

TensorBoard 重点查看 `Metrics/ee_target/flange_position_error`（m）、
`flange_linear_velocity_error`（m/s）、`flange_angular_velocity_error`（rad/s）、
`flange_relative_linear_speed` 和 `flange_world_linear_speed`（m/s）。
TCP 线／角速度误差及相对线速度仍保留作对照，`arm_joint_velocity_rms`
为六关节之间的 RMS（rad/s）；结合
`position_error`、底盘速度跟踪奖励与 episode 长度观察。CommandTerm 默认在回合
重置时汇总这些指标的当前值，它们不是整回合的时间 RMS；需要精确摆幅或时间 RMS
时应在评估阶段连续记录。对照时使用相同目标与速度条件，并确认机器人确实在走。
