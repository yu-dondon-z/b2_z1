> 历史归档：内容和命令对应原记录日期，不是当前VD13操作指令。当前状态见[文档入口](../../README.md)。

# VD10 枪体接地：修复与验证

本轮只处理新枪的可用启动和进入焊缝工况。训练工作区为 `/home/dong/b2_z1_VD2`。
2.93 kg、外径 56 mm、长 1100 mm 的资产不变；不改驱动增益、奖励权重、接触
传感器或终止阈值，不导出或替换部署 actor，不启动板载控制。
本文的启动结果不代表机械臂左右抖动、步态砸地或奖励收敛已经达标。

## 发现与修改

1. **静态起点不能代表负载后的最低枪高。** 原 0.45 m 枪尖起点仍在约
   0.40 s 接地；零动作、8 环境、150 个 50 Hz 步，按本步实际 dones 统计有
   56 次 `tool_support`。同一环境记录到机械臂关节偏移与机身约 8 cm 下沉。
   新 VD10 初始六关节角为
   `[0, 0.66647356, -1.25178768, 0.88531411, 0, 0] rad`，基身高 0.50 m
   时 TCP 约为世界高度 0.85 m、机身后方 0.95 m；枪轴初始俯角 0.30 rad。
   这同时修改 `use_default_offset=True` 动作的标称角，因此旧 VD10 冒烟检查点
   不能作为同版本断点继续使用。
2. **复位命令须采样最终物理状态。** Isaac Lab 的父复位在命令重置之后才把
   episode 计时归零；站立复位也可能在运动学 forward 前捕获旧 TCP。
   新增仅 VD10 使用的环境子类，在父复位完成后统一 forward、重采命令。
   站立、AMP、显式和自动复位均从最终实测 TCP 开始，原 VD5/VD9 注册入口保留。
3. **新管口中心不应作为接地表面点。** 最终枪轴俯角仍为 `pi/2−0.45`。
   若端面轴中心直接落在地面，外缘会低于地面约 12.179 mm。现在按目标世界
   枪轴计算有限圆柱的最低点，把中心抬到管口外缘具有 **5 mm 几何间隙** 的高度；
   水平机身时中心高度约 17.179 mm。任意轴向使用
   `max(0, L*axis_z) + r*sqrt(1−axis_z²) + 0.005`。
   这明确改变了管口中心的 Z 目标，保留地面焊缝 XY，既不缩小碰撞体也不改接触。
   5 mm 是几何间隙，实际接触力仍由仿真测量。
4. **新枪采用 0.80 m 最终后向工作距离。** 原 0.70 m 配合 1.1 m 管体，
   部分基身姿态下与尾部存在解析几何干涉；改用 0.80 m 后核查到达路径和间隙。
   不使用此前异常的 `mj_geomDistance` 长圆柱负距离作为自碰证据。
5. **高位起点只是课程起点。** 位置和俯角一起随原地面课程过渡到完整目标，
   20,000 个 50 Hz vector steps，约 834 个 PPO 更新；每回合仍保留 2 s 启动过渡。
   管口高度补偿也乘启动和地面课程系数，零时刻严格等于实测 TCP。
   Play 直接使用完整地面工况，不再先播放 400 s 的高位训练课程。
6. **诊断修正。** `get_term()` 会保留旧回合标志，只在本步 `dones=True` 时
   累加归因；同时发生多个终止时仅记录 Isaac Lab 最后归因，不冒充全部原因。
   枪高由真实有限圆柱支持函数计算，不再对任意倾角一律减 28 mm。
   200 Hz 轨迹新增逐环境高度、力和 episode 时钟，避免拼接不同环境的极值。

## 已完成的真实物理启动检查

全程使用 200 Hz PhysX、50 Hz 控制，保留全部真实接触、机身/手臂触地、倾倒及
枪体受力终止；没有 actor、没有优化器更新。各运行 250 actor steps，即 5 s。
地面课程仍处于高位起始阶段，这不是完整地面焊缝验证。

| 复位条件 | 环境数 | 枪体受力终止 | 最大枪体向上力 | 全环境最低圆柱表面高度 | 其余终止 |
| --- | ---: | ---: | ---: | ---: | --- |
| 全部站立 | 32 | 0 | 0 N | 67.05 mm | base_contact 99 次 |
| 原 80% AMP / 20% 站立混合 | 64 | 0 | 0 N | 7.654 mm | base_contact 293，bad_orientation 23 次 |

零动作下机身会失稳，不能因枪体未接地就称整机稳定。实际 actor 的连续训练和
完整地面回放结果见下文。

原始证据位于实验目录 `logs/rsl_rl/b2_z1_vd10_hollow_gun_amp/`：

- `2026-11-13_10-00-16_vd10_safe_nominal_probe_20261113/`：0.45 m 起点失败。
- `2026-11-13_10-43-47_vd10_high_nominal_probe_20261113/`：新起点站立复位。
- `2026-11-13_10-45-53_vd10_high_mixed_probe_20261113/`：新起点混合复位。

各 probe 的 `vd10_zero_action_probe.json` 记录本步终止归因和环境 ID；
`vd10_200hz_trace.json` 记录真实逐环境高度与接触力；源代码、物理资产审计与
实际运行配置同目录保留。接触力不是关节驱动力矩；这里没有把 PD 估算当成
实际驱动力矩，也没有据此宣布驱动差异是根因。

## 连续训练与回归

固定上述配置，从 VD5 Startup `model_4498.pt` 加载 policy/AMP，重置优化器和
新枪课程；没有沿用旧枪的优化器或地面课程。运行目录：

`logs/rsl_rl/b2_z1_vd10_hollow_gun_amp/2026-11-13_10-47-49_vd10_safe_start_continuous_1000_20261113/`

本段 **1000 轮已正常完成**，最终 `model_999.pt`，覆盖从高位到完整地面工况；
没有在中途调整奖励或增益。保存配置为 `learning_rate=5e-5, schedule=fixed`。
第 833 号更新的 rollout 末课程达到 1；从 834 号更新起完整地面训练，只有
166 个完整更新，不能以这段长度认定奖励已经收敛。

TensorBoard 窗口均值如下，完整数据与统计口径见
[`training_scalars.json`](../../../outputs/vd10_startup_20261113/training_scalars.json)。

| 更新编号窗口 | 平均回合长度（50 Hz 步） | 总奖励 | TCP 位置误差（m） |
| --- | ---: | ---: | ---: |
| 0–99 | 422.17 | 25.233 | 0.12547 |
| 500–599 | 1268.25 | 188.637 | 0.05917 |
| 834–999（完整地面） | 1093.24 | 120.549 | 0.08181 |
| 950–999 | 1154.67 | 122.888 | 0.08043 |

课程接近地面时误差和接触指标仍在变化。TensorBoard 的 contact/termination
值来自回合复位时的日志快照，可能在没有新复位时重复；不把它们解释成精确的
枪体终止率。下面的独立回放才按实际 `dones` 逐事件计数。

CPU 回归 `tests/test_vd10_startup.py` 联合原 VD5 启动/命令和新枪资产测试：
**27 passed, 7 subtests passed**。覆盖真实继承链的启动连续性、旋转机身下的
管口间隙、重复更新不累加、异步/重复复位时序、新 profile 恢复与旧 profile 拒绝、
Play 完整课程。几何扫描与真实物理结果分别报告。

离线几何复现为
[`check_vd10_startup_geometry.py`](../../../scripts/diagnostics/check_vd10_startup_geometry.py)，
结果见 [`geometry.json`](../../../outputs/vd10_startup_20261113/geometry.json)。检查水平
路径 101 点、4769 个 AMP 帧各 21 点；新工况最小管口几何间隙 5 mm、尾部
包络间隙 61.16 mm、后腿包络间隙 109.31 mm，关节硬限最小裕量 0.4886 rad。
这是所采样参考姿态/路径的静态可行性，不代表实际运动中始终维持这些间隙。

## 最终 actor 的完整地面独立回放

使用 `--resume` 成功恢复 `model_999.pt` 的 optimizer、AMP、迭代及新版启动
profile/课程，再冻结 `ground_blend=1`。只做确定性 actor 推理，没有优化器更新；
保留原 80% AMP / 20% 站立混合复位、全部真实接触和终止条件。
64 个环境各运行 1500 个控制步，即 30 s，采样 6000 个 200 Hz 物理步。

| 首个回合的最终归因 | 环境数 |
| --- | ---: |
| 跑满 30 s（time_out） | 57 |
| base_contact | 3 |
| bad_orientation | 3 |
| tool_support | 1 |

57/64（89.06%）只是回合跑满比例，不是焊接任务成功率。枪体终止发生于环境
62 的 20.60 s；同一环境在 20.585 s 测得向上力 230.98 N、圆柱最低点高度
0.467 mm，随后由真实受力条件终止。启动前 2 s 的枪体受力终止为 **0**。
整个 30 s 窗口含自动复位后的回合，共 65 次复位，其中 base_contact 4 次、
bad_orientation 3 次、tool_support 1 次、time_out 57 次。

全环境/物理步中向上力超过 1 N 的样本占 0.5714%，超过 20 N 的占 0.00521%；
这两个比例是采样点占比，不是独立碰撞事件率。2 s 启动后枪体最低表面高度的
中位数为 25.82 mm，最小值为 −0.0619 mm；几何离地不能代替真实接触力。

原始运行：
`logs/rsl_rl/b2_z1_vd10_hollow_gun_amp/2026-11-13_11-13-58_vd10_full_ground_policy_probe_20261113/`。
结果汇总见
[`full_ground_policy_probe.json`](../../../outputs/vd10_startup_20261113/full_ground_policy_probe.json)，
由 [`analyze_vd10_ground_probe.py`](../../../scripts/diagnostics/analyze_vd10_ground_probe.py)
从 `vd10_policy_probe.json` 和逐环境 `vd10_200hz_trace.json` 计算，可重复运行。

结论：启动即反复因枪体接地而终止的主要阻碍已解除，可以继续同配置长训。
仍有枪体受力、机身触地和倾倒失败；没有证明奖励收敛，也没有完成新枪的
MuJoCo 枪尖左右振荡或 B2 落脚冲击验证，不作为部署候选发布。

## 继续长训、TensorBoard 与中止恢复

从本次已验证的 `model_999.pt` **追加 5000 轮**，不重新做旧枪 warm-start：

```bash
cd /home/dong/b2_z1_VD2
bash scripts/train_vd10.sh --resume \
  --load_run '^2026-11-13_10-47-49_vd10_safe_start_continuous_1000_20261113$' \
  --checkpoint '^model_999\.pt$' \
  --num_envs 256 --max_iterations 5000 \
  --run_name vd10_safe_start_continue5000_20261113 \
  agent.algorithm.learning_rate=5e-5 agent.algorithm.schedule=fixed
```

最后一行显式固定本次实际运行的 `5e-5/fixed`，避免后续默认值变化造成漂移。
当前 VD10 类默认值也为 `5e-5/fixed`；断点会恢复 optimizer，但不会自动替代
命令行的 schedule 配置。
`--resume` 会恢复 policy、critic、AMP、optimizer、迭代和地面课程/profile，
新建日志目录，不覆盖本次 1000 轮的日志或检查点。`--max_iterations` 是本次
追加次数；当前 runner 重用加载编号一次，因此正常结束文件名为
`model_5998.pt`，以实际更新数和新运行目录核对。

另开终端：

```bash
/home/dong/anaconda3/envs/env_isaaclab/bin/tensorboard \
  --logdir /home/dong/b2_z1_VD2/logs/rsl_rl/b2_z1_vd10_hollow_gun_amp \
  --host 127.0.0.1 --port 6006
```

本机浏览器打开 `http://127.0.0.1:6006`。观察回报、回合长度、TCP 位置/轴向
误差及枪体接触曲线，完整地面配置下留足持续训练时间；一次短窗口的改善不算
收敛。训练后再独立测左右高频抖动和落脚冲击，不只看总回报。

可用 Ctrl+C 中止；脚本默认每 100 轮和正常结束保存，中止时**不保证额外保存**，
最多损失距上次完整保存的进度。以下命令选择刚才续训系列的最新运行及完整
`model_<数字>.pt`，排除临时 `.partial` 文件，再追加所需轮数：

```bash
cd /home/dong/b2_z1_VD2
bash scripts/train_vd10.sh --resume \
  --load_run '.*_vd10_safe_start_continue5000_20261113.*$' \
  --checkpoint '^model_[0-9]+\.pt$' \
  --num_envs 256 --max_iterations 5000 \
  --run_name vd10_safe_start_continue5000_20261113_resume \
  agent.algorithm.learning_rate=5e-5 agent.algorithm.schedule=fixed
```

恢复时的 5000 仍是新增轮数，可改为剩余计划更新数。不要选择 probe 目录或旧版
VD10 接地失败检查点；本次只给长训命令，没有后台启动这段 5000 轮训练。
