# VB12：阶段推进与部署安全门槛分离

## 原因和范围

VB11 末期的奖励仍然较高，但 `startup_not_ready` 终止比例接近 1，
实际有命令的运动比例降为 0。静止的速度奖励并不能证明完成了轨迹任务。
本版修复任务被准备门槛永久阻塞的问题，并修复视觉等待历史路径覆盖期间
TCP 的 Z 目标跟随机身上下移动的问题。碰撞、质量、软管反力模型保持不变。

这是训练代码分支 `VB12`，不是已验收的部署策略。
没有修改 `/home/dong/rl_sar_b2_z1`，没有操作实机。

## 训练改动

- 从筛查过的 VB10 `model_12300.pt` 续训，重置优化器；不使用退化的 VB11 14299。
- 85% 运动任务、15% 静止任务。运动任务独立按时间进入平滑起步：
  一半环境准备 1～2 秒，一半准备 4～6 秒，不再由策略的表现决定是否给速度。
  这两类时间与轻重负载、初始姿态独立采样；“短准备”不保证初始姿态已准备好。
- 保留严格准备状态作为指标：成功一次 +3；准备时间结束后再过 3 秒仍未成功，
  一次 -5；非超时的物理失败 -10。实际奖励系数包含 dt=0.02 的换算。
- 增加连续 TCP 误差惩罚；准备失败不再在训练中触发重复的零命令重置。
  身体碰撞、低高度、倾倒、枪持续接触和接触预算等物理终止不取消。
- 清晰初始臂姿和旧版冷起步各占 50%；重置时动作 EMA 和历史仍按实测关节一致初始化。
- 仅训练时：Z1 6 个动作的探索标准差初始化为 0.08、上限 0.10；
  姿态引导权重 0.12、本体参考权重 0.10。仍是 70 维观测 → 18 维动作的全身策略，
  训练辅助教师不参与播放、导出或部署；没有固定机械臂或改 PD 增益。
- 保留 50% 标称枪质量 + 3kg 软管、10% 轻载、40% 原有随机负载域。

## 视觉验收

`Schedule-Play` 和 `Schedule-Vision-Play` **仍使用实测准备门槛**，
保留准备超时终止，不会强行绕过安全条件起步。
未获得真实后方缓存路径点时保持原 XY，Z 继续指向道路上方 1cm；
采用已有世界高度平滑器，初始收敛阶段不是瞬间跳到 1cm。
这个高度只在既有工作空间内可达；未改变 z 工作空间限制，也不伪造缓存路径。

## 本机启动训练

在开发电脑终端执行，不是在 B2 板载电脑：

```bash
cd /home/dong/b2_z1
bash scripts/rsl_rl/train_amp_vb12_schedule.sh
```

默认 256 环境、**新增 500 轮**、每 50 轮保存。长训没有自动启动。
VS Code 对应 `VB12：阶段推进修复＋退化监控（续训500轮）`。
Ctrl+C 会停止该入口自己创建的训练进程，不会停止其他任务。

监控入口将完整训练输出保存在 `outputs/vb12_guard_<时间>_<PID>/training.log`，
`status.json` 保存命令、真实训练目录和停止原因。
任何 TensorBoard 标量出现 NaN/Inf 会停止。
至少积累 200 轮后，按最近 50 轮检查：
有运动阶段却没有命令，或有运动命令而实际向前运动比例低于 15%，会停止。
标称重载域单独进行同类检查，不能用轻载环境的运动掩盖重载失败；
枪接触预算导致的终止比例连续窗口超过 85% 也会停止。
这些是严重退化保护，不是步态质量或部署验收标准。

重点观察：

- `VB12/scheduled_motion_fraction`、`requested_motion_fraction`：时间阶段是否真正转换成速度任务。
- `VB12/movement_given_request`：有向前命令时，实际身体速度是否大于 0.05m/s。
- `VB12/full_load_actual_commanded_motion_fraction`：标称负载下的运动比例。
- `VB11/full_load_ready_fraction`、TCP 误差、枪接触力：不能只看总体 reward。
- `Episode_Termination/*`：物理失败率必须下降；训练不再包含 startup_not_ready 项。

```bash
/home/dong/anaconda3/envs/env_isaaclab/bin/tensorboard \
  --logdir /home/dong/b2_z1/logs/rsl_rl/b2_z1_amp --port 6006
```

## 完整视觉播放

先完成正式训练和筛查，再用 VS Code `VB12：完整视觉验收（选择训练后权重）`，
输入该次正式训练的 checkpoint 绝对路径。标称软管 3kg，最多播放 300 秒；
600 秒为单 episode 上限，并非保证无重置。

```bash
cd /home/dong/b2_z1
B2_Z1_HOSE_MASS_KG=3 \
  /home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/play_va15_vision.py \
  --vb12 --checkpoint /绝对路径/正式训练目录/model_XXXXX.pt \
  --episode-length-s 600 --max-steps 15000
```

验收必须包括：冷起步、标称 3kg、至少 60 秒完整视觉、无物理重置，
真实后方缓存接管、TCP/机身中心线误差、机身与枪上端的稳态晃动。
短测通过只表示代码运行和指标有效，不能据此宣称解决抖动或可以上实机。

## 回归测试

```bash
/home/dong/anaconda3/envs/env_isaaclab/bin/python -m unittest discover \
  -s scripts/vision -p 'test_vb*.py'
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/test_loaded_tool_stability.py
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/test_path_guidance.py
```

## 2026-10-14 实测记录

- 65 项 CPU 回归通过；Python、Shell 和 VS Code JSON 检查通过。
- 128 环境、50 轮短训（12300～12349）：
  `logs/rsl_rl/b2_z1_amp/2026-10-14_14-57-01_VB12_schedule_smoke`。
  TensorBoard 全部标量、actor/critic 参数有限；Z1 std=0.0805～0.0823、均低于 0.10；
  网络仍为 70→18。末 20 轮运动阶段比例 49.86%、实际有速度命令比例 25.21%，
  有命令时实际向前比例 88.45%。这些是环境时间比例，不是任务成功率。
- **短测策略没有通过部署验收**：标称负载域准备成功指标仍为 0；
  3kg 冷起步、30 秒合成轨迹回放触地预算重置 6 次，速度命令比例 0，
  TCP 误差 P95=20.47cm。位移统计包含站立漂移，不能当成行走。
  见 `outputs/vb12_smoke_screen_20261014/summary.json` 与 `3kg.csv`。
- 为隔离命令修复，另用**旧 VB11 12800 参考权重**、VB12 视觉配置、3kg 进行 30 秒回放。
  准备门槛首次通过 3.0 秒，真实 TCP 缓存点首次接管 16.5 秒。
  5～15 秒目标 world-Z=0.0099999～0.0100135m、P95-P05=0.00000483m，
  此期间机身高度 P95-P05=4.87cm；目标不再复现机身的周期性上下运动。
  同期 TCP 实际跟踪误差 P95 仍为 10.11cm。
  19.26、23.66 秒因枪接触预算重置，所以它也不是合格策略。
  异步视觉不是逐帧匹配回放，不能把全部差异都归因于这一处修改。
  见 `outputs/vb12_smoke_screen_20261014/vision_reference12800_3kg.csv`，
  相机/识别/控制原始日志在 `/tmp/b2_z1_va15_vision_ate_uasd`。
- 全部短测进程已结束。尚未自动开始默认 500 轮正式训练，未更新部署权重。
