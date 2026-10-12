# VB8：负载下的机身与焊枪稳定性

2026-10-12，独立分支 VB8；不改实机部署，不覆盖 VB7 权重。

## 检查证据

3 kg 完整视觉播放的 12–55 秒稳定段（P95–P05 为稳健波动范围，非绝对峰峰值）：

| 策略 | 机身高度波动 | 实际 TCP 高度波动 | joint4 贴近下限的时间比例 |
|---|---:|---:|---:|
| VB7 10000，已有记录 | 3.42 cm | 1.12 cm | 100% |
| VB7 13000，本次重播 | 2.88 cm | 1.19 cm | 99.5% |

13000 本次约 60 s 没有重置，移动 9.10 m，但完整窗口 TCP 目标误差 P95
14.41 cm，较 10000 旧记录的 11.14 cm 更大，不能仅凭机身稍稳就替换。
比较包含异步视觉及不同运行时间，属于诊断而非多种子认证。
CSV：`outputs/vb8_diagnosis/vb7_13000_vision_3kg.csv`。

已确认的代码问题/目标冲突：旧 `ee_stability` 惩罚相对机身的加速度，
而维持世界中的枪尖不动正需要抵消机身加速度；旧姿态目标又固定在
机身坐标系，世界姿态随机身 roll/pitch。旧任务没有完整姿态跟踪奖励，
因此改变姿态命令本身不足以改善，VB8 还直接惩罚实际枪轴倾斜与角速度。
joint4 长时间抵住 URDF 下限 -1.5184364 rad 是事实；它对晃动的因果贡献
需通过新策略对比验证，不能把每一次机械臂运动都视作无用抖动。

## 针对性修改

- 机身世界垂向速度 pseudo-Huber cost，权重 -0.35、尺度 0.10 m/s，
  不惩罚水平前进；保留原高度目标、足底离地高度和步态奖励。
- 枪体世界横滚/俯仰角速度 cost：-0.20、尺度 0.50 rad/s，放行 yaw。
- 实际枪的局部 z 轴对齐重力轴（正反向等价）的 bounded cost：-0.30。
- 六轴实际与请求姿态距离物理限位的 0.10 rad 缓冲区 cost：-0.40。
  同时检查期望目标，避免动作越界而实际关节卡在限位上的漏洞。
- TCP 世界法向速度：权重 -0.20 → -0.40，垂向尺度 0.20 → 0.10 m/s。
  沿轨迹前进不受惩罚，横向尺度保持 0.20 m/s。
- 仅机械臂动作变化惩罚 -0.12 → -0.16；机身角速度惩罚改为 -1.25。
- 取消旧相对机身稳定惩罚（权重 0），保留世界稳定、零空间与 TCP 跟踪。
- 枪姿态命令改为重力对齐、yaw 跟随机身，仅新 VB8 任务开启。
  旧 VA/VB 播放不改变。平坦道路假设不能直接用于未知倾斜道路。

仍为一个 18 自由度策略；不增加外部 IK，不冻结机械臂，不额外过滤腿部
动作，不加大 PD，不减小真实负载，不禁用枪的质量/惯量/碰撞。
VB7 负载范围、轻负载代理混合、AMP、路径/航向、终止和课程升级门槛保留。
续训重建环境从课程第0级开始复核；恢复权重不等于恢复原课程进度。

## 先短续训，再筛选

保留 VB7 10000 的行走能力，以小学习率 1e-5 续训 500 轮，而非直接
再跑 8000 轮。默认约保存 10000/10050/…/10499；具体以实际目录为准。
本次实现的短训用于验证接线和数值，不代表已完成这次正式 500 轮训练。

VS Code F5：`VB8：机身＋焊枪稳定微调（从10000续训500轮）`。

```bash
cd /home/dong/b2_z1
bash scripts/rsl_rl/train_amp_vb8_quiet_load.sh
```

内存紧张可用 `VB8_NUM_ENVS=256`；轮数用 `VB8_ITERATIONS=500` 设置。

训练后的完整视觉验证（将路径替换为实际经过筛选的 VB8 检查点）：

```bash
cd /home/dong/b2_z1
B2_Z1_HOSE_MASS_KG=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/play_va15_vision.py \
  --vb8 --checkpoint /absolute/path/to/VB8/model.pt \
  --episode-length-s 600 --max-steps 3000 \
  --motion-trace-csv outputs/vb8_diagnosis/trained_3kg.csv
```

不要用 VB7 权重配 `--vb8` 当作正式修复；新命令契约需要对应微调。
Ctrl+C 关闭此会话的仿真和视觉子进程。原 VB7 调试仍可对比。

## 监控和验收

```bash
/home/dong/anaconda3/envs/env_isaaclab/bin/tensorboard \
  --logdir /home/dong/b2_z1/logs/rsl_rl/b2_z1_amp --port 6006
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/test_loaded_tool_stability.py
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/analyze_loaded_tool_motion.py \
  outputs/vb7_10000_vision_3kg/trace.csv outputs/vb8_diagnosis/trained_3kg.csv
```

观察新增 `base_world_heave`、`gun_world_tilt_rate`、`gun_vertical_axis`、
`arm_limit_buffer` 和原 TCP、机身高度、速度跟踪、足底抬高/接触周期、
航向、课程、异常终止，不以总奖励一项判定好坏。
同负载/路径/时长多次验证；1/3/6 kg 和轻工具代理分别验收。
希望机身稳健起伏减到约 2 cm 内、TCP 到约 5 mm 内、joint4 不长期抵限；
这是目标，不是已达到的指标。必须同时保留前进距离、速度、抬腿和
轨迹精度，无跌倒/下蹲重置，才能升级为推荐策略。
不足时应调整而不是盲目长训；不据此声称实机安全或 10 kg 负载认证。

## 本次配置验证

- 9 项 CPU 单元测试通过（坐标、转向放行、限位与请求目标、历史命令兼容、
  重力姿态转换）；原局部路径几何 4 项测试通过。
- 最终配置从 VB7 10000 续训 50 轮、128 个环境，保存到 10049；
  `2026-10-12_15-05-11_VB8_quiet_load_final_smoke` 的 96 项标量均有限，
  未发现 NaN/Inf。新增奖励实际参与了训练。
- 这是软件/数值和短候选测试，不是正式 500 轮的收敛或性能证明。
- 最终短训 10049 的 3 kg 完整视觉重播 60 s：移动 10.73 m，0 重置，
  最大机身倾角 4.12 度；12–55 s 机身起伏 4.04 cm、TCP 起伏 1.13 cm，
  joint4 仍 100% 靠近下限。该窗口 TCP 目标误差 P95 为 8.66 cm。
  对比旧 10000 的 3.42 cm / 1.12 cm / 6.80 cm，短训还没有改善，
  **不升级为推荐，不修改现有 VB7 播放入口**。不能用“无 NaN”冒充性能修复。
  记录：`outputs/vb8_diagnosis/vb8_smoke_10049_vision_3kg.csv`。
  正式微调仍从原 VB7 10000 开始，不从这个短训检查点继续；500 轮是
  待筛选试验，不保证改善，不宜直接延长到 8000 轮。
