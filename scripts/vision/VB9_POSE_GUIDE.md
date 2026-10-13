# VB9：先退出机械臂限位，再扩展负载

2026-10-12，独立分支 VB9；VB7/VB8 权重保留，不改实机部署。

## 为什么不用同一配置继续长训

VB8 10499 的 3 kg 视觉回放，12–55 s 稳定窗口：机身起伏 4.90 cm，
枪上端连接点起伏 5.54 cm，枪尖 0.91 cm，joint4 仍 100% 贴近下限。
上端运动由实测关节角及 URDF 重建，重建 TCP 与 CSV 的误差 P95 约
0.0013 mm；不是仅根据目标点或渲染状态猜测。

静态、水平机身高 0.52 m、TCP 目标 (-0.775,0,0.01)m 的运动学参考解：
六轴约 `[0,2.7673,-2.4606,-0.3067,0,0]` rad，枪轴竖直，离关节硬限位
至少约 0.20 rad。它证明有另一种可达姿态，不证明碰撞/负载/实机安全。
旧策略的低噪声探索很难自行发现 joint3/joint4 大幅协调变化的这一分支。

## 实现

训练侧 `PoseReferenceTeacher`：

1. 读取当前 70 维 actor 观测中的机身坐标 TCP 目标与投影重力。
2. 从实体枪 URDF 建立 FK 与解析雅可比，计算物理枪轴而非 dummy_target 的轴。
3. 从上述清限位分支求有界参考；同时满足关节 0.12 rad 余量和既有动作
   scale=0.25、clip=5。有界步仅保留改善参考误差的候选，避免投影后发散。
4. 非有限输入、明显倾倒、目标超范围或位置误差≥1.5 cm/枪轴误差≥0.08
   的参考不参与辅助损失。没有强行把不可达目标推到关节限位。
5. 用 TCP 平移雅可比的阻尼零空间投影引导同一个 18 自由度 actor，
   主要调整冗余姿态，保留任务方向的跟踪/重力补偿自由度；另有5%的
   全姿态弱锚定。不是强制六轴精确对齐。前200个新增
   iteration 的有效系数从10%平滑升到100%；低有效覆盖率不放大损失。
   腿部参考仅弱约束0.05，不冻结腿、不冻结 actor trunk。

参考只是稳态姿态的训练辅助，不是假称对 0.2 EMA 的精确动态逆控制。
全部 rollout **只执行学生策略的输出**，仍经过原0.2 EMA。70/18布局、
观测默认关节角、动作尺度/剪裁、负载物理与实体枪碰撞不改变。
教师不导出、不接入播放/部署；导出仍是原 actor，无新增输入。
训练辅助损失为类 IK 参考，推理不是另一个 IK 控制器。

首个100轮全姿态MSE试验，3kg/60s纯策略回放无重置，joint4贴限位
比例从100%减到14.6%，机身起伏2.70cm、枪上端3.83cm，但稳态TCP
位置误差P95变成12.98cm。因此未升级为推荐，最终改用上述投影损失，
减少参考姿态与任务跟踪/负载补偿的冲突。

新增/加强：

- 上端连接点世界法向速度代价：-0.50，垂向尺度0.12、横向0.20 m/s。
  包含 `v(COM)+omega×(point-COM)`，沿路径前进不惩罚。
- 机身世界垂向速度 -0.80，枪体横滚/俯仰角速度 -0.35，
  枪轴重力对齐 -0.60，机械臂限位余量 -0.80。
- 原 AMP、物理足底抬高、接触周期、轨迹/航向奖励与跌倒终止保留。
- 训练重置25%采用清限位姿态、75%保留旧姿态；仅写 reset state，
  默认关节角和归一化不改。播放仍从旧名义姿态开始验证接管。
- 关闭原25%的 AMP参考状态初始化，避免它随后覆盖新的姿态混合。
  这是初始化分布改变，不是关闭 AMP 判别器或替换动作先验。

## 两阶段，不自动升级

阶段A：0–3kg等效软管、保留焊枪质量/惯量随机化（含轻工具代理），
参考辅助权重0.40，学习率2e-5，从原 VB7 10000 续训1000轮，约到10999。
每100轮保存；阶段A不能作为0–6/10kg已验证的证明。

```bash
cd /home/dong/b2_z1
bash scripts/rsl_rl/train_amp_vb9_pose_guide.sh 1
```

F5：`VB9：退出限位＋整枪稳定（阶段A，续训1000轮）`。
默认256环境；`VB9_NUM_ENVS=512` 可加大规模。
`VB9_ITERATIONS` 指新增轮数，不是绝对终止轮数。
每次重新启动续训，辅助权重的200轮渐增会重新开始；不是按权重文件的绝对轮次计算。

阶段A通过实测筛选后，才用指定检查点启动阶段B，0–6kg等效软管，
参考辅助降到0.08。**脚本不会自动选最后权重，也不会自动进入阶段B。**

```bash
cd /home/dong/b2_z1
VB9_LOAD_RUN=实际通过验证的阶段A目录名 \
VB9_CHECKPOINT=model_通过验证的轮次.pt \
bash scripts/rsl_rl/train_amp_vb9_pose_guide.sh 2
```

## 验证与监控

训练关注：`Loss/arm_teacher`、`Loss/pose_guidance_fraction`（有效参考比例，
不是损失/成功率）、腿部参考、PPO/AMP有限性、三个稳定代价、机械臂限位、
TCP误差、机身高度/抬腿/航向、异常终止和课程。
辅助 MSE 的数值受渐增权重影响，不能简单把跨阶段变大解释为姿态退化。

教师有效比例应显著非零；比例低要检查可达性、动作界限与坐标，不放宽
屏蔽阈值伪造通过。不同负载/路径和多次重复验证后才推荐权重。

完整视觉（使用实际经过筛选的 VB9 检查点）：

```bash
cd /home/dong/b2_z1
B2_Z1_HOSE_MASS_KG=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/play_va15_vision.py \
  --vb9 --checkpoint /absolute/path/to/VB9/model.pt \
  --episode-length-s 600 --max-steps 3000 \
  --motion-trace-csv outputs/vb9_diagnosis/vision_3kg.csv
```

Ctrl+C 关闭该会话的仿真和视觉子进程。不要给旧 VB7/VB8 权重贴 VB9 标签。

```bash
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/test_vb9_pose_guide.py
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/analyze_vb9_gun_motion.py \
  outputs/vb7_10000_vision_3kg/trace.csv outputs/vb9_diagnosis/vision_3kg.csv
```

希望同时改善机身与上端起伏、枪体摆角、限位占比，并保留行走距离、
速度、足底抬高与轨迹精度，无跌倒/下蹲重置；不允许用站着不走、减载、
禁碰撞或额外控制器来宣称策略成功。FK和屏蔽测试不等于真实碰撞/力矩认证。
尚未验证10kg、斜坡道路或实机；本任务不发送任何机器人控制命令。

## 2026-10-13：最终投影损失的小试记录

最终配置只做了100个新增iteration，未启动正式1000轮长训：
`2026-10-12_16-20-06_VB9_pose_nullspace_pilot/model_10099.pt`。
训练标量均有限，有效参考覆盖率85.4%；仍在200轮渐增的前半段。
12项参考/FK/雅可比/梯度/重置测试、9项稳定性测试、4项路径测试通过。
TorchScript导出通过70维输入→18维有限输出检查，无teacher权重。

以下为相同3kg等效软管、60s真值轨迹纯策略回放的12–55s窗口。
“起伏”是P95−P05，不是最大峰峰值；单次结果不是统计认证。

| 指标 | VB7 10000 | VB8 10499 | VB9 投影小试10099 |
| --- | ---: | ---: | ---: |
| 机身起伏 | 2.89cm | 4.33cm | 2.39cm |
| 枪上端连接点起伏 | 3.95cm | 5.05cm | 4.28cm |
| TCP位置误差P95 | 6.02cm | 6.95cm | 11.67cm |
| joint4距硬限位≤0.01rad占比 | 100% | 100% | 0% |

VB9全程约行走9.82m、零重置，joint4余量P05为0.0467rad。
这说明退出旧限位姿态和机身稳定性有早期进展，但枪上端尚未超过VB7，
TCP跟踪仍明显退化。因此**不替换VB7推荐权重，不视为阶段A验收通过**。
先正式训练阶段A并逐检查点筛选，不能凭奖励有限或无跌倒直接升阶段B。
阶段B的0–6kg长程动态效果尚未验证。

另完成一次60s完整视觉闭环（RHACrackNet实际分割＋历史轨迹＋学生actor），
日志 `/tmp/b2_z1_va15_vision_dh692ou5`，trace见下方。识别覆盖可见裂缝，
由实测关节重建的TCP与记录的实际TCP一致性P95约0.00105mm，全程零重置；12–55s窗口行走
7.22m，机身起伏2.88cm、枪上端4.38cm、TCP误差P95为13.17cm，joint4
贴限位占比0%。这仅说明完整视觉通路兼容、退出限位有效，**跟踪精度
尚未合格**；未进行多负载、多随机种子或实机测试。

```bash
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/analyze_vb9_gun_motion.py \
  outputs/vb9_diagnosis/nullspace_10099_vision_3kg.csv
```

复查小试真值数据：

```bash
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/analyze_vb9_gun_motion.py \
  outputs/vb9_diagnosis/nullspace_10099_actor_only_3kg/3kg.csv
```
