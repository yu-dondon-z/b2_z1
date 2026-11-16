# VD6：从 VD5 定向续训，改善落脚与枪尖抖动

当前状态（正式训练后更新）：已完成从 VD5 model_4498 追加 500 轮的 VD6 训练，最终检查点为 model_4997。已完成 Isaac Lab 路径任务及 Isaac Lab/MuJoCo 同指令对照。Isaac Lab 中连续性有改善，但 MuJoCo 横向枪尖抖动增大，暂不替换 VD5 部署策略。完整结果见 [VD6_TRAINING_REVIEW_20261111.md](VD6_TRAINING_REVIEW_20261111.md)。

帐号交接入口：[HANDOFF_VD6_GAIT_TCP_20261111.md](/home/dong/b2_z1/docs/HANDOFF_VD6_GAIT_TCP_20261111.md)，包含已完成事项、原始数据、待补诊断与下一步；[接管提示词](/home/dong/b2_z1/docs/VD6_TAKEOVER_PROMPT_20261111.md)。

本版仍位于历史目录 `/home/dong/b2_z1_VD2`，任务名、实验目录和训练入口均明确使用 VD6，避免与 VD2、VD5 策略混用。

## 为什么选择续训

同一个 VD5 model_4498，在 MuJoCo、Isaac Lab 分别运行 60 秒，关闭视觉，固定底盘直行指令 0.2 m/s 与机身坐标下的末端目标。两端仍能复现快速落脚和枪尖高频运动。因此训练侧也需要调整，不能把问题全部归给视觉跳变或 MuJoCo。

另外，两端又各完成了训练原本 0.70 m 工作距离下的 30 秒对照，同样存在快速落脚与高频运动，所以问题不只出在部署目标伸到了 0.80 m。

完整数据、测量定义与复现命令：
`/home/dong/rl_sar_b2_z1/docs/reports/vd5/VD5_GAIT_DIAGNOSIS_20261111.md`。

## 本次改动

| 新奖励 | 作用 | 初始权重 |
| --- | --- | --- |
| vd6_soft_landing | 足底在离平地最后 10 cm 的腾空下降段，对超过 0.35 m/s 的向下速度惩罚 | -1.0 |
| vd6_impact_excess | 使用最近四个 200 Hz 接触样本，惩罚超过 350 N 的足部竖直力；避免只看 50 Hz 末帧而漏掉冲击 | -0.10 |
| vd6_arm_slew | 对机械臂关节目标相邻 20 ms 的跳变量惩罚；0.02 rad 死区，0.05 rad 归一化 | -0.10 |
| vd6_arm_second_difference | 对机械臂关节目标的二阶差分惩罚；0.03 rad 死区，0.06 rad 归一化 | -0.05 |

软着地项只针对下降、临近地面、未接触三项同时满足的足端，不惩罚正常抬腿和支撑。力的阈值是奖励阈值，并非改变传感器接触判定。350 N 允许正常负载转移，初始权重仍需正式训练后检验。

保留原来的 70 维 actor 输入、18 维输出、0.25 动作缩放、0.2 m/s 训练前进速度、两秒启动过程、80% 参考姿态/20% 普通站姿重置，以及腿部镜像损失。机器人 PD 仍为 160/5。训练的末端工作距离仍为 0.70 m。

没有恢复 VD3 的旧 Bezier 导数/完整法兰姿态残差项。它们与 VD5 世界焊缝命令不匹配。新增的机械臂项直接约束指令连续性，无需使用错误的目标速度导数。

新奖励中的足端位置、速度、接触力只供训练奖励计算，未加入 actor 观测。平地高度项明确要求 plane terrain，换成高低地形时须重新定义相对地面高度。

## 从哪个策略继续

初始检查点：
`logs/rsl_rl/b2_z1_vd5_startup_amp/2026-11-10_11-47-57_vd5_startup_80ref_20stand_continue1000/model_4498.pt`

SHA256：`9af8e2ca67382d3186bb9ea145930c25797b06e6189963bb1d996fee91f3fedf`。

`--vd6-from` 明确迁移 VD5 Startup：保留 actor、critic、AMP 判别器与归一化状态、优化器、迭代号和课程；地面混合值恢复为 1.0，曲线课程恢复为 0.1。新 checkpoint 的实验标识为 `b2_z1_vd6_smooth_amp`。同版恢复使用 `--resume`，不能把迁移参数和 resume 混用。

课程结构未变，checkpoint 继续使用 `vd5_curriculum_state` 保存兼容课程数据，并由实验标识区别 VD6。旧版本任务仍按原规则加载。

## 已验证的内容

- 相关自动化测试：VD6 奖励/加载入口 4 项，VD5 命令/启动 20 项，AMP 10 项，VD3 训练加载 18 项通过。
- VD6 16 环境、2 轮及 20 轮训练完成，优化器、AMP 和课程恢复成功。日志分别位于 `logs/rsl_rl/b2_z1_vd6_smooth_amp/2026-11-11_15-13-21_vd6_smoke_16env_2iter` 与 `2026-11-11_15-14-41_vd6_smoke_16env_20iter`。这些只供程序验证，不能作为推荐策略。
- 从上述 VD6 `model_4517.pt` 使用 `--resume` 又完成两轮，验证同版断点恢复成功。日志：`logs/rsl_rl/b2_z1_vd6_smooth_amp/2026-11-11_15-20-49_vd6_resume_smoke_2iter`。检查 checkpoint 中的实验标识为 VD6，课程为 ground_blend=1.0、curve_level=0.1，优化器和 AMP 状态均存在。
- 8 环境、500 actor 步、普通站姿启动、原 VD5 策略的奖励触发验证完成，无摔倒/重置。报告：`outputs/vd6_reward_probe_20261111.json`。4000 个环境步中，软着地项触发 639 次，冲击项 735 次，机械臂一阶连续项 1880 次、二阶项 1276 次，所有值有限。该验证说明奖励接通，不说明训练后已有改善。

## 正式续训命令

以下为本次已经执行完成的 500 轮训练入口，保留用于复现；当前不建议直接照此再追加一轮训练，应先依据评估报告处理迁移后的横向抖动：

```bash
cd /home/dong/b2_z1_VD2
bash scripts/train_vd6.sh --max_iterations 500
```

输出：`logs/rsl_rl/b2_z1_vd6_smooth_amp/<时间>_vd6_soft_landing_arm_continuity/`。迭代号承接 4498，而 `--max_iterations 500` 表示追加 500 轮。

中断后使用该 VD6 运行目录继续，不要再次从原 VD5 初始化：

```bash
cd /home/dong/b2_z1_VD2
bash scripts/train_vd6.sh --resume --load_run '<本次VD6运行目录名>' --checkpoint 'model_*.pt' --max_iterations 500
```

## 训练后的判定

先看相同初态、相同指令下的软着地、机身上下运动、枪尖高频抖动和关节目标跳变。再回到双视觉闭环，检查 TCP 地面投影误差是否仍在用户要求的 5 cm 内，以及本体是否偏航、踩线。

同时检查实际前进速度、滑脚、抬腿高度和中途重置，防止用拖脚、少走路或降低速度换取低冲击。镜像项保留，但不以单独镜像指标替代实际行走效果。正式评估已完成，结果见上述评估报告；本次未发布新策略，也未同步板载。
