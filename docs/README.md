# B2+Z1 训练文档

2026-11-16：当前候选为 **VD14最终5999（75cm夹持点）**。两端姿态回放改善；视觉接口修正后挥杆减轻，支持不足零行走等待／恢复的60秒样本完成，未完成验收。目录名 `b2_z1_VD2` 不代表只包含VD2；`CODE_VERSION.json` 保留VD5历史基线，不代表当前候选。源码与最终模型见 [VD14 GitHub快照](VD14_GITHUB_SYNC_20261116.md)。

| 用途 | 入口 |
| --- | --- |
| 当前VD14：Play、MuJoCo及双视觉实跑结果 | [启动命令与实际结果](../../rl_sar_b2_z1/docs/reports/vd14/VD14_PLAY_MUJOCO_20261116.md) |
| VD14姿态候选：已完成6000轮，最终5999 | [修改、训练、TensorBoard、中止恢复](VD14_POSTURE_TRAINING_20261116.md) |
| 当前VD13：Play、MuJoCo命令与真实结果 | [播放／视觉记录](../../rl_sar_b2_z1/docs/reports/vd13/VD13_PLAY_MUJOCO_20261116.md) |
| 当前VD13：配置、训练、TensorBoard、中止恢复 | [VD13训练说明](VD13_MOUNT75_TRAINING_20261115.md) |
| 新安装验证与为什么需要训练 | [VD13候选报告](../../rl_sar_b2_z1/docs/reports/vd13/VD13_MOUNT75_DECISION_20261115.md) |
| 两个工作区、保存位置与日常入口 | [工作区说明](TRAINING_WORKSPACE.md) |
| VD12最终模型与视觉失败对照 | [最终Play](../../rl_sar_b2_z1/docs/reports/vd12/VD12_FINAL_PLAY_MUJOCO_20261114.md)、[双视觉记录](../../rl_sar_b2_z1/docs/reports/vd12/VD12_VISION_DEPLOYMENT_20261114.md) |
| 历代训练方案／VD5发布记录 | [历史归档](archive/README.md) |
| 各版本效果与失败证据 | [阶段评估](../../rl_sar_b2_z1/docs/reports/README.md) |
| 脚本、共用工具与测试 | [脚本索引](../scripts/README.md)、[测试说明](../tests/README.md) |
| 曲线跟踪设计背景 | [base_curve_tracking.md](base_curve_tracking.md) |
| 本次整理、备份与恢复 | [维护记录](maintenance/README.md) |

当前VD14只在训练前修改姿态奖励，后续没有再改奖励或追加短训。部署侧已补传感器局部运动补偿、像素证据、四元数符号连续性及后视等待／恢复；动作回放和有限视觉样本不代表连续视觉、砸地或收敛验收通过。默认actor未替换，不涉及板载操作；VD13条目保留作对照。

日常说明放在 `docs/`；阶段评估放在 `docs/reports/<版本>/`；已被替代的方案放在 `docs/archive/<版本>/`。历史记录的“当前”“已同步”等表述只对应原日期。
