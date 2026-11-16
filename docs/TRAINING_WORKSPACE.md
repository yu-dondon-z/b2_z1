# 训练工作区

只在 `/home/dong/b2_z1_VD2` 训练；MuJoCo工作区为 `/home/dong/rl_sar_b2_z1`。历史目录名不代表当前任务版本，不另建同名训练副本。

2026-11-16当前候选为 **VD14最终5999：75cm夹持点／0.95m工作距离**。已完成6000轮和两端30秒动作回放；后部下沉／前脚过窄明显改善，双视觉2.66秒接地保护停止。`CODE_VERSION.json` 记录的是VD5已同步基线，不能用它推断当前候选。

姿态奖励与恢复方式见[VD14训练说明](VD14_POSTURE_TRAINING_20261116.md)，[本次Play与视觉证据](../../rl_sar_b2_z1/docs/reports/vd14/VD14_PLAY_MUJOCO_20261116.md)。VD13原模型／配置保留；连续视觉未验收，不为视觉接口问题盲改奖励或续训。

## 日常入口

- [VD14姿态候选：训练、TensorBoard和中止恢复](VD14_POSTURE_TRAINING_20261116.md)。
- [VD13配置、训练、TensorBoard和中止恢复](VD13_MOUNT75_TRAINING_20261115.md)。
- 当前播放入口：[scripts/play_vd14.sh](../scripts/play_vd14.sh)；[实际结果与MuJoCo命令](../../rl_sar_b2_z1/docs/reports/vd14/VD14_PLAY_MUJOCO_20261116.md)。
- 保留的训练入口：[scripts/train_vd13.sh](../scripts/train_vd13.sh)，本次没有追加训练。
- [文档索引](README.md)、[脚本索引](../scripts/README.md)、[测试说明](../tests/README.md)。
- [新安装验证与训练决定](../../rl_sar_b2_z1/docs/reports/vd13/VD13_MOUNT75_DECISION_20261115.md)。

## 保存位置

`source/` 保存任务与资产，`scripts/rsl_rl/` 保存共用训练实现，`logs/rsl_rl/` 保存训练日志和检查点，`outputs/` 保存诊断结果。全部保留；不要在训练期间切换／恢复其他源码版本。

旧版本方案见 [历史归档](archive/README.md)。旧训练shell已移入 `scripts/archive/<版本>/`，原路径保留兼容软链接。存在历史入口不等于建议重训，也不能把旧模型直接用于新安装。

## 历史迁移记录

原工作区合并、源码快照、VD3复现及旧目录映射的完整记录保存在 [2026-11-05迁移记录](archive/workspace/TRAINING_WORKSPACE_20261105.md)。本次整理与原文备份见 [维护记录](maintenance/README.md)。

仿真验收须检查枪尖摆动、落脚冲击和双视觉跟踪。当前未替换默认actor；未经新授权不做板载同步或实机运动。
