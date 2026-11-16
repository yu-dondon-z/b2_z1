# 训练脚本

## 当前VD14姿态候选（最终5999）

- [play_vd14.sh](play_vd14.sh)：`bash scripts/play_vd14.sh --viewer --real-time --duration 30`；固定VD14最终模型，真实视频及200Hz日志。
- [train_vd14.sh](train_vd14.sh)：从保留的VD13最终6199开始，只加强水平姿态和前脚横向最小站距代价；独立实验，不改旧模型／增益／AMP。
- [长训练、TensorBoard、中止恢复和验证范围](../docs/VD14_POSTURE_TRAINING_20261116.md)。
- [实际Play／MuJoCo命令与结果](../../rl_sar_b2_z1/docs/reports/vd14/VD14_PLAY_MUJOCO_20261116.md)：姿态改善、两端30秒；双视觉2.66秒接地保护，未验收。

## 保留VD13对照

- [play_vd13.sh](play_vd13.sh)：固定最终6199，真实Isaac Play／视频／200Hz日志；窗口播放用`bash scripts/play_vd13.sh --viewer --real-time --duration 30`。
- [train_vd13.sh](train_vd13.sh)：75cm夹持点、0.95m工作点的固定配置长训练。已训练到6199，本次不重训。
- [训练／TensorBoard／中止恢复](../docs/VD13_MOUNT75_TRAINING_20261115.md)。
- [rsl_rl/train.py](rsl_rl/train.py)、[rsl_rl/play.py](rsl_rl/play.py)：共用训练与播放实现，保持原路径。

[VD13实际结果与MuJoCo入口](../../rl_sar_b2_z1/docs/reports/vd13/VD13_PLAY_MUJOCO_20261116.md)：Play与真值回放均完成30秒；双视觉2.24秒支撑不足停止，尚未验收。下面的VD12脚本不等同于VD13部署。

## 保留的对照与诊断

| 用途 | 入口 |
| --- | --- |
| VD12最终5999 Play | [play_vd12_final.sh](play_vd12_final.sh) |
| VD12阶段／VD11穿模复核 | [play_vd12_checked.sh](play_vd12_checked.sh)、[play_vd11_checked.sh](play_vd11_checked.sh) |
| 已完成的VD12禁区检查 | [probe_vd12_guard.sh](probe_vd12_guard.sh)，无需为整理重复运行 |
| 离线轨迹、落脚与训练分析 | [diagnostics/README.md](diagnostics/README.md) |
| 环境基础工具 | [list_envs.py](list_envs.py)、[zero_agent.py](zero_agent.py)、[random_agent.py](random_agent.py) |
| 历代训练／VD3 Play入口 | [archive/README.md](archive/README.md) |

历史shell已按版本归档，原 `scripts/train_vd*.sh`／旧VD3 Play路径保留兼容软链接；原训练参数不变，不代表建议重新训练。共用Python、任务配置、资产和测试不按版本号删除。

测试见 [tests/README.md](../tests/README.md)，归档和恢复见 [维护记录](../docs/maintenance/README.md)。新离线分析放入 `scripts/diagnostics/`，生成结果放入 `outputs/`。
