# 历史启动脚本

2026-11-15：VD3–VD12的14个旧训练／VD3 Play shell按版本存放于本目录，原`scripts/`路径保留相对软链接。直接调用归档文件或旧路径均可定位同一训练工作区；原任务、参数、模型选择和恢复语义不变，只增加历史入口提示。

当前入口是 [train_vd13.sh](../train_vd13.sh)，使用前先读 [VD13训练说明](../../docs/VD13_MOUNT75_TRAINING_20261115.md)，不要在已有训练运行时重复启动。

| 子目录 | 旧入口 |
| --- | --- |
| `vd3/` | `train_vd3.sh`、`train_vd3_symmetry.sh`、`play_vd3.sh`、`play_vd3_curve.sh` |
| `vd4/` | `train_vd4.sh` |
| `vd5/` | `train_vd5.sh`、`train_vd5_startup.sh` |
| `vd6/`–`vd12/` | 对应`train_vdN.sh` |

这是旧实验复现入口，不是建议重训列表。共用Python和受保护Play不移动。原文备份、迁移清单与恢复说明见 [维护记录](../../docs/maintenance/README.md)。
