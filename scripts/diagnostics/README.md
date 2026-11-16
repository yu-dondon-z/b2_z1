# 独立训练分析

这些工具从主 `scripts/` 移入此目录，按需运行。以下命令从项目根目录执行，依赖原有 Python 环境和输入数据。

| 脚本 | 用途 |
| --- | --- |
| [play_vd14_checked.py](play_vd14_checked.py)、[vd14_play_contract.py](vd14_play_contract.py) | 从`../play_vd14.sh`启动；固定最终5999，额外核对VD14实际姿态奖励及profile，复用不变的75cm物理模型 |
| [review_vd14_play.py](review_vd14_play.py) | 离线比较已有30秒连续首次episode：完整URDF FK、机身系有符号前脚宽、髋高度／倾角、枪尖频带与相位对齐落脚，不重跑物理 |
| [play_vd13_checked.py](play_vd13_checked.py)、[vd13_play_contract.py](vd13_play_contract.py) | 从`../play_vd13.sh`启动；最终6199／75cm安装，复用已测试Play循环，先校验VD13专用模型、TCP和完整地面课程 |
| [analyze_vd8_training.py](analyze_vd8_training.py) | VD8 TensorBoard 标量与检查点阶段统计 |
| [analyze_vd9_training.py](analyze_vd9_training.py) | VD9 TensorBoard 标量与检查点阶段统计 |
| [evaluate_vd2_mujoco.py](evaluate_vd2_mujoco.py) | 历史 VD2 检查点开环速度评估，依赖 `/home/dong/b2_z1/scripts/vision` |
| [play_vd12_checked.py](play_vd12_checked.py) | 从`../play_vd12_checked.sh`启动；真实VD12 Isaac评估，保留课程／终止及200 Hz pre-reset日志，不训练或导出 |
| [analyze_vd12_play.py](analyze_vd12_play.py) | CPU分析上述日志；使用明确的WORLD频带和触地协议，无匹配基线不宣称改善或部署 |
| [render_vd12_recorded_play.py](render_vd12_recorded_play.py) | 完整实际18轴与root状态的CPU FK简化视频；仅展示姿态，不冒充实时Isaac或MuJoCo动力学 |

```bash
python scripts/diagnostics/analyze_vd9_training.py --help
python scripts/diagnostics/analyze_vd8_training.py --help
```

仿真探测脚本仍位于 `scripts/rsl_rl/`，见 [脚本索引](../README.md)。静态资产测试已统一到 `tests/`。
