# VA13 中心线与航向微调

从 `2026-09-24_09-50-29_va7_whole_body_from_scratch/model_7999.pt`
初始化，保留 70 维观测、18 维全身动作及 AMP。机械臂仍由同一策略控制。

现有 VA13 配置包含中心线误差、航向误差和带方向的角速度误差；保留
TCP 跟踪、机械臂动作平滑和零空间速度约束。训练时用 VA7 的动作均值
作为参考，机械臂参考损失权重为 5.0，腿部为 0.6；转弯时减弱腿部参考
约束。参考网络仅在训练中使用。

课程先使用 65% 直线和半径 18–26 m 的圆弧，速度 0.16–0.24 m/s。
满足跟踪、存活率和姿态条件后才增加曲率和速度。机身低于 0.39 m
终止回合，另有腿部碰地及低姿态惩罚。

## 启动

```bash
cd /home/dong/b2_z1
bash scripts/rsl_rl/train_amp_va13_low_noise.sh
```

默认 1024 个环境，在 VA7 权重基础上增加 300 轮，约到编号 8298。
每 25 轮保存。重新执行脚本仍从 VA7 初始化，不会自动选择最近的 VA13。
这是一轮待评估的微调实验，不应将最后一个检查点直接作为部署版本。

VS Code 中选择 `VA13：保留机械臂的中心线与航向微调（300轮）`，按 F5。

```bash
conda activate env_isaaclab
tensorboard --logdir /home/dong/b2_z1/logs/rsl_rl/b2_z1_amp --port 6006
```

重点查看 `Episode_Reward/base_centerline_error`、`base_heading_error`、
`signed_yaw_rate_error`、TCP 跟踪奖励、`Loss/arm_teacher`、
`Loss/leg_teacher`、机身低高度和非法接触终止率。负误差奖励越接近零越好，
但必须同时检查前进速度、存活时间和 TCP 跟踪，避免靠停下或跪行降低误差。

## 已完成验证

2026-09-28 使用 128 个环境完成 10 轮优化，退出码 0，84 项 TensorBoard
标量均为有限值；训练参考损失正常记录。输出目录为
`logs/rsl_rl/b2_z1_amp/2026-09-28_10-01-56_va13_resume_validation`。
这只验证运行链路，尚不足以验证长回合步态、左右弯跟踪或实机效果。

探索噪声按实际关节名称解析 12 个腿部动作索引后设为 0.18，不能直接
使用前 12 个动作：动作顺序中存在机械臂关节。无效噪声参数会报错。
