# VD14：先纠正机身后倾和前脚过窄

2026-11-16。用户在VD13 Play中发现后部下沉、前脚靠拢。保留VD13最终6199及原配置，新建隔离的VD14姿态奖励候选。**现已完成6000轮，最终5999**；两端动作样本30秒，姿态明显改善。部署侧修正拍摄→控制转换、后视暗缝支持和四元数翻号后，曲线／直线双视觉样本均完成30秒，明显挥杆减轻；目标仍主要为有界估计，尚未部署验收。[Play／MuJoCo命令与实际结果](../../rl_sar_b2_z1/docs/reports/vd14/VD14_PLAY_MUJOCO_20261116.md)。训练配置、默认actor和实机未改动。

正式模型：`logs/rsl_rl/b2_z1_vd14_posture_amp/2026-11-16_10-41-45_vd14_posture_long_20261116/model_5999.pt`。真实Isaac播放：`bash scripts/play_vd14.sh --viewer --real-time --duration 30`。以下训练命令与无学习检查是实现阶段记录，不代表需要重新从VD13开始。

## 为什么改这两项

VD13原回放样本（`outputs/vd13_checked_play/play_20261116_094343_ovd2114q`，3秒后）后髋比前髋低的中位数约12.1cm，前脚横向站距中位数约21.0cm。AMP有工作，但它是软辅助奖励，不保证策略逐项符合参考姿态，不能用“有AMP”替代动作验收。

全部4769帧AMP参考动作的前脚横向间距P5／中位数约45.0／51.8cm，约96.9%的帧不低于42cm；根部四元数（数据集xyzw）对应的总倾角P95约3.73°。选择宽松边界，不要求复刻每帧动作。

VD14只变更两项：

| 项目 | VD13 | VD14 |
| --- | --- | --- |
| 总倾角超限平方代价 | 死区0.10rad，尺度0.20rad，权重−3 | 死区0.07rad（约4°），尺度0.15rad，权重−8 |
| 前脚过窄平方代价 | 无直接项 | 机身系有符号`FL.y−FR.y`低于0.42m才惩罚，尺度0.10m，权重−2 |

前脚代价不因超过42cm继续给奖励；包含摆动期，不伪造接触、不人为挪脚、不做硬终止。使用完整机身坐标变换，避免转弯时把世界Y误当横向。更强的姿态代价仍是软奖励，不能宣称已经解决下沉。

保持2.93kg、1.1m管身、75cm夹持、0.95m工作距离、枪体禁区、70维actor输入、18维关节位置动作、PD增益、速度限制、PPO/AMP参数、对称增强、原枪尖抖动和200Hz落脚冲击奖励不变。没有增加actor特权信息。

## 长训练

```bash
cd /home/dong/b2_z1_VD2
bash scripts/train_vd14.sh \
  --run_name vd14_posture_long_20261116 \
  --max_iterations 6000
```

入口固定从下面的**实际VD13模型**加载policy／critic／AMP归一化与判别器，并恢复相同安装的完整地面课程；重置优化器和本次轮号，在新实验中训练6000轮。不是对旧任务直接`--resume`，也不是从零学习走路。

```text
logs/rsl_rl/b2_z1_vd13_mount75_amp/2026-11-15_23-51-49_vd13_mount75_resume/model_6199.pt
SHA256: 6121df7a2c75da9183486146773c5698852ff3c3e48c0353f19ad05b2b1dede0
```

新日志：`logs/rsl_rl/b2_z1_vd14_posture_amp/`。256环境，每100轮保存，禁止自动删除旧检查点。2026-11-16检查持久盘剩余约15GB，单个检查点约12MB；训练中注意磁盘余量，不自动清理历史日志。

不要根据前几百轮结果反复改奖励。固定配置完成长训练后结合曲线和Play判断；若指标仍持续改善，再按同一配置追加。总奖励因权重变化不能直接与VD13横比，6000轮也不自动等于收敛。

## TensorBoard

```bash
/home/dong/anaconda3/envs/env_isaaclab/bin/python -m tensorboard.main \
  --logdir /home/dong/b2_z1_VD2/logs/rsl_rl/b2_z1_vd14_posture_amp \
  --host 127.0.0.1 --port 6006
```

打开 `http://127.0.0.1:6006`。关注`Episode_Reward/base_tilt`、`Episode_Reward/vd14_front_width`、`Metrics/ee_target/vd14_front_width_m`、`Metrics/ee_target/vd14_front_width_below_min`、`Metrics/ee_target/vd14_base_pitch_rad`，并同时检查速度／枪尖跟踪、落脚冲击、存活时长和AMP判别器。只看总奖励不够；训练期指标有随机重置／扰动，最终还需受控回放。

## 中止后续训

Ctrl+C后从最近**已经完整保存**的VD14检查点恢复；中断当下未保存的进度不能保证保留：

```bash
cd /home/dong/b2_z1_VD2
bash scripts/train_vd14.sh \
  --resume \
  --load_run '.*_vd14_posture_(long|resume).*' \
  --checkpoint 'model_.*.pt' \
  --run_name vd14_posture_resume \
  --max_iterations 3000
```

该表达式只匹配正式`long`／`resume`目录，排除本次无学习smoke目录；启动前选择最新匹配run及其中最新检查点，再创建新日志目录。`3000`是**再训练3000轮**。同版本恢复policy／AMP／优化器／课程／轮号；修改姿态奖励或混用VD13检查点会被拒绝。加载模式互斥，`--resume`时不会再隐式从6199热启动。不保证恢复随机数、重放缓存或每个环境的瞬时状态，因此不是逐位连续复现。

## 实现阶段已验证与未验证（训练前记录）

- CPU回归覆盖奖励方向、转动／平移不变性、交叉脚处理、旧资产保留、源版本拒绝、检查点保存及同版本恢复。新增18项，加VD13的10项、VD12训练22项、旧倾角／支撑22项，共72项通过。
- 实际Isaac无学习检查：4环境、150个50Hz控制步（3秒），保持200Hz物理与真实终止，成功加载6199并恢复满地面课程；0重置／0终止。原始结果见[probe日志](../logs/rsl_rl/b2_z1_vd14_posture_amp/2026-11-16_10-25-05_vd14_no_learning_smoke_fixed_20261116/vd10_policy_probe.json)，同目录保留资产审计、配置和200Hz追踪。
- 首次nominal诊断因重置方式在checkpoint校验前被改而拒绝，已修正为先校验／恢复原训练配置，再显式做诊断重置；失败run也保留。
- 对比实际保存配置，除上述两处奖励及smoke所需的4环境／名义重置／输出缓存，新旧环境配置一致；PPO/AMP配置未变。没有任何优化器更新，当前行为仍是VD13策略，**没有新的改善结果**。

训练前的验证只确认管线。训练后的姿态／枪尖／落脚及视觉实跑另见[最终Play记录](../../rl_sar_b2_z1/docs/reports/vd14/VD14_PLAY_MUJOCO_20261116.md)；已新增独立VD14 MuJoCo入口，旧入口与默认actor不动，不做板载同步或实机运动。
