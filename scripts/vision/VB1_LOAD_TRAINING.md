# VB1：实体焊枪＋软管负载适应训练

VB1 保留 VA15 的策略观测、18 维动作、AMP 数据、奖励和真值曲线课程，改用实体焊枪 URDF，并在每个环境 **episode 重置时**独立抽样软管载荷。episode 内载荷固定。现有视觉播放仍可用 VB1 权重检查；VB1 训练使用合成真值曲线，不依赖分割网络。这是为了先分离负载适应与视觉误差。

| 阶段 | 等效悬垂质量 | 货车锚点扰动（各轴） | 松弛长度扰动 |
| --- | --- | --- | --- |
| 1 | 1–3 kg | 0 | 0 |
| 2 | 1–6 kg | ±(0.25, 0.25, 0.20) m | ±0.25 m |
| 3 | 1–10 kg | ±(0.50, 0.50, 0.40) m | ±0.50 m |

抽样范围是建模假设，不是软管已经测得的真实载荷。焊枪本体固定质量约 3.259 kg；其余力由软管模型施加。1–10 kg 适应能力需要训练和验证，不能由参数范围直接推断。

## 先做配置自检

```bash
cd /home/dong/b2_z1
conda activate env_isaaclab
B2_Z1_VB1_STAGE=1 python scripts/vision/check_vb1_load_randomization.py --headless
B2_Z1_VB1_STAGE=2 python scripts/vision/check_vb1_load_randomization.py --headless
B2_Z1_VB1_STAGE=3 python scripts/vision/check_vb1_load_randomization.py --headless
```

## 阶段 1：从 VA15 `model_5000.pt` 续训

```bash
cd /home/dong/b2_z1
conda activate env_isaaclab
B2_Z1_VB1_STAGE=1 python scripts/rsl_rl/train_amp.py \
  --task Unitree-b2z1-AMP-VB1-Loaded-Gun \
  --num_envs 512 --headless --max_iterations 1000 \
  --resume --reset_optimizer \
  --load_run 2026-09-28_15-07-31_va15_curriculum_from2500 \
  --checkpoint model_5000.pt \
  --run_name VB1_s1_1to3kg
```

`--max_iterations` 是**续训额外轮数**，并非总轮数；每 50 轮保存一次。日志仍位于 `logs/rsl_rl/b2_z1_amp/`，新运行文件夹带 `VB1_s1_1to3kg` 后缀。短时测试运行名为 `VB1_smoke*`，不要用作正式阶段的源权重。

## 阶段 2、3：仅在上一阶段通过后执行

从阶段 1 中选择经验证最好的检查点，替换 `<阶段1运行目录名>` 和 `<检查点文件名>`：

```bash
B2_Z1_VB1_STAGE=2 python scripts/rsl_rl/train_amp.py \
  --task Unitree-b2z1-AMP-VB1-Loaded-Gun \
  --num_envs 512 --headless --max_iterations 1000 \
  --resume --reset_optimizer \
  --load_run '<阶段1运行目录名>' --checkpoint '<检查点文件名>' \
  --run_name VB1_s2_1to6kg
```

阶段 3 从阶段 2 的最佳检查点继续，使用 `B2_Z1_VB1_STAGE=3`、`--max_iterations 2000` 和 `--run_name VB1_s3_1to10kg`。不要盲目选最后一轮；如果高负载使步态/TCP 明显退化，回到上一阶段检查力范围、锚点和奖励，再决定是否继续。

## 日志与分级验证

```bash
tensorboard --logdir /home/dong/b2_z1/logs/rsl_rl/b2_z1_amp --port 6006
```

重点查看 `VB1/hose_mass_kg`、`VB1/hose_force_mean_n`、`VB1/hose_torque_mean_nm`、`VB1/hose_tension_peak_n`，并和跌倒/终止率、平均 episode 长度、速度与航向跟踪、TCP 误差、中心线偏差、足部接触和 AMP 损失一起看。软管指标在 episode 完成时汇总；短时测试初期可能显示零。

固定负载、单环境、合成曲线扫描（把权重改为当前候选）：

```bash
python scripts/vision/evaluate_vb1_load_sweep.py \
  --checkpoint /home/dong/b2_z1/logs/rsl_rl/b2_z1_amp/<VB1运行目录>/model_<轮数>.pt \
  --masses 1 3 5 7 10 --max-steps 3000
```

输出目录含每档 CSV、播放日志和 `summary.json`：重置次数、TCP/中心线误差的 95 分位、机身最大倾角、运动距离和软管力/力矩。每阶段至少比较多个检查点，并用不同随机种子、锚点与松弛长度做更长的重复测试。**扫描脚本只验证合成真值轨迹**；通过后，再用 `scripts/vision/play_va15_vision.py --loaded-gun --checkpoint <权重>` 检查视觉闭环，最后做独立 MuJoCo 与实机分级验证。具体精度和执行器限值须以实际焊接要求和硬件数据设定，不能用脚本的“正常退出”代替安全验收。
