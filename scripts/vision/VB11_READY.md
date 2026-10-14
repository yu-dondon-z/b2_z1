# VB11：悬空待机、真实负载与平滑起步

独立分支VB11，从VB10的`model_12300.pt`续训。该权重在固定3kg抽查中
机身起伏/枪接触比12498小，但仍触地，仅是训练起点，不是合格部署策略。
不修改实机、不覆盖VB7推荐权重；VB10及更早任务保持原行为。

## 改动

- 同一个70维观测/18维动作全身AMP actor，保留0.25动作尺度、0.20 EMA、
  原PD增益、实体枪质量/惯量/碰撞和软管作用力；没有部署IK或额外动作滤波。
- 20%合成训练episode全程零速悬空待机，其余从4–7s零速准备开始。
  满足最终后向工作区中心、离道路1cm的目标后才平滑起步。不是要求
  相机前方点凭空出现在后方；视觉仍使用真实图像和已有历史轨迹缓存。
- 合成与视觉使用同一最终目标状态检查：位置误差<8cm、垂向误差<1cm、
  平滑命令距最终待机位置<5mm、最大机械臂速度<0.8rad/s、机身倾角<0.1rad、
  高度>0.45m、枪接触力<5N、枪底余量>5mm，连续0.30s后一次性锁存。
  行走中的普通跟踪误差不重新关门，避免反复启停；重置/真正视觉丢失完全
  停止才重新检查。重新待机的超时钟从该次停止开始，不继承上一段行走时间。
- 12s仍未准备好则终止，而不是长期零速存活刷奖励。持续触地仍为>10N、
  连续0.8s、初始4s宽限，没有延长宽限。计时使用命令实际重置年龄，不受
  runner初始随机episode_length_buf影响。
- 增加间歇触地债务：>5N按真实接触时长累积，非接触以0.25倍速消退，
  超过0.4s终止（初始4s宽限）。短暂冲击可恢复，反复撑地不能不断重置
  原来的连续计时器来绕过失败判定。
- 增加TCP有符号偏低代价、1.5cm尺度高度精度奖励、道路上方3mm下界代价、
  枪底5mm余量代价。支撑力代价-0.75→-1.5，接触惩罚-3→-6。TCP目标仍为
  道路上方1cm，不靠升高目标/缩小枪体/关闭碰撞来掩盖失败。
- 启动时按环境分配精确配额：50%保留URDF枪质量约3.259kg，并在每次重置
  固定3kg等效软管与4Ns/m阻力；10%轻工具代理（仍有正质量与碰撞），其余
  枪质量0.5–4.2kg、软管0–3kg随机化。轻载和零软管能力继续训练；不把软管
  等效质量宣称为已经认证的真实软管总重。
- 清限位姿态重置比例25%→75%，仍有25%旧姿态冷启动。不改观测默认关节角。
  在所有manager重置之后将EMA/上一动作按实际重置关节角初始化，避免清限位
  姿态被第一步拉回旧默认目标。下一次物理刷新后只捕获一次TCP起点，避免
  用尚未刷新连杆变换的旧TCP，或每步跟随实际TCP来抹掉跟踪误差。
- 保留原步态、抬腿、中心线、航向奖励和曲线课程升级门槛。训练无需启动视觉
  网络，仍用真值曲线；完整视觉须单独验收。

## 训练

```bash
cd /home/dong/b2_z1
bash scripts/rsl_rl/train_amp_vb11_ready.sh
```

默认256环境，新增2000轮，从VB10正式运行的12300轮重新开始，预计最后权重
为14299轮。学习率1e-5、重置优化器、每100轮保存、AMP预加载20000。
不要同时开多个训练/播放占满显存。

F5：`VB11：悬空待机＋真实负载（续训2000轮）`。
可用`VB11_NUM_ENVS`和`VB11_ITERATIONS`调整规模。建议500轮先检查满载组
和冷启动，未合格不盲目加到8000轮；正式脚本不会使用短测权重作为起点。

TensorBoard：

```bash
/home/dong/anaconda3/envs/env_isaaclab/bin/tensorboard \
  --logdir /home/dong/b2_z1/logs/rsl_rl/b2_z1_amp --port 6006
```

重点关注：

- `VB11/full_load_ready_fraction`、`full_load_tcp_height_abs_error_m`、
  `full_load_gun_contact_fraction`、`full_load_support_force_mean_n`：真实枪+3kg组。
- `VB11/qualified_moving_fraction`：通过门槛且确实执行非零速度的运动比例；
  `actual_moving_fraction`单独包含漂移，不能当作行走成功。
- `Episode_Termination/gun_contact_budget`、`startup_not_ready`和持续触地。
- 原TCP/速度误差、机身/整枪稳定、步态抬腿、异常终止、曲线课程。

门槛通过比例是步级统计，不是episode成功率；20%的全程静态域不应被当作
未执行行走的失败，也不能用它们的低速度误差证明整体运动能力。

## 运行验证（不是策略验收）

2026-10-14：16项VB11 CPU测试及13项VB10、12项VB9、9项稳定性、4项路径
回归测试共54项通过，VSCode JSON和shell语法检查通过。
曾做50/100轮探索性短测，随后加入重置EMA同步与一次性物理刷新后TCP捕获。
最终配置的50轮128环境运行：
`2026-10-14_10-09-05_VB11_coherent_reset_smoke/model_12349.pt`。
日志标量无NaN/Inf，固定满载组比例确实为50%；最后10轮门槛通过步级比例
约7.05%，满载组仅1.45%，间歇接触终止约57.24%、未准备好终止约27.09%。
说明新约束正在暴露旧策略的不足，绝不是已经可以部署的成绩。
不把短测权重设为正式训练起点或默认推荐策略。

该短测权重还做了16s/3kg完整视觉通路检查：有效裂缝点确实输入控制器，
查看叠加图确认点落在可见裂缝；发生1次间歇触地终止，速度命令全程为0，
门槛正确拒绝尚不合格的起步。不是视觉行走成功，不能由最后一帧证明全程
分割准确。CSV：`outputs/vb11_coherent_reset_vision_smoke.csv`；日志与叠加图：
`/tmp/b2_z1_va15_vision_zk9a7obv`。TorchScript导出70维输入→18维有限输出，
训练姿态教师未接入播放。全部自建短测进程已结束，没有启动正式2000轮。

## 指定训练后的权重验收

```bash
cd /home/dong/b2_z1
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/evaluate_vb1_load_sweep.py \
  --task Unitree-b2z1-AMP-VB11-Ready-Play \
  --checkpoint /absolute/path/to/model.pt --masses 1 3 --max-steps 3000 \
  --output-dir outputs/vb11_selected_load_check
```

完整视觉：

```bash
B2_Z1_HOSE_MASS_KG=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/play_va15_vision.py \
  --vb11 --checkpoint /absolute/path/to/model.pt --episode-length-s 600 \
  --max-steps 3000 --motion-trace-csv outputs/vb11_selected_vision.csv
```

F5：`VB11：完整视觉验收（选择训练后权重）`，输入框默认留空，必须输入
实际VB11检查点，防止误用历史权重。Ctrl+C仅关闭该会话。
验收要求冷启动零异常重置、真正行走、枪无持续/反复地面支撑、TCP实际高度
与跟踪精度合格，再扩展曲率/种子/更大负载。10kg和实机尚未验证。
