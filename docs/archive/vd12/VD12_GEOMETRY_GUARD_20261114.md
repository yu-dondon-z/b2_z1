> 历史归档：内容和命令对应原记录日期，不是当前VD13操作指令。当前状态见[文档入口](../../README.md)。

# VD12：整枪避碰与固定长训

日期：2026-11-14。正式6000轮训练已于17:22结束，最终为`model_5999.pt`。已完成最终Isaac名义及mixed Play，并提供MuJoCo隔离仿真入口，见[最终模型查看命令与验证](../../../../rl_sar_b2_z1/docs/reports/vd12/VD12_FINAL_PLAY_MUJOCO_20261114.md)。**未通过完整任务／实机验收，默认部署actor未变，不代表抖动／砸地彻底解决。** 历史[2000轮阶段评估](../../../../rl_sar_b2_z1/docs/reports/vd12/VD12_2000_PLAY_REVIEW_20261114.md)保留。

VD12保留2.93 kg新枪、驱动、70维策略输入和VD11运动质量奖励；新增整枪对机器人已有碰撞体的保守净隙惩罚与终止。200 Hz采样并锁存，50 Hz原回合边界终止，不伪造自碰撞力。两条原工作目标下放路径及4769个AMP原始复位姿态均未被新禁区误拒。

实现、验证结果与原始日志见[完整记录](../../../../rl_sar_b2_z1/docs/reports/vd12/VD12_GEOMETRY_GUARD_20261114.md)。VD11原检查点与默认部署actor均未替换。

## 正式训练

正式run为`2026-11-14_15-07-38_vd12_geometry_guard_long_20261114`，6000轮已结束；下面保留启动命令供复现，**不要为Play重复启动训练**。

```bash
cd /home/dong/b2_z1_VD2
bash scripts/train_vd12.sh \
  --num_envs 256 \
  --max_iterations 6000 \
  --run_name vd12_geometry_guard_long_20261114 \
  --logger tensorboard
```

默认从VD11正式新枪训练的`2026-11-13_17-56-49_vd11_motion_quality_long_20261113/model_5999.pt`迁移策略／critic和AMP；**重置优化器、轮数与高位→地面课程**，从头适应新约束。不要加`--resume`，不要从两轮smoke检查续训；`--vd12-from=`意味着完全随机初始化，不是此次建议。

日志写入持久磁盘`logs/rsl_rl/b2_z1_vd12_geometry_guard_amp/<时间>_vd12_geometry_guard_long_20261114/`。每100轮和正常结束时保存，保留全部检查点；新训练6000轮的最终文件为`model_5999.pt`。6000是计划，不保证收敛；固定配置充分训练后再评效果。

## TensorBoard（另开终端）

```bash
/home/dong/anaconda3/envs/env_isaaclab/bin/tensorboard \
  --logdir /home/dong/b2_z1_VD2/logs/rsl_rl/b2_z1_vd12_geometry_guard_amp \
  --host 127.0.0.1 --port 6006
```

浏览器打开`http://127.0.0.1:6006`，选正式long run，不要把smoke／probe当正式训练。关注回报、回合长度、`Episode_Termination/vd12_gun_exclusion`与两项VD12奖励；奖励曲线稳定不代替无穿模、抖动及落脚验收。

## 中止后续训

从启动日志复制**实际正式训练目录的完整名称**，替换下面`替换为实际正式训练目录名`；不要使用`.*`，否则可能选到smoke或失败的空目录。

```bash
cd /home/dong/b2_z1_VD2
bash scripts/train_vd12.sh \
  --resume \
  --load_run '^替换为实际正式训练目录名$' \
  --checkpoint '^model_[0-9]+\.pt$' \
  --num_envs 256 \
  --max_iterations 4000 \
  --run_name vd12_geometry_guard_long_20261114_resume \
  --logger tensorboard
```

`--max_iterations 4000`表示**本次再训练4000轮**，不是总轮数目标，应按需要调整。runner沿用加载的迭代标签，因此会重复一个标签；例如从`model_2000.pt`追加4000轮结束在标签5999。

续训会读取指定目录中最新编号的完整检查点，严格核对VD12几何profile并恢复策略、AMP、优化器、轮数和课程；新run仍单独保存，不覆盖旧目录。若再中止，应指向最新的续训目录，而不是反复回原目录。Ctrl+C或崩溃不保证额外保存，从最近一次完整保存恢复，可能损失不足100轮进度。物理回合会重新复位，不是逐位恢复上一个仿真瞬间。

几何profile或源码SHA不匹配会明确拒绝，不要删字段绕过，也不要把跨版本迁移当同版续训。首次中止上述正式run后，目录名填写`2026-11-14_15-07-38_vd12_geometry_guard_long_20261114`；若后来已有续训run，则填写最新续训目录。

## 2000轮Play

阶段结果及真实Isaac GUI命令见[2000轮阶段评估](../../../../rl_sar_b2_z1/docs/reports/vd12/VD12_2000_PLAY_REVIEW_20261114.md)。使用独立`bash scripts/play_vd12_checked.sh`，恢复VD12检查点实际课程和几何profile，不再误用VD11入口。当前RTX视频与原训练并开曾显存不足；完整结果来自无渲染的真实Isaac物理回放，不能把离线重建冒充实时Isaac画面。训练与默认部署actor均未更改。
