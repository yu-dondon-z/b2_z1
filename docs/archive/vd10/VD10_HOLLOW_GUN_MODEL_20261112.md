> 历史归档：内容和命令对应原记录日期，不是当前VD13操作指令。当前状态见[文档入口](../../README.md)。

# 独立候选模型：2.93 kg 中空焊枪

> 2026-11-13 后续进度见 [枪体接地修复](VD10_GROUND_CONTACT_FIX_20261113.md)。
> 下文冒烟失败、长训前关卡与命令状态保留为当时记录，当前状态以新记录为准。

这是独立的**新枪训练候选与启动检查**，不是对步态/枪尖效果或部署的证明。
保留 `b2_z1_vd3_zero_gun.urdf`、VD1 原始枪、全部历史日志和 checkpoint；
未更改 VD5/VD9 已注册任务、导出 actor、板载程序或机器人控制。

## 实物规格与安装假设

- 枪管外径 56 mm、内径 52 mm、壁厚 2 mm、长 1100 mm、质量 2.93 kg。
- 截面均匀中空圆管模型：枪轴的中心惯量 `Ixx=0.0021389 kg·m²`，
  横向 `Iyy=Izz=0.2965111167 kg·m²`；体积约 0.000373221 m³，
  对应密度约 7850.57 kg/m³。采用用户提供的实测总质量，没有额外增加连接件质量。
- 相对末端 `link06` 的局部坐标，枪轴为 **+X**（joint6 轴向）；
  旧模型枪轴沿 −Z。原始末端法兰中心在 `link06` 的 `(+0.075,0,0) m`；
  夹环中心取 `(+0.075,0,+0.100) m`。这里“向上”解释为末端默认姿态的
  **局部 +Z**，机械臂转动后不会一直是世界 +Z。夹环的 3D 打印结构不建模也不计重。
- 用户允许合理选取前伸量：选 **夹环前方 900 mm、后方 200 mm**，使连接处位于枪管靠后的上部。
  因此枪体质心相对 `link06` 为 `(0.425,0,0.100) m`，
  枪尖 TCP 为 `(0.975,0,0.100) m`；TCP 朝向与 `link06` 一致（单位四元数）。
  前后分段不是照片量测值，可根据实际夹持位置重生成。
- 外轮廓碰撞为直径 56 mm 的实心包络圆柱，**不支持真实内孔接触**；
  可见网格是具有 52 mm 通孔的闭合环形管。Isaac 候选配置关闭了旧版本的
  `replace_cylinders_with_capsules`，MuJoCo 候选同样保留长度恰为 1100 mm 的
  圆柱，避免胶囊端帽额外延长 56 mm。真实管壁质量/惯量以中空圆管计算。
- 枪体的质量、惯量、碰撞和外观合入 `link06`，保证原本监测
  `link06` 的接触传感器能看到枪体受力；没有新增独立的固定连接刚体。

## 资源和边界

生成器：`source/b2_z1/b2_z1/assets/vd10_hollow_gun_asset.py`；新 URDF、
模型规格及 SHA256：`source/b2_z1/b2_z1/assets/b2_z1/b2_z1_vd10_hollow_gun.urdf`
与同名 `.manifest.json`；可见网格 `gun/meshes/vd10_hollow_tube.stl`。

Isaac 的可选 `B2_Z1_HOLLOW_GUN_CFG` 不替换原 `B2_Z1_GUN_CFG`。
MuJoCo 的 `build_scene(..., robot_version="vd10")` 指向同一 URDF，
`validate_vd10_model` 验证质量、惯量、TCP 与碰撞尺寸。
训练侧的 `tcp_frame.tcp_transform_for_asset()` 只对新资源选用新 TCP，
旧资源继续使用旧数值。

独立的 `Unitree-b2z1-VD10-Velocity` 训练任务和 `scripts/train_vd10.sh`
已注册；训练 runner 使用 VD10 专用 TCP 左右镜像。它不替换原 VD5/VD9
任务或部署 actor。**当前 VD10 仅完成冒烟训练，尚不具备长训和仿真部署条件**；
旧策略在新模型上的短时动作不能当成新模型的收敛或部署评估。

## 已完成检查与阻碍（2026-11-13）

1. 生成新文件时验证 VD3 源 SHA256 不变；
   `tests/test_vd10_hollow_gun_asset.py`、原版对称性和训练检查合计
   `37 passed, 6 subtests passed`（使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`）。
2. MuJoCo 分别加载、编译并通过 VD3/新模型静态校验：两者均为 18 关节，
   新模型总质量 81.9290202 kg、枪体 2.93 kg、枪体碰撞半径 28 mm、
   圆柱半长 550 mm；历史 VD3 仍为 78.9990202 kg。
3. Isaac Lab/PhysX 实际导入验证为 24 刚体、18 关节，`link06` 为
   `3.2187581 kg`，总标称质量 `81.929016 kg`，平行轴惯量及底座质量随机化
   检查通过；详见各次运行的 `vd10_runtime_asset.json`。这是**导入质量/惯量**
   检验，不是动态效果检验。
4. 使用旧 VD5 actor 热启动与完全不加载旧 actor 两种极短冒烟检查，均出现
   高频枪体接地终止；无旧 actor 的 8 轮测试中，已终止回合的枪体受力终止
   比例在初始采样之后约 94–100%，平均回合约 13 个 50 Hz 步。
   此数据只说明启动不可用，
   **不能用于判断奖励收敛**。原始事件在
   `logs/rsl_rl/b2_z1_vd10_hollow_gun_amp/2026-11-13_09-03-23_vd10_scratch_startup_probe_20261113/`。
5. 200 Hz 真实物理采样显示，初始管体离地约 9–16 cm，但随后有环境的管体
   降到地面以下，并出现超过 1 kN 的向上接触力；采样文件
   `vd10_200hz_trace.json` 同时记有单环境管高/受力及全环境最低管高/最大受力，
   **不能把不同环境的极值当成同一条轨迹**。已在同一个采样环境确认枪体
   低于地面时发生受力；但这些日志**没有**实际关节驱动力矩或约束活动明细，
   不能将接触力当作电机力矩，也不能据此断言是策略还是驱动实现为根因。
6. 从上述冒烟运行的 `model_7.pt` 做 1 轮 VD10 同版本断点恢复成功，
   optimizer、AMP、当前迭代及地面课程状态按现有代码恢复；
   **这不证明接地问题已解决，更不能直接续训这些失败的检查点**。

## 长训前的关卡

应先设计/验证新枪的无撞地启动及工况过渡：在完整 50 Hz 控制和 200 Hz 物理
接触下，检查多个随机复位、从站立和 AMP 参考重置到逐步靠近真实焊缝的全过程，
证明不持续因 `tool_support` 终止。随后才值得把 `train_vd10.sh`
用于数千轮训练，并用训练曲线和独立 MuJoCo 步态砸地/枪尖左右抖动检验结果。
目前不推荐直接启动脚本默认的 4000 轮，也不要把 VD10 冒烟检查点作为基线。

## 训练命令备忘（**启动关卡通过之后**；目前不要执行长训）

从零训练和从旧 VD5 actor 热启动都尚未通过启动检查，不能据此选定最终训练
起点。以下仅核对了脚本入口、TensorBoard 和同版本断点恢复的写法，**不是当前
已验证有效的长训推荐**。一旦明确选定起点、确认连续无撞地再执行，例如从零：

```bash
cd /home/dong/b2_z1_VD2
bash scripts/train_vd10.sh --vd10-from= --num_envs 256 \
  --max_iterations 4000 --run_name vd10_validated_long
```

另开终端实时看 TensorBoard：

```bash
/home/dong/anaconda3/envs/env_isaaclab/bin/tensorboard \
  --logdir /home/dong/b2_z1_VD2/logs/rsl_rl/b2_z1_vd10_hollow_gun_amp \
  --host 127.0.0.1 --port 6006
```

在本机打开 `http://127.0.0.1:6006`。如果训练中止，用原长训目录中**最后一个
完整保存的** `model_<N>.pt`（默认每 100 轮及正常完成时保存）；不要选
`vd10_*_probe` 的冒烟目录，也不能在修改了枪模型/训练课程后直接恢复旧任务。
完整目录名和文件名分别传给 `--load_run` 和 `--checkpoint`：

```bash
cd /home/dong/b2_z1_VD2
bash scripts/train_vd10.sh --resume \
  --load_run '<原长训目录名，例如 2026-11-13_12-00-00_vd10_validated_long>' \
  --checkpoint 'model_<最后完整保存的轮数>.pt' \
  --num_envs 256 --max_iterations 2000 --run_name vd10_validated_resume
```

这里 `--max_iterations 2000` 意味着从加载点再运行 2000 次 PPO 更新，
**不是**把总轮数设为 2000。现有 runner 恢复时会再次使用检查点所记的
迭代编号（例如从 `model_7.pt` 再跑一轮仍写作 `model_7.pt`），
因此按新生成的运行目录和实际更新数核对，旧目录不会覆盖。完整效果须在
奖励/终止情况稳定后另行验证 MuJoCo 落脚冲击与枪尖左右高频 RMS。
