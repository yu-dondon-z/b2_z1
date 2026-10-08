# VA15 实体焊枪和软管等效载荷

此版本在 `b2_z1` 的 VA15 视觉播放任务上启用 `b2_z1_gun.urdf`。焊枪本体约 3.259 kg，包含惯量、外观以及 1.10 m 长、半径 0.028 m 的碰撞几何（Isaac Sim 导入为胶囊体）。它替代无枪机器人，不叠加第二个焊枪。原来的 VA15 播放任务保持不变。

软管尚无实测质量和安装路径，当前**不是**软体仿真。上端连接点在焊枪局部坐标 `(0, -0.03, 0.48)` m；货车顶部固定点默认位于环境原点 `(0, -2.0, 2.8)` m。连接点承受等效悬垂段重量 `m g`、加速度反作用 `-m a`、速度阻尼，以及软管超过松弛长度后的单向拉力；力矩按 `τ = (连接点 − 焊枪质心) × F` 施加。动力项滤波、加速度及合力限制用于避免数值突变。此模型不模拟软管自身碰撞、弯曲、拖地或货车运动。

默认有效质量 3 kg，接受 1–10 kg。进入 `/home/dong/b2_z1` 后运行：

```bash
conda activate env_isaaclab
B2_Z1_HOSE_MASS_KG=3 python scripts/vision/play_va15_vision.py --loaded-gun
```

可按试验布置设置货车锚点和软管松弛长度，例如：

```bash
B2_Z1_HOSE_MASS_KG=5 \
B2_Z1_HOSE_TRUCK_ANCHOR_XYZ_M='0,-2,2.8' \
B2_Z1_HOSE_SLACK_M=4 \
python scripts/vision/play_va15_vision.py --loaded-gun
```

VS Code 中可选“VA15视觉：实体焊枪+软管负载（model_5000）”。默认 checkpoint 是 VA15 `model_5000.pt`；`--checkpoint` 可替换。对照播放不带 `--loaded-gun` 的原任务。

无窗口物理核对：

```bash
B2_Z1_HOSE_MASS_KG=1 python scripts/vision/check_va15_loaded_gun.py --headless
B2_Z1_HOSE_MASS_KG=10 python scripts/vision/check_va15_loaded_gun.py --headless
```

核对脚本读取 PhysX 质量、实际碰撞体、静态软管力与力矩，并通过焊枪与测试球接触验证求解器力。**这不代表未经训练的策略可在整个 1–10 kg 范围内稳定行走**。当前 VA15 策略是在无此软管负载的条件下训练的；高负载或货车锚点改变后，应先仿真分级测试，再做负载随机化续训，最后才考虑实机。
