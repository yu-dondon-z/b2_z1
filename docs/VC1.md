# VC1 — B2 + Huibo 双相机训练版本

保存日期：2026-11-02（Asia/Shanghai）。版本分支和标签均为 `VC1`，
基于本地 `VB13` 的 `7578a21` 保存当前程序状态。
GitHub 目标：`yu-dondon-z/b2_z1`。本次未合入上游 `dual_cmd`。

## 当前功能

- 工作时本体前进速度指令为 0.20 m/s，横向为 0；前相机 RGB 用于航向纠偏。
- 后部中间 D435i 朝后斜下 45°，Play 使用实际渲染的 RGB-D 测量工具和地面路径；
  保留相机外壳，关闭视锥显示。训练默认使用高吞吐合成路径。
- Huibo 七轴替换 Z1，独立模型 `b2_huibo.urdf`，提供模型生成、预览和逐轴诊断入口。
- 法兰和枪轴同向同轴朝下，默认后倾约 13.19°，枪尖保持 B2 坐标
  `(-0.775,0,-0.48) m`。Huibo 枪体长 0.60 m，另有 95 mm 转接段，均有碰撞。
  原厂关节轴与位置限位保持；源 ZIP、解压描述及旧 Z1 模型保留。
- Huibo actor 73 维观测、19 维动作，critic 105 维，B2 腿部 AMP 43 维。
  七轴策略从头训练，不能直接使用六轴 Z1 的 VA15/VA16 权重。

## 验证范围

本次发布前本地 CPU 回归共 69 项通过：正向持枪 9 项、双相机感知/控制/
七轴运动学等 53 项、跟踪评估 7 项。
当前姿态的实际 Isaac 渲染报告为
`outputs/b2_huibo_axial_grip_20261102_05/scene_report.json`：`pass=true`，
枪碰撞启用、相机视锥关闭、前后感知 valid，静态 RGB-D 枪尖误差约 1.998 mm。
这些报告和截图保留在本地，不随本次源代码发布上传。

此前已有固定基座、零重力的七轴响应检查和少量训练烟测，具体姿态及条件见
[交接文档](HANDOFF_B2_HUIBO_20261102.md)。零重力模型预览、CPU 测试和烟测
不代表带重力行走或焊缝跟踪性能已合格；尚无可用的 Huibo 行走跟踪策略。
用户要求的精度为 TCP 地面投影到裂缝中心误差 5 cm 以内，仍待动态验证。

## 使用入口

在项目目录运行，Python 使用已安装本项目和 Isaac Lab 的环境：

```bash
python scripts/preview_b2_huibo.py
python scripts/assets/test_huibo_upright.py
bash scripts/rsl_rl/train_amp_b2_huibo_dual_vision.sh
```

本次系统盘空间不足时可单独把预览 USD 转换缓存放到内存盘：

```bash
python scripts/preview_b2_huibo.py --usd-cache-dir /dev/shm/b2_huibo_preview_usd
```

完整配置和后续工作见 [模型与训练说明](../scripts/vision/HUIBO_INTEGRATION.md)
及 [双相机说明](../scripts/vision/VA16_DUAL_VISION.md)。
日志、策略权重、数据备份、构建和代理缓存未纳入本次新增内容，原文件不删除。
本次仅保存代码版本，没有同步板载、启动硬件控制或机器人运动。
