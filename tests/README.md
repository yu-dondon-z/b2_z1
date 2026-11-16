# 训练回归测试

现有 `test_*.py` 覆盖 AMP、曲线跟踪、课程状态、VD3–VD9 奖励与模型契约，属于可重复的回归检查，按原路径保留。它们需要的依赖不同，应按相关变更选择测试。

VD10 静态资产测试已从 `scripts/test_vd10_hollow_gun_asset.py` 移到本目录，使用带 `trimesh` 的 Python 环境：

```bash
python -m unittest discover -s tests -p 'test_vd10_hollow_gun_asset.py' -v
```

这类单元检查不启动训练，也不代表已通过完整仿真或实机验证。新回归测试放在 `tests/`；手动诊断和一次性分析放到 [scripts/diagnostics/](../scripts/diagnostics/README.md)。

VD13的10项独立检查覆盖75cm法兰偏移、平行轴惯量、旧模型保留、TCP、启动参数与同版本恢复契约：

```bash
/home/dong/anaconda3/envs/env_isaaclab/bin/python -B \
  -m unittest discover -s tests -p test_vd13_mount75.py -v
```

实际保存／恢复还做过8环境2+1轮管线检查，非收敛评估；见[训练说明](../docs/VD13_MOUNT75_TRAINING_20261115.md)。

VD14的18项CPU检查覆盖姿态奖励、坐标变换、独立配置、加载互斥、检查点保存和恢复，另有5项离线Play姿态测量测试（共23项）；实际Isaac／MuJoCo和视觉失败记录见[最终Play](../../rl_sar_b2_z1/docs/reports/vd14/VD14_PLAY_MUJOCO_20261116.md)。

```bash
/home/dong/anaconda3/envs/env_isaaclab/bin/python -B \
  -m unittest discover -s tests -p 'test_vd14*.py' -v
```
