# VD14 GitHub源码与模型快照

2026-11-16。训练仓库：[yu-dondon-z/b2_z1，VD14分支](https://github.com/yu-dondon-z/b2_z1/tree/VD14)。
配套部署仓库：[rl_sar_b2_z1_v1，VD14分支](https://github.com/yu-dondon-z/rl_sar_b2_z1_v1/tree/VD14)。
两个仓库必须配套使用；目录名 `b2_z1_VD2` 不代表当前策略为VD2。

## 保存内容

- 当前训练源码、模型资产、奖励与PPO/AMP实现、训练/Play脚本、测试和文档。
- VD14最终完整检查点及原始 `params/`，位于
  [`test_ckpt/b2_z1_vd14_posture_amp/2026-11-16_10-41-45_vd14_posture_long_20261116/`](../test_ckpt/b2_z1_vd14_posture_amp/2026-11-16_10-41-45_vd14_posture_long_20261116/)。
  `CHECKPOINT.json` 保存文件大小与SHA256；模型约12MB，普通Git文件，不需要LFS下载。
- `model_5999.pt` 为6000轮结束时保存，包含actor/critic、优化器、AMP判别器/归一化器及最终课程。
  原始模型SHA256：`49e41e91c9e82e178a7e8ea9f2f835a6b8e440a131ebaeae3c2911f68f5d3372`。

日志、视频、临时资产、IDE设置、构建产物和中间检查点不上传，原文件全部保留。
`CODE_VERSION.json` 仍为VD5历史发布记录，不用于表示当前候选。
既有VD5分支、上游仓库和部署默认actor不变。

## Play及继续训练

以下命令使用当前开发电脑路径与已有Isaac Lab环境；换机需准备Isaac Lab 2.2.1、
Isaac Sim 5.0及对应资产路径，并安装本仓库 `source/b2_z1`。不包含Isaac安装包。

```bash
cd /home/dong/b2_z1_VD2
bash scripts/play_vd14.sh --viewer --real-time \
  --checkpoint test_ckpt/b2_z1_vd14_posture_amp/2026-11-16_10-41-45_vd14_posture_long_20261116/model_5999.pt
```

如后续确需同配置续训，可直接从附带检查点恢复，不依赖未上传的 `logs/`：

```bash
cd /home/dong/b2_z1_VD2
mkdir -p logs/rsl_rl/b2_z1_vd14_posture_amp
cp -an test_ckpt/b2_z1_vd14_posture_amp/2026-11-16_10-41-45_vd14_posture_long_20261116 \
  logs/rsl_rl/b2_z1_vd14_posture_amp/
bash scripts/train_vd14.sh --resume \
  --load_run '2026-11-16_10-41-45_vd14_posture_long_20261116' \
  --checkpoint model_5999.pt --run_name vd14_posture_resume --max_iterations 3000
```

`cp -an`仅补齐缺失文件，不覆盖本机已有训练记录；`--load_run`接收目录名匹配式而非绝对路径。
这是再训练3000轮的示例，不表示本次上传启动了训练，也不保证逐位续接所有环境/随机状态。
训练与TensorBoard细节见 [VD14训练说明](VD14_POSTURE_TRAINING_20261116.md)。
PyTorch检查点包含Python对象，仅加载可信来源。

## 当前结论

当前为本地仿真候选，**不是实机验收发布**。姿态回放改善，视觉接口已修正，
后视支持不足现在零行走等待并可恢复；60秒样本完成但仍包含估计/等待，
不能写成全程视觉有效、完全收敛或砸地/抖动全部解决。
部署入口、依赖和已知限制见配套仓库
[同步说明](https://github.com/yu-dondon-z/rl_sar_b2_z1_v1/blob/VD14/docs/VD14_GITHUB_SYNC_20261116.md)。
