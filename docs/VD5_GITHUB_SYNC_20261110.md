# VD5 GitHub 版本说明（2026-11-10）

源工作区：`/home/dong/b2_z1_VD2`。目录名沿用 VD2，当前任务为 **VD5 Startup**。
GitHub 仓库：`https://github.com/yu-dondon-z/b2_z1`，分支：`VD5`。
源代码历史从 `b64b0faf8b4db3a9c1c2e05235521f61ab4dc47d` 继续。

本次保存当前训练代码、VD3/VD4/VD5 环境与启动脚本、已有 CPU 回归测试、
训练和评估说明，以及下列完整 VD5 检查点和原始训练参数。
`logs/`、`outputs/`、缓存、IDE 配置和 `.code_versions/` 本地归档继续按原规则排除。
检查点复制至原有的 `test_ckpt/` 目录结构，原始训练文件保留在原位置。

## 当前检查点

```text
test_ckpt/b2_z1_vd5_startup_amp/
  2026-11-10_11-47-57_vd5_startup_80ref_20stand_continue1000/
    model_4498.pt
    CHECKPOINT.json
    vd5_initial_curriculum.json
    params/agent.yaml
    params/agent.pkl
    params/env.yaml
    params/env.pkl
```

- 任务：`Unitree-b2z1-VD5-Startup-Velocity`。
- 实验：`b2_z1_vd5_startup_amp`。
- 完整检查点 SHA256：`9af8e2ca67382d3186bb9ea145930c25797b06e6189963bb1d996fee91f3fedf`。
- 本轮训练：从迭代 3499 续训到 4498，共追加 1000 轮。
- 检查点包含策略、优化器、AMP 判别器/归一化器及最终 VD5 课程状态。
- `vd5_initial_curriculum.json` 是该 run 的初始课程记录，最终状态以检查点为准。

文件大小约 12 MiB，以普通 Git 文件保存；目录内的 `.gitattributes` 明确禁用
该检查点的 LFS 过滤。`CHECKPOINT.json` 记录每个附带文件的 SHA256。

## 验证与使用

在已安装 Isaac Lab 的 Python 环境中运行 CPU 测试：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

同步前在 `env_isaaclab` 环境中运行，193 项全部通过。
本次同步不重新训练，也不进行 MuJoCo 或实机验证。

复查该检查点在普通站姿下的几何命令跟踪，可使用已有探测入口：

```bash
export PYTHONPATH="$PWD/source/b2_z1${PYTHONPATH:+:$PYTHONPATH}"
python scripts/rsl_rl/probe_vd5_commands.py \
  --checkpoint test_ckpt/b2_z1_vd5_startup_amp/2026-11-10_11-47-57_vd5_startup_80ref_20stand_continue1000/model_4498.pt \
  --task VD5Startup --reset-mode nominal --ground-blend 1.0 --restore-curve \
  --seam-metrics --seed 42 --steps 1500 --num_envs 8 --device cuda:0 --headless \
  --output outputs/vd5_github_recheck_seed42.json
```

该命令用于复现后续评估，需准备 Isaac Lab、Isaac Sim 和场景资产；本次同步
只运行上述 CPU 测试。启动脚本的默认 Python/资产目录属于原开发环境，换机时
通过 `VD5_PYTHON` 和 `VD5_ASSET_ROOT` 指向本机安装。
原始训练日志和逐步评估数据仍在本地 `logs/` 与 `outputs/`；本仓库保存评估报告。

最终候选与已知底盘/足端问题以
[VD5_TASK_TRACKING_REVIEW_20261110.md](VD5_TASK_TRACKING_REVIEW_20261110.md) 为准。
此前 4250/4498 的混合起步筛选记录保留在
[VD5_CONTINUE1000_REVIEW_20261110.md](VD5_CONTINUE1000_REVIEW_20261110.md)。
