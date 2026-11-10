# 统一训练工作区

只使用 `/home/dong/b2_z1_VD2`。目录名保留，2026-11-09 新版本为 VD5
`world_seam_ground_curriculum_v1`，见根目录 `CODE_VERSION.json`。
后续版本在此目录保存，不再建立并列的项目副本。

VD3、VD4 入口和历史检查点保留。VD5 使用独立任务和实验目录，从已部署的 VD3
镜像 `model_2000.pt` 微调，不使用下面迁移记录里的旧 `model_18000.pt`。
当前方案、验证结果和启动指令见 `VD5_WORLD_SEAM_COMMANDS_20261109.md`。

## 2026-11-05 完成 VD3 目录迁移

原 `/home/dong/b2_z1_VD3` 的代码、检查点和诊断结果已保存到本工作区，
旧目录及其 Git worktree 登记已删除。VD2 当前训练源码、暂存区和分支保持原样；
迁移期间运行的镜像微调进程未停止，也未切换它使用的代码。

两份完整源码树保存为本地 Git 版本，未推送：

| 版本 | 本地 Git 分支 | 用途 |
| --- | --- | --- |
| 当前训练源码 | `code-versions/vd3-support-mirror-20261105` | 零枪重、支撑保护、腿部镜像微调 |
| 原 VD3 最新源码 | `code-versions/vd3-curve-play-20261105` | TCP 使用训练时的高度和插值，XY 对齐到平缓曲线上方的 Play |

这些版本由独立临时 index 生成，包含当时的未提交修改及新增文件。
当前 `VD2` 分支仍保留原来的工作状态，不能只用旧 `VD3` 分支名恢复最新代码。
不要在训练运行期间 checkout 其他版本。

原始补丁、文件内容、权限、SHA256、原 index 另存于：

```text
.code_versions/VD3_SUPPORT_MIRROR_20261105.tar.gz
.code_versions/VD3_CURVE_PLAY_20261105.tar.gz
.code_versions/VD3_IGNORED_RESIDUAL_20261105.tar.gz
.code_versions/versions_20261105.json
```

最后一个归档保留未入 Git 且与当前工作区不同的剩余文件，包括缓存及旧
worktree 的 `.git` 指针；其中指针仅作历史记录，不能解压覆盖当前 `.git`。
11 月 4 日的两份旧源码快照也继续保留。`.code_versions/` 已被本地 Git 排除。

运行结果的相对路径保持不变：

```text
原 VD3/logs/    → VD2/logs/       （63 个文件）
原 VD3/outputs/ → VD2/outputs/    （59 个文件）
```

主检查点现在位于：

```text
/home/dong/b2_z1_VD2/logs/rsl_rl/b2_z1_vd3_curve_amp/2026-11-04_09-44-59_vd3_zero_gun_flange_stability/model_18000.pt
```

迁移清单和逐文件 SHA256 校验结果见
`outputs/storage_migration_20261105/migration.json` 及
`vd3_inventory_before.json`。旧报告、保存的 YAML/Pickle 中的历史绝对路径
保留为原始证据；手动重放旧配置时，将 `/home/dong/b2_z1_VD3` 前缀替换为
`/home/dong/b2_z1_VD2`，并选择对应代码版本。新增启动入口不依赖旧目录。

## 启动入口

继续使用当前训练版本：

```bash
bash /home/dong/b2_z1_VD2/scripts/train_vd3.sh
bash /home/dong/b2_z1_VD2/scripts/train_vd3_symmetry.sh
```

运行当前训练源码的 Play：

```bash
bash /home/dong/b2_z1_VD2/scripts/play_vd3.sh
```

运行原 VD3 的曲线 TCP Play（默认 `model_18000.pt`）：

```bash
bash /home/dong/b2_z1_VD2/scripts/play_vd3_curve.sh
```

首次运行从本地 Git 版本导出源码到 `.code_versions/runtime/`，共用本工作区的
数据集和日志；不修改训练源码。可用 `--prepare-only` 仅准备并核对路径，
用 `VD3_CHECKPOINT=/绝对路径/model_*.pt` 覆盖检查点，额外 CLI 参数直接追加。
运行导出目录可由保存的 Git 提交重新生成。

如果终端原先停留在已删除的 VD3 目录，先执行 `cd /home/dong/b2_z1_VD2`。

## 其他历史说明

11 月 4 日源码整合的原始记录保留在 `.code_versions/VD3_INTEGRATION_20261104.json`
及 `.code_versions/snapshots_20261104_095209.json`。
左右镜像训练配置及验证见 `VD3_SYMMETRY_20261105.md`。

原 `/home/dong/b2_z1` 是独立的 VD1／部署调试仓库，本次未改动。
`/home/dong/rl_sar_b2_z1/scripts/diagnose_vd1_isaac_response.py` 会导入本工作区的
VD3 零枪重配置，不能将结果当作原 VD1 有枪重模型的对照。
