# VD5 Startup 80/20 训练检查（2026-11-10）

## 实际训练设置

已完成的 run：`logs/rsl_rl/b2_z1_vd5_startup_amp/2026-11-10_10-44-52_vd5_startup_80ref_20stand_corrected_20261110/`。

环境 YAML、初始课程 JSON、最终检查点三处均确认 `reference_initialization_prob=0.8`。这是按每次重置独立抽样的概率，并非每批环境严格保持 80/20。

从原始 VD5 `model_3250.pt` 适配 250 轮，保留策略、优化器和课程状态，保存 `model_3375.pt` 与 `model_3499.pt`。最终 ground blend 为 0.84048，曲线课程等级为 0.10，后方取点距离为 0.70 m；起步渐入仍为 2 秒，普通站姿根高度为 0.50 m。

最后 10 个训练日志窗口中，time_out 指标约 92.97%，base_contact 约 6.25%，bad_orientation 约 0.78%。这些是训练日志的汇总值，不是独立评估成功率。

## 相同条件下的独立仿真

使用 VD5Startup 任务、seed=42、8 环境、1500 控制步（30 秒）、ground blend 冻结为 0.84048、相同曲线等级。禁用观测噪声和外力推搡，相机资产保留但没有图像输入。混合测试使用参考初始化概率 0.8。

每个环境只统计其**第一次终止前的首回合**；失败后的自动重置回合不算成功。32 秒旧探测包含 30 秒超时后新回合的部分样本，不能凭进程正常退出或出现 8 次 time_out 判断 8/8 成功。

| 检查点 | 初始化 | 首回合正常结束 | 早期失败 | TCP 指令点 XY 误差 P95 | 最大倾角 |
|---|---|---:|---|---:|---:|
| 原 VD5 model_3250 | 混合 80/20 | 6/8 | 2 次小腿接触 | 1.36 cm | 18.94° |
| 新 run model_3375 | 混合 80/20 | 6/8 | 2 次小腿接触 | 1.25 cm | 20.05° |
| 新 run model_3499 | 混合 80/20 | 6/8 | 1 次小腿接触、1 次枪支撑 | 1.36 cm | 21.99° |
| 新 run model_3499 | 普通站姿 | 8/8 | 0 | 1.20 cm | 3.72° |

混合起步失败集中在首回合约 0.50–0.70 秒。普通站姿本身的测试通过，参考初始化与起步命令过渡的适配仍需关注；目前尚未单独记录每次重置抽取的参考帧及其速度，不能把冲突机制当成已证实的根因。

上述 XY 误差在机身坐标系中相对于当前混合指令点计算，包含起步阶段；不是实际相机测得的焊缝中心误差，也不是 ground blend=1.0 下的最终焊缝精度。首回合统计由 JSON 中逐环境样本在第一次 `terminated=true` 处截断。

原始数据位于 `outputs/vd5_startup_20261110/`：

- `prob80_source_first_episode.json`
- `prob80_iter3375_first_episode.json`
- `prob80_iter3499_first_episode.json`
- `prob80_final_nominal_30s.json`

## 下一步

不从零重训。保留 source、3375、3499 三个对照，从本次 80/20 run 的 `model_3499.pt` 继续 1000 轮；约再 665 轮 ground blend 达到 1.0，剩余约 335 轮训练完整地面目标。曲线课程是否升级仍取决于其自身规则。

这是一段有上限的课程推进试验，不是保证能消除早期失败。完成后优先检查训练的接触/倾倒终止、完整地面目标的多种子首回合成功率，再决定是否更改参考初始化与起步过渡。若失败率未降低，停止仅靠增加轮数，先定位起步失败。

```bash
cd /home/dong/b2_z1_VD2
bash scripts/train_vd5_startup.sh \
  --resume \
  --load_run 2026-11-10_10-44-52_vd5_startup_80ref_20stand_corrected_20261110 \
  --checkpoint model_3499.pt \
  --num_envs 128 \
  --max_iterations 1000 \
  --run_name vd5_startup_80ref_20stand_continue1000 \
  agent.amp_cfg.amp_num_preload_transitions=20000 \
  agent.save_interval=250
```

`--max_iterations 1000` 在本 runner 中表示追加 1000 轮。`--resume` 保留 80/20 配置对应的优化器和课程状态，另建带时间戳的 run。此次检查未启动后续训练，也未进行板载同步。
