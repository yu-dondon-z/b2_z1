# VB10：保留退出限位，修正撑地与视觉接管

独立分支VB10；不改VB7/VB8/VB9任务的默认物理、权重或实机部署。
训练仍为70维观测、18维输出、同一个全身actor与原0.20 EMA；不加入部署IK。

## 改了什么

1. **保持TCP离地1cm**。原URDF圆柱长1.10m、半径0.028m，中心z=-0.05；
   导入胶囊时多出半球，底部约-0.628m，低于TCP的-0.620m。
   VB10只把枪的碰撞表示改为同尺寸平端凸网格（48边外接多边形，径向略
   大而不是缩小）；底部恢复到-0.600m。其他关节碰撞、枪质量/惯量、
   TCP变换不改。仍是URDF的圆柱近似，不是实机测量CAD。
   原URDF不写入修改；派生URDF/网格在/tmp按内容哈希自动生成。
2. **禁止依赖持续撑地**：原枪接触惩罚-1→-3，新增接触力代价-0.75，
   20N尺度；枪接触力连续>10N超过0.8秒终止，初始4秒容许姿态过渡。
   每个环境独立计时，重置清除。没有关闭碰撞、忽略接触或增大电机力矩。
3. **加强精细TCP跟踪**：fine-grained奖励1.25→2.5，尺度0.10→0.05m；
   旧默认机械臂姿态偏离惩罚降到-0.08，避免把清限位姿态拉回旧限位。
   训练辅助参考由0.40降到0.08；保持限位余量、枪轴与整枪稳定性奖励。
4. **视觉接管**：在尚未缓存到后方轨迹点时，先平滑接近后向工作区中心、
   1cm高度的待机TCP目标。实测位置误差<8cm、垂向误差<1.5cm、机械臂
   最大速度<0.8rad/s、机身倾角<0.1rad/高度>0.45m、枪接触力<5N，
   连续满足0.30秒后才允许起步。不伪造相机历史路径点。
   这是一次性启动门槛，行走后不会因普通TCP误差反复关门、造成走走停停；
   只有环境重置或真正失去视觉并完全停止才重新检查。
5. TCP准备仍连续推进；本体速度额外延迟1.5秒，再按原平滑时长上升。
   合成训练与文件播放一致。加入初始横偏±6cm、航向±0.06rad的恢复训练。
   旧任务新增参数默认零/关闭，不多消耗随机数。

## 验证证据与限制

- 13项VB10 CPU测试：闭合/外接网格、质量/惯量/关节/腿碰撞不变、
  独立接触计时、重置与启动宽限、速度延迟、启动门槛和一次性锁存。
- 原12项姿态参考、9项稳定性、4项路径测试通过。
- 实际Isaac导入：枪是Mesh而非Capsule，尺寸约0.05612×0.05612×1.10000m，
  枪质量3.259047kg，真实障碍物接触峰值105.985N。
  日志 `outputs/vb10_collision_check.log`。
- 仅100轮128环境小试，`2026-10-13_10-19-58_VB10_clearance_pilot`。
  无NaN/Inf；前20→后20轮平均枪接触比例86.4%→65.9%，平均竖直
  支撑力59.2→26.9N，TCP误差10.6→6.4cm。持续接触终止仍很高，
  **不是合格策略**，仅说明约束/日志有效且开始适应撤除旧地面支撑。
  不把短试权重作为默认播放/部署，不自动开始正式长训。
- 100轮小试的30s/3kg真值回放仍6次重置（5次持续枪接触、1次低机身）。
  20s视觉通路接收到有效裂缝，但实际枪接触/待机条件不合格，启动门槛
  保持本体速度命令全程0、首尾位移约2.5mm，5次持续枪接触重置。
  这是**正确拒绝不合格策略起步**，不是视觉行走能力验证成功。
  该回放CSV为`outputs/vb10_pilot_screen/vision_11098_3kg.csv`。
  原小试控制台的arm-filter重置计数显示0，以CSV真实重置列为准。
  本次同时修复该日志错误：不启用额外机械臂动作滤波也会统计实际done。

碰撞修正不等于旧权重会立刻悬空：VB9已经学会利用撑地，需要重新适应
悬空负载。正式训练后仍须回放验收，不能把训练奖励提高视为成功。

## 正式训练（新增1500轮，0–3kg等效软管）

```bash
cd /home/dong/b2_z1
bash scripts/rsl_rl/train_amp_vb10_clearance.sh
```

固定从VB9正式运行`model_10999.pt`开始，学习率1e-5，每100轮保存，
默认256环境，AMP预加载20000。F5：
`VB10：悬空TCP＋视觉接管修复（续训1500轮）`。
`VB10_ITERATIONS`是新增轮数；`VB10_NUM_ENVS`可调整并行环境数量。
每次重新启动辅助参考权重会重新做200轮渐增。

TensorBoard重点：`VB10/gun_contact_fraction`、`VB10/gun_vertical_support_mean_n`、
`VB10/gun_bottom_clearance_mean_m`、`Episode_Termination/gun_sustained_contact`、
TCP误差、课程、速度/抬腿/航向与异常终止。VB10接触日志是步级群体统计，
不是单独某个机器人完整回放的成功率。

## 检查正式训练后的指定权重

```bash
cd /home/dong/b2_z1
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/evaluate_vb1_load_sweep.py \
  --task Unitree-b2z1-AMP-VB10-Clearance-Play \
  --checkpoint /absolute/path/to/VB10/model.pt \
  --masses 1 3 --max-steps 3000 --output-dir outputs/vb10_selected_load_check
```

CSV新增实际接触力、竖直支撑力和碰撞底部离地量，不能只看枪尖起伏：

```bash
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/analyze_vb9_gun_motion.py \
  outputs/vb10_selected_load_check/3kg.csv
```

完整视觉：

```bash
B2_Z1_HOSE_MASS_KG=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/play_va15_vision.py \
  --vb10 --checkpoint /absolute/path/to/VB10/model.pt \
  --episode-length-s 600 --max-steps 3000 \
  --motion-trace-csv outputs/vb10_selected_vision.csv
```

Ctrl+C只关闭本会话。若待机门槛一直不通过，要检查实际TCP误差/接触，
不能直接绕过门槛让尚在撑地的策略行走。

验收必须包括：冷启动无异常重置、实际前进而非站着刷稳定奖励、稳态
几乎无枪体接触且无持续支撑、TCP跟踪/航向不退化、首段视觉接管偏差，
再扩展不同负载和曲率、多随机种子。阶段B/10kg/实机尚未验证。
