# VA16 双相机训练工程交接（2026-11-02）

**同日后续更新：用户已要求替换为七轴 Huibo，并关闭相机视野线框。**
新模型、独立任务和训练入口已经另行建立；最新接管请先看
[`HANDOFF_B2_HUIBO_20261102.md`](HANDOFF_B2_HUIBO_20261102.md)。
本文件下方仍保留原 Z1 阶段记录，所述 70/18 维策略及历史跟踪效果只属于 Z1。
相机模型保留，`show_rear_camera_frustum` 当前默认 `False`。

## 用户目标与范围

工程为 `/home/dong/b2_z1`，不是实机部署工程 `rl_sar_b2_z1`。
用户确定工作段 B2 前进指令固定 `vx=0.20 m/s, vy=0`，前视光学 RGB
纠偏并给出 `w`；机身后部 D435i 向后下方观察地面与焊缝枪，用真实
RGB-D 测量 TCP 和焊缝并闭环。验收误差是 TCP 地面投影到裂缝中心，
目标不超过 5 cm；不是三维 TCP 距离或只看命令跟踪误差。

本次只修改训练工程和本地 Isaac 仿真，没有同步板载、启动 DDS 或控制实机。
保留原 VA15 任务、模型、URDF 和既有日志。进入工作区时已有的八个训练
脚本删除状态及 `.codex_stage/` 等无关文件未处理。

## 当前实现

- 新任务 `Unitree-b2z1-AMP-VA16-DualVision`：无 RTX 相机的固定速度代理续训。
- 新任务 `Unitree-b2z1-AMP-VA16-DualVision-Play`：单环境真实渲染双相机闭环。
- 可选 `Unitree-b2z1-AMP-VA16-DualVision-Visual-Train`：单环境外接视觉桥接，
  尚无已验收的一键图像训练入口。
- 当前闭环复用 VA15 `model_5000.pt`，维持 70 维观测、18 维动作。
  视觉在策略外部生成速度和末端目标，不能声称原 actor 已学会新视觉输入。
- 前相机只导出 RGB；后相机导出 RGB 与对齐光轴深度。后视拟合实测地面平面，
  前视 RGB 射线与该平面求交，无前视深度输入。
- 后相机按用户要求改为尾壳中央安装，光心相对 `base_link` 为
  `(-0.445, 0, 0.005) m`，yaw 180°、俯视 45°，参考自带 `r_dc_link`
  稍外置并沿外壳向后下方观察。没有为相机可见性改变枪体或标记位置。
  旧 `(-0.48, 0.30, 0.34) m`、yaw 225°/俯视 55° 为上一版侧置方案；
  旧高位中央安装的遮挡诊断不能直接外推为本次低位尾壳中央不可见。
- 场景新增银壳、黑面板、三镜头的 D435i 示意模型，无新增刚体质量或碰撞。
  青蓝视锥按真实相机内参绘制，仅主视口可见，不进入 RGB-D 传感器图像；
  显示范围 0.10–1.50 m，实际深度裁剪仍为 0.10–6.00 m。
  主视口为机器人后方斜视跟随，便于检查相机壳体和视野。
- 枪体沿用独立场景的接触圆柱，不改原 no_stick URDF 的 1 g link_gun 质量。
  青色/品红色球是无碰撞、无附加刚体质量的视觉标记，半径 35 mm，轴向间距
  100 mm，TCP 在青球沿远离品红球方向再延伸 100 mm。由实际深度表面拟合球心，
  不从仿真 TCP 或关节 FK 生成视觉测量。
- 闭环以实测 TCP 生成有界目标修正；关节编码器 FK 只传播采集后运动增量。
  不调整执行器增益，不伪造接触或帧时效。工作段固定 0.20，准备、起步过渡、
  观测无效减速停车另行处理。
- 单独准备阶段先在零行走指令下将工具送入后相机视野；看到真实 TCP 后才接入跟踪。
- 采集帧按 sequence、reset generation、原始 monotonic 采集时间核对。
  环形图像读取前后核对元数据，旧图重新发布不能刷新帧龄。
- 物理裂缝真值只用于足部奖励和独立评估，不传给视觉目标控制。

主要文件：

| 文件 | 用途 |
|---|---|
| `source/b2_z1/b2_z1/tasks/manager_based/b2_z1/b2_z1_va16_dual_vision_env_cfg.py` | 新场景、相机、任务配置 |
| `.../mdp/commands/dual_vision_command.py` | 双视觉控制、准备/跟踪、采集、状态 |
| `.../mdp/commands/dual_vision_contract.py` | 独立时间协议与后视目标几何 |
| `.../mdp/commands/dual_vision_kinematics.py` | URDF 编码器 TCP 正运动学 |
| `.../mdp/commands/dual_vision_rewards.py` | 物理 mesh 中心线距离，仅奖励/评估 |
| `scripts/vision/dual_camera_perception.py` | RHANet、地面 ROI、RGB-D 标记/TCP |
| `scripts/vision/run_dual_camera_bridge.py` | 两相机配对与感知文件协议 |
| `scripts/vision/play_va16_dual_vision.py` | 独立启动、窗口、日志、清理本次子进程 |
| `scripts/vision/evaluate_va16_dual_run.py` | 50 Hz 真值误差、机身和足部诊断 |
| `scripts/rsl_rl/train_amp_va16_dual_vision.sh` | VA15 固定速度代理续训 |
| `scripts/vision/VA16_DUAL_VISION.md` | 用户运行与协议文档 |

既有 `scripts/rsl_rl/play_amp.py` 仅增加可选导出目录，以及 VA16 专用的
50 Hz active、实际速度、四足位置/接触力日志；旧任务默认行为保留。

## 已做验证与失败留存

当前中央 45° 安装首次静态检查：
`outputs/va16_rear_center45_20261102_01/scene_report.json`。
后视感知有效，RGB-D TCP 对独立真值误差 2.378 mm；枪体碰撞和原质量保留。
但初始机身高度约 0.70 m 时，原工作域近端 `x=-0.55 m` 的三个采样点
落在图像下边界以外，完整旧工作域检查为 `pass=false`，不得改写为通过。
该静态结果不等于全工作域或动态跟踪达标，动态验证另用实际记录评估。

本轮中央安装静态 GUI 检查：
`outputs/va16_rear_center45_20261102_03/scene_report.json`，真实主视口截图为
`outputs/va16_rear_center45_20261102_03/viewport_camera_frustum.png`。
报告读取实际光轴俯角 44.9999925°，确认壳体没有附加刚体、质量或有效碰撞，
视锥可用；当前 TCP 测量误差 2.3765 mm。
`pass=true` 仅对应安装及当前可见 TCP，`full_workspace_visible=false` 仍保留。
03/04 的旧 `fast_shutdown=False` 检查器在插件完整卸载阶段返回 139，
显式 `overlay.close()` 后仍可复现，不能改写历史退出状态或声称底层问题已定位。
检查器最终采用与默认播放相同的标准 Kit shutdown 模式，不覆盖异常或退出码。
最终 GUI 检查 `outputs/va16_rear_center45_20261102_05/` 报告和截图均生成、
进程退出 0；安装与当前可见 TCP 检查通过，full_workspace_visible 仍为 false。
显示对象同时修复了 inference_mode 下提前分配相机缓存的问题，以及关闭时释放
相机/私有 UI frame、幂等清理与关闭后不重建的资源生命周期问题。

当前中央 45° 安装的独立动态与默认 GUI 验证均实际完成：

| 运行目录（位于 `outputs/va16_dual_vision/`） | 结果 |
|---|---|
| `run_20261102_101653_bp0ynucm` | 20 s 真实 RHANet/VA15 闭环，进程退出 0；945/1000 个 active 样本，独立 TCP 投影误差平均 7.99095 mm、P95 17.8314 mm、最大 30.6489 mm，`pass=true` |
| `run_20261102_101959_nqq0vksk` | 默认 GUI launcher 有界 150 步，进程退出 0；模型、主视口视锥和三联视觉预览正常，启动/结束源码哈希无变更；仅为 GUI/退出链验证 |

本次新中央安装的 20 s 动态满足所记录平地场景的 5 cm 条件，
不表示原完整工作域均可见，也未覆盖全部光照、遮挡、长程曲线或实机。

上一版侧置安装的静态场景：
`outputs/va16_dual_scene_20261102_07/scene_report.json` 通过。
有 70/18 接口、枪体物理碰撞、两标记可见性等实际场景检查。
该姿态下 RGB-D TCP 对独立 PhysX 真值误差 2.706 mm，不能外推全部姿态或实机精度。

下面四次动态记录均位于 `outputs/va16_dual_vision/`，原图与失败日志保留。
它们全部使用旧侧置相机 `(-0.48, 0.30, 0.34) m`，只证明该历史安装的结果，
不能声称本次中央 45° 安装已达到同样的 5 cm 动态精度：

| 运行目录 | 结果 |
|---|---|
| `run_20261102_094258_6jcnctvq` | 失败：前视 RHANet 图像边缘假线导致无可用前向切线，未真正跟踪 |
| `run_20261102_094557__rookw_h` | 失败：后视将枪体轮廓当焊缝，真实 TCP 最大偏差约 41.6 cm |
| `run_20261102_095233_07sq83iw` | 修复后 20 s 动态通过本次平地 5 cm 条件；940 个 active 样本，平均 8.99 mm，P95 21.72 mm，最大 31.14 mm |
| `run_20261102_095512_iobalqyn` | 40 s 弯曲裂缝测试通过本次条件；1912/2000 个 active 样本，平均 9.57 mm，P95 23.51 mm，最大 40.22 mm |

关键修复不是盲调增益：前视在中心线提取前限制实测平面前方 3 m 以及图像边界；
后视在 RHANet 中心线提取前使用实测深度/平面，剔除地面上方超过 15 mm 的枪体与
腿部，前景膨胀 4 px；保留地面以下最多 100 mm 的真实凹陷，并限制后续取深度也
只能来自该地面 ROI。run2 离线对照保存在 `rear_ground_roi_fix_20261102/`：
最新帧修复后识别出 41.75 cm 偏差，与独立物理裂缝误差约 40.88 cm 接近。
这解释为何不能把 `RHANet valid=true` 当作正确跟踪。

run3 四足支撑时脚中心最小离线距离 FL/FR/RL/RR 为
26.16/10.92/22.42/18.02 cm；后脚抬脚阈值事件 RL/RR 为 31/32，
接触比约 80.6%/79.8%。后脚摆动中心最高 5.71/6.28 cm，相对驻地中心粗估仅
抬高约 2.5/3.1 cm，抬腿仍偏低。力阈值事件可能含抖动，不能等同完整步态周期；
脚中心距线不能代替足底边缘净距或踩线碰撞验收。

run3 控制/感知源在运行期间未修改；provenance 显示 launcher 与评估脚本有变更，
不隐去此记录。run4 已完成并生成独立 `run_report.json`；以上均是实际 RHANet
而非 classical 模式。run4 启动/结束源哈希无变更、无重置。验收仅覆盖本次局部
平地曲线、视角和光照。

run4 足中心支撑时最小距线 FL/FR/RL/RR 为 7.17/9.87/11.26/17.35 cm，
左前脚余量比 run3 小；未验证足底边缘净距。后脚抬腿仍约 3 cm。
稳定段（active 且命令速度 ≥0.198）实际前进速度均值约 0.189 m/s。
实际双相机感知吞吐为墙钟 4.41 对/秒，处理耗时均值 143 ms、P95 201 ms，
发布时帧龄均值 180 ms、P95 246 ms，控制器读取帧龄 P95 431 ms。
Isaac 在本机慢于实时，配置的 25 Hz 不是实机已经达到的处理频率。

CPU 回归共 54 项通过：感知 13、桥接 3、控制/几何/FK/方法/奖励 31、
评估 7；覆盖高处枪轮廓竞争、真实帧龄、重置隔离、测量反馈、启动准备和
50 Hz 峰值误差，未把测试通过当作真实渲染验收。

短程续训已实际完成：
`logs/rsl_rl/b2_z1_amp/2026-11-02_09-51-54_va16_fixed020_smoke/`，
128 环境、追加 10 轮、30720 步，生成 `model_5009.pt`。
它只验证加载/训练/保存链路，不是已比较优于 VA15 的正式策略。
只读核对 checkpoint 的 92 个 tensor、3,101,705 个元素全部有限，iter=5009，
actor 输入 70、输出 18；TensorBoard 85 个 scalar tag 无 NaN/Inf。
保存的 env.yaml 无两相机、standing mask 为零、四个课程速度范围均固定 0.20。

## 启动命令

可视化双相机闭环（不加 headless）：

```bash
cd /home/dong/b2_z1
/home/dong/anaconda3/envs/env_isaaclab/bin/python scripts/vision/play_va16_dual_vision.py
```

默认打开 Isaac 窗口与前 RGB、后 RGB/TCP、后深度三联图；默认实际加载既有
RHANet_dep 权重，无神经网络失败后自动退化成颜色阈值的路径。

固定速度代理续训（默认 1024 环境、追加 2000 轮，尚未长训）：

```bash
bash scripts/rsl_rl/train_amp_va16_dual_vision.sh
```

独立评估某次完整运行：

```bash
/home/dong/anaconda3/envs/crack_vision/bin/python scripts/vision/evaluate_va16_dual_run.py \
  outputs/va16_dual_vision/run_具体目录
```

## 尚未完成的边界与下一步

1. 中央 45° 安装已完成本轮 20 s 动态和默认 GUI launcher 检查；
   专用静态 GUI 检查器的退出修复仍需确认最终独立运行结果。
   全工作域近端出图限制仍在，继续检查遮挡、照明、完整曲线和起停恢复；
   上一版侧置安装的 20 s/40 s 结果不能代替本轮或完整场景验收，
   按 50 Hz 真值误差、丢帧、四脚离线距离和后腿抬腿诊断，不能只看低频 status。
2. 高并行 synthetic 代理阶段验证固定速度步态，之后需要真实双视觉噪声/延迟/
   遮挡课程；可选单环境 visual task 不代表已完成双视觉策略训练。
   代理阶段是平路与虚拟直线/圆弧，未体验真实裂缝槽接触；初始课程主要是
   直线及 24–36 m 缓弯，而当前相机场景最小曲率半径约 8.4 m。
   synthetic 末端目标按曲线弧长后退约 0.775 m；file 按机身 x=-0.775 m
   插值并加入实测反馈，不能把两种目标分布说成完全相同。
3. 深度可去除高处枪/腿，但同地面的影子不能靠该 ROI 排除，仍依赖 RHANet 泛化。
   前视借用后视地面单平面，暂不覆盖台阶、坡折等非共面地形。
4. 当前原始感知的 `tcp_error_m` 为平面法向投影，正式控制/评估沿 gravity/XY；
   本轮平地近似一致，坡地应统一误差口径后再验收。
5. 需独立测量脚底净距、完整抬腿周期和更长距离；本轮不宣称足端全部达标。
6. 实机 D435i 需要实体标记、对齐深度、相机内外参与标记到 TCP 的安装标定；
   当前理想深度与初始虚拟相机参数不能当实机标定结果。真实执行器验证和 MuJoCo/
   实机部署迁移尚未实施。同步或启动实机仍按用户后续明确指令执行。
