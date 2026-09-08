# SPNet 消融实验运行说明

## 实验目的与固定条件

本实验比较 naive incremental 2DGS 在开启和关闭 SPNet 时的表现，用于判断
SPNet 的候选点补充是否值得其计算开销。

脚本对每条序列自动运行一对实验：

- `with_spnet`：`enable_spnet=true`；
- `without_spnet`：`enable_spnet=false`，仅使用 LiDAR candidate positions。

两组共同固定：

- `use_Gaussian_regress=false`；
- `generate_dataset=false`；
- DA3 保持开启，仅提供 normal priors；
- 随机种子默认为 `20260908`；
- 使用相同 bag、Gaussian-LIC 配置、Coco-LIC 配置、帧划分和优化预算。

本轮不记录几何代理指标，也不记录全图 alpha coverage。评测只包含总体渲染、
LiDAR 盲区、SPNet 候选贡献、运行时间和峰值显存。

## 前置检查与编译

实验必须在可检测到 NVIDIA GPU 的实例上执行：

```bash
nvidia-smi -L
```

如果没有输出 GPU，关闭当前无 GPU 实例并切换到 GPU 模式。启动脚本也会执行
同样的检查，并在无 GPU 时以非零状态退出。

从 catkin workspace 编译修改后的包：

```bash
cd /root/catkin_gaussian
catkin build gaussian_lic --no-deps
source devel/setup.bash
```

脚本还会自动检查以下输入：

- `/root/catkin_gaussian/devel/setup.bash`；
- `/root/catkin_coco/devel/setup.bash`；
- bag、Gaussian-LIC YAML 和 Coco-LIC YAML；
- YAML 中声明的 DA3 权重；
- `with_spnet` 运行所需的 SPNet 权重。

## 执行命令

以下命令均从 Git 仓库根目录执行：

```bash
cd /root/catkin_gaussian/src/Gaussian-LIC
```

### 先运行一条序列

建议先完整跑通论文主要消融序列：

```bash
./bash_scripts/run_spnet_ablation.sh \
  --run-name spnet_smoke_hku00 \
  --sequence hku_campus_seq_00
```

这条命令会依次完成 `with_spnet` 和 `without_spnet`，不是截断数据的短
smoke test。确认两个目录均存在完成标记和逐帧指标后，再运行三序列实验。

### 运行默认三序列实验

```bash
./bash_scripts/run_spnet_ablation.sh \
  --run-name spnet_fast_20260908
```

默认序列为：

- `hku_campus_seq_00`；
- `degenerate_seq_00`；
- `tuhh_day_04`。

每条序列运行两个 condition，共六次完整运行。默认单次运行超时为 10800 秒，
默认输出根目录为 `/root/autodl-tmp/experiments/spnet_ablation`。

### 覆盖种子、超时或输出位置

```bash
./bash_scripts/run_spnet_ablation.sh \
  --run-name spnet_fast_seed_20260909 \
  --seed 20260909 \
  --timeout-seconds 14400 \
  --output-base /root/autodl-tmp/experiments/spnet_ablation
```

也可以使用同名环境变量：`RUN_NAME`、`SEED`、`RUN_TIMEOUT_SECONDS`、
`OUTPUT_BASE` 和 `COCO_WS`。

### 为单条序列覆盖输入路径

路径覆盖只允许和一个 `--sequence` 一起使用。Gaussian-LIC 配置必须保持
package-relative，因为 launch 文件会在前面添加 `$(find gaussian_lic)`。

```bash
./bash_scripts/run_spnet_ablation.sh \
  --run-name spnet_custom_hku00 \
  --sequence hku_campus_seq_00 \
  --bag-path /root/autodl-tmp/dataset/hku_campus_seq_00.bag \
  --gaussian-config config/r3live.yaml \
  --gaussian-launch r3live.launch \
  --coco-config /config/ct_odometry_r3live.yaml
```

对应的环境变量为 `BAG_PATH_OVERRIDE`、`GAUSSIAN_CONFIG_OVERRIDE`、
`GAUSSIAN_LAUNCH_OVERRIDE` 和 `COCO_CONFIG_OVERRIDE`。

### 监看运行状态

以单序列命令为例：

```bash
tail -F /root/autodl-tmp/experiments/spnet_ablation/spnet_smoke_hku00/hku_campus_seq_00/with_spnet/gaussian_lic_log.txt
```

完成的运行必须在 `gaussian_lic_log.txt` 中出现：

```text
Gaussian-LIC Done!
```

Coco-LIC 播放完 bag 后可以先于 Gaussian-LIC 正常退出；脚本不会因此终止，
而会继续等待 `Gaussian-LIC Done!`。完成标志写入后，Gaussian-LIC 节点会主动
关闭 ROS；脚本等待其自然退出并检查退出状态。若节点在 30 秒内没有退出，脚本才
清理其进程组。Gaussian-LIC 进程退出但没有该标志的运行应视为失败。按 `Ctrl-C`
中断脚本时，脚本只终止自己启动的 Gaussian-LIC、Coco-LIC 和显存采样进程，
不执行全局 `rosnode kill -a`。

## 输出目录

一次三序列实验的目录结构为：

```text
/root/autodl-tmp/experiments/spnet_ablation/<run-name>/
├── hku_campus_seq_00/
│   ├── with_spnet/
│   └── without_spnet/
├── degenerate_seq_00/
│   ├── with_spnet/
│   └── without_spnet/
├── tuhh_day_04/
│   ├── with_spnet/
│   └── without_spnet/
└── analysis/
```

每个 condition 目录包含：

```text
<condition>/
├── manifest.yaml
├── command.txt
├── gaussian_config.yaml
├── coco_config.yaml
├── gaussian_lic_log.txt
├── roslaunch_gaussian.log
├── roslaunch_coco.log
├── gpu_memory.csv
├── point_cloud.ply
├── render/
├── gt/
├── blind_mask/
├── alpha/
├── visual_quality/
│   ├── train/frame_metrics.csv
│   └── test/frame_metrics.csv
└── ablation/
    ├── candidate_flow.csv
    └── runtime_summary.csv
```

`manifest.yaml` 保存序列、condition、实际开关、种子、bag、配置、Git commit
和 dirty 状态。`command.txt` 保存实际 Gaussian-LIC launch 命令。原始实验产物
全部位于 `/root/autodl-tmp`，不会写入 Git 仓库。

## 逐帧与运行级 CSV

### `visual_quality/test/frame_metrics.csv`

每张 test 图像一行：

```text
image_name,psnr,ssim,lpips,blind_ratio,blind_psnr,blind_alpha_coverage
```

- `image_name`：同时用于 A/B 配对并定位 `render/`、`gt/`、`blind_mask/`
  和 `alpha/` 下的同名图像；
- `psnr`、`ssim`、`lpips`：整张 test 图像的渲染指标；
- `blind_ratio`：当前图像中 LiDAR 盲区的像素比例；
- `blind_psnr`：只在 LiDAR 盲区计算的 PSNR；
- `blind_alpha_coverage`：盲区中渲染 alpha 高于统一阈值的像素比例。

LiDAR 盲区按 `spnet_patch_size` 划分：一个 patch 没有任何有效 LiDAR 投影时，
整个 patch 记为盲区。A/B 两组的同一帧必须具有相同 `blind_ratio`，否则分析器
拒绝比较。

### `ablation/candidate_flow.csv`

每个关键帧一行：

```text
image_name,lidar_candidates,spnet_proposed,spnet_inserted,total_inserted,spnet_insert_rate
```

它用于判断 SPNet 实际提出了多少补点，其中多少通过筛选并进入地图。
`without_spnet` 中的 SPNet 计数应为零。

### `ablation/runtime_summary.csv`

每次运行一行：

```text
enable_spnet,spnet_calls,spnet_total_seconds,spnet_mean_ms,total_mapping_seconds,total_extending_seconds,final_gaussians
```

SPNet 计时包含推理和补点后处理；Without 模式的 SPNet 调用数和时间应为零。

### `gpu_memory.csv`

```text
unix_time,memory_used_mb
```

脚本每秒采样一次 GPU compute process 显存。最终分析取其最大值作为该运行的
峰值显存。

## 汇总已有结果

实验脚本在六次运行全部成功后会自动调用分析器。也可以手动重新汇总：

```bash
python3 scripts/analyze_spnet_ablation.py \
  --experiment-root /root/autodl-tmp/experiments/spnet_ablation/spnet_fast_20260908
```

`analysis/` 下生成：

- `summary_by_sequence.csv`：每条序列的 With、Without 均值及
  `With - Without` 差值；
- `paired_frame_metrics.csv`：所有 test 帧的一一配对数据；
- `run_summary.csv`：候选贡献、运行时间、最终 Gaussian 数和峰值显存；
- `worst_overall_frames.csv`：总体 PSNR 较差的帧从差到好排序；
- `worst_blind_frames.csv`：盲区 PSNR 较差的帧从差到好排序；
- `largest_spnet_regressions.csv`：按 SPNet 导致的 PSNR 变化从差到好排序。

对于 PSNR、SSIM 和 blind alpha coverage，正差值表示 SPNet 更好；对于 LPIPS，
负差值表示 SPNet 更好。查看差帧时，用 CSV 中的 `sequence` 和 `image_name` 在
两个 condition 的 `render/`、`gt/`、`blind_mask/` 和 `alpha/` 中打开同名文件。

若只需要绘制某次运行已有的逐帧曲线：

```bash
python3 scripts/plot_visual_quality_metrics.py \
  --result_root /root/autodl-tmp/experiments/spnet_ablation/spnet_fast_20260908/hku_campus_seq_00/with_spnet
```

## 有效性与失败判定

只有满足以下条件的 A/B pair 才能用于结论：

- 两组日志都有 `Gaussian-LIC Done!`；
- 两组 `manifest.yaml` 的 bag、配置、种子和 baseline 模式一致；
- test CSV 的 `image_name` 集合完全一致；
- 同名帧的 `blind_ratio` 完全一致；
- `with_spnet` 有 SPNet 调用记录，`without_spnet` 没有；
- 未发生超时、中途退出或缺失指标文件。

分析器遇到缺帧、重复帧、盲区 mask 不一致或缺少运行级 CSV 时会以非零状态
退出。不得把失败、跳过或不完整运行写入最终比较表。
