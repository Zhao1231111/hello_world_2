# hku_campus_00 70 s 单次 Oracle teacher 实验

本页只说明如何运行与读取输出，不记录实验结果。实验脚本依次执行三个阶段：

1. `frontend_capture`：单独运行 Coco-LIC，并录制它发布给 Gaussian-LIC 的 `/image_for_gs`、`/pose_for_gs` 与 `/points_for_gs`；
2. `reference`：回放这份录制 bag，使用规则初始化正常联合优化，并在目标批次平均有效优化次数首次达到预算时保存 teacher 快照；
3. `teacher_replay`：再次回放同一份录制 bag，只在目标 insertion 处用快照参数替换规则初始化，随后仍按 baseline 联合优化。

这样 reference 与 replay 的前端输入逐消息一致，不依赖 Coco-LIC 两次独立运行是否给出完全相同的数值输出。

两条轨迹的横轴 `G` 都是已经完成的单视图参数更新总数。仅当 `G > G_t` 且相对更新次数满足配置的检查间隔时，才对当时已经到达的目标关键帧邻域做整图 PSNR、SSIM 和 LPIPS 评估。指标按关键帧分别保存和绘图，不计算跨帧平均值。

## 运行

先在 GPU 实例中重新编译：

```bash
cd /root/catkin_gaussian
catkin_make -DCMAKE_BUILD_TYPE=Release -j1
```

本次约定参数的运行命令为：

```bash
cd /root/catkin_gaussian
./src/Gaussian-LIC/bash_scripts/run_oracle_teacher_single_insert_hku70.sh \
  --target-frame-id 199 \
  --teacher-budget 32 \
  --neighbor-radius 10 \
  --eval-every-global-updates 3 \
  --bag-duration 70 \
  --run-name hku_campus_00_70s_frame199_k32_m10
```

目标帧、teacher 预算、邻域半径、检查间隔和 bag 时长均为必填命令行参数，脚本及 C++ 实现中没有本次实验数值的默认值。若数据或配置路径不同，可显式追加 `--bag`、`--gaussian-config`、`--coco-config` 或 `--output-base`。

## 输出

默认结果目录为：

```text
/root/autodl-tmp/experiments/oracle_teacher_single_insert_hku70/<run-name>/
```

主要文件：

- `artifacts/cocolic_to_gaussianlic.bag`：唯一的前端输入记录，两个 Gaussian-LIC 分支均回放该文件；
- `frontend_capture/frontend_input_bag_info.txt`：录制 bag 的 topic 与消息摘要；
- `artifacts/teacher_snapshot.pt`：带 schema 的完整 teacher 参数包；相邻 YAML 文件记录预算快照元数据；
- `reference/oracle_teacher/metrics.csv`、`teacher_replay/oracle_teacher/metrics.csv`：逐关键帧、逐绝对 `G` 的整图指标；
- `comparison/paired_metrics.csv`：严格按 `(eval_frame_id, G)` 配对后的两分支指标与差值；
- `comparison/frame_*_curves.png`：每个待检查关键帧自己的 PSNR、SSIM、LPIPS 曲线；
- 两个分支各自的 effective config、命令、完整日志和状态文件，以及顶层 `manifest.yaml`、Git 状态和 diff。

脚本拒绝覆盖已有结果目录；任一分支未命中目标帧、未达到 teacher 预算、未来邻域不完整、检查点无法严格配对或子进程异常时，整次实验都会失败并保留状态说明。
