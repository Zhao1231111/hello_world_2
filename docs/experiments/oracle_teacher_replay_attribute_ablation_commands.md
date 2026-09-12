# Oracle teacher 属性回放消融：手动执行命令

本记录只运行 replay treatment。它复用已经固定的 Coco-LIC 输出 bag、reference
metrics 和 teacher artifact，**不会**重新录制前端、重新 capture teacher 或重新运行
reference。每个 variant 建立独立目录，避免覆盖已有结果。

属性字段 `oracle_replay_attributes` 的含义如下：选中的属性从 teacher artifact 回插；
未选中的属性保留当前 replay 产生的 rule-based baseline 初始化。合法值为 `all`、`none`，
或逗号分隔的 `xyz,sh_dc,sh_rest,opacity,scaling,rotation`。

| variant | `ATTRIBUTES` |
| --- | --- |
| 完整 teacher（已有对照） | `all` |
| `no_sh_rest` | `xyz,sh_dc,opacity,scaling,rotation` |
| `no_opacity` | `xyz,sh_dc,sh_rest,scaling,rotation` |
| `no_scaling` | `xyz,sh_dc,sh_rest,opacity,rotation` |

## 1. 每个 variant 开始前：在终端 A 准备独立目录和有效配置

先按本次实验修改变量。后续若改变目标帧、预算、邻域或检查频率，只修改这一处；这些值
不会被写死在 C++ 代码中。

```bash
REPO=/root/catkin_gaussian/src/Gaussian-LIC
CATKIN_WS=/root/catkin_gaussian
EXP_ROOT=/root/autodl-tmp/experiments/oracle_teacher_single_insert_hku70/hku_campus_00_70s_frame199_k32_m10_recorded_bag
FRONTEND_BAG="${EXP_ROOT}/artifacts/cocolic_to_gaussianlic.bag"
ARTIFACT="${EXP_ROOT}/artifacts/teacher_snapshot.pt"
REFERENCE_METRICS="${EXP_ROOT}/reference/oracle_teacher/metrics.csv"

TARGET_FRAME_ID=199
TEACHER_BUDGET=32
NEIGHBOR_RADIUS=10
EVAL_INTERVAL=3
SEED=20260911

VARIANT=no_sh_rest
ATTRIBUTES=xyz,sh_dc,opacity,scaling,rotation
RUN="${EXP_ROOT}/replay_attribute_ablation_${VARIANT}_$(date -u +%Y%m%dT%H%M%SZ)"
CONFIG="${RUN}/gaussian_config_effective.yaml"

[[ -s "$FRONTEND_BAG" && -s "$ARTIFACT" && -s "$REFERENCE_METRICS" ]] || {
  echo "缺少既有 bag、artifact 或 reference metrics" >&2; exit 1;
}
[[ ! -e "$RUN" ]] || { echo "拒绝覆盖已有目录：$RUN" >&2; exit 1; }
mkdir -p "$RUN"

# 有效配置只存在本次结果目录：原始 r3live.yaml 不会被改动。
cp "${REPO}/config/r3live.yaml" "$CONFIG"
printf '\n# Oracle teacher replay 属性消融（本次 run 专用）\n' >> "$CONFIG"
printf 'oracle_teacher_mode: "replay"\n' >> "$CONFIG"
printf 'oracle_target_frame_id: %s\n' "$TARGET_FRAME_ID" >> "$CONFIG"
printf 'oracle_teacher_budget: %s\n' "$TEACHER_BUDGET" >> "$CONFIG"
printf 'oracle_neighbor_radius: %s\n' "$NEIGHBOR_RADIUS" >> "$CONFIG"
printf 'oracle_eval_interval: %s\n' "$EVAL_INTERVAL" >> "$CONFIG"
printf 'oracle_artifact_path: "%s"\n' "$ARTIFACT" >> "$CONFIG"
printf 'oracle_replay_attributes: "%s"\n' "$ATTRIBUTES" >> "$CONFIG"

git -C "$REPO" status --short > "${RUN}/git_status.txt"
git -C "$REPO" diff --binary > "${RUN}/git_diff.patch"
CONFIG_REL="$(realpath --relative-to="$REPO" "$CONFIG")"
```

## 2. 终端 A：启动 Gaussian-LIC

先确认没有遗留的 ROS master。下面的 `config_path` 使用相对 `gaussian_lic` package 的
路径，适配当前 launch 文件会在其前面拼接 package root 的行为。

```bash
source "${CATKIN_WS}/devel/setup.bash"
nvidia-smi -L
rosparam get /run_id && { echo "已有 ROS master，先停止它" >&2; exit 1; } || true

cd "$CATKIN_WS"
export OMP_NUM_THREADS=1
roslaunch gaussian_lic r3live.launch \
  config_path:="$CONFIG_REL" \
  dataset_path:="$RUN" \
  result_path:="$RUN" \
  generate_dataset:=false \
  use_Gaussian_regress:=false \
  enable_spnet:=false \
  enable_ablation_logging:=true \
  experiment_seed:="$SEED" \
  > "${RUN}/roslaunch_gaussian.log" 2>&1
```

保持终端 A 运行，等待它建立订阅者。不要在此阶段重新启动 Coco-LIC。

## 3. 终端 B：回放既有前端 bag

确认终端 A 已启动后，在另一个终端执行：

```bash
source /root/catkin_gaussian/devel/setup.bash
rosbag play --delay=1 "/root/autodl-tmp/experiments/oracle_teacher_single_insert_hku70/hku_campus_00_70s_frame199_k32_m10_recorded_bag/artifacts/cocolic_to_gaussianlic.bag"
```

等待终端 A 的日志出现 `Gaussian-LIC Done!` 后自然结束。若出现异常，不要使用
`rosnode kill -a`；只停止本次终端 A 启动的 `roslaunch` 进程。

## 4. 分析本 variant

回到终端 A 原来的 shell，确认结果完整后执行：

```bash
[[ -s "${RUN}/oracle_teacher/metrics.csv" ]] || {
  echo "replay metrics 不完整：${RUN}" >&2; exit 1;
}

python3 "${REPO}/scripts/analyze_oracle_teacher_single_insert.py" \
  --reference-metrics "$REFERENCE_METRICS" \
  --replay-metrics "${RUN}/oracle_teacher/metrics.csv" \
  --output-dir "${RUN}/comparison_against_reference"
```

随后仅替换第 1 步的 `VARIANT` 和 `ATTRIBUTES`，依次运行 `no_opacity`、`no_scaling`。
分析时以每个关键帧的曲线和稳定阈值所节省的全局更新次数为主；不要只比较第一个检查点。
