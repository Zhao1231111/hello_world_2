# 多 K Oracle teacher：手动执行命令

本记录用于在**一条独立 baseline reference 轨迹**中，为每个 insertion 批次同时导出多个
teacher 预算 `K` 的参数快照，然后分别运行使用某一个 `K` 的 cascade replay。

例如设置 `FRAMES=(84 199)`、`TEACHER_BUDGETS=(64 128 256)` 后，reference 会生成：

```text
teacher_frame84_k64.pt
teacher_frame84_k128.pt
teacher_frame84_k256.pt
teacher_frame199_k64.pt
teacher_frame199_k128.pt
teacher_frame199_k256.pt
```

每个 insertion 批次分别维护平均优化次数，绝不会把 frame 84 和 frame 199 的点混在一起
计算 K。reference 轨迹只运行一次；K=64、128、256 的 replay 使用同一份 reference，
但各自写入独立目录。

不需要重新录制 Coco-LIC bag。本记录不修改共享 `r3live.yaml`，也不会覆盖已有 run。

## 1. 设置实验变量并准备多 K reference

在终端 B 执行。以后更换 insertion 帧、K 集合或邻域大小时，只修改本节变量。

```bash
REPO=/root/catkin_gaussian/src/Gaussian-LIC
CATKIN_WS=/root/catkin_gaussian
OUTPUT_BASE=/root/autodl-tmp/experiments/oracle_teacher_multi_insert_hku70
FRONTEND_BAG=/root/autodl-tmp/experiments/oracle_teacher_single_insert_hku70/hku_campus_00_70s_frame199_k32_m10_recorded_bag/artifacts/cocolic_to_gaussianlic.bag

# insertion 帧必须严格递增，并且必须是实际发生 Gaussian insertion 的关键帧。
FRAMES=(4 84 189 199 224)

# 同一条 reference 轨迹需要保存的 K 集合。C++ 会排序，并拒绝非正数或重复值。
TEACHER_BUDGETS=(128 256 512)

NEIGHBOR_RADIUS=5
EVAL_INTERVAL=3
SEED=20260911

FRAMES_STR=$(IFS=-; echo "${FRAMES[*]}")
BUDGETS_STR=$(IFS=-; echo "${TEACHER_BUDGETS[*]}")
RUN_LABEL="frames${FRAMES_STR}_ks${BUDGETS_STR}_m${NEIGHBOR_RADIUS}"

RUN="${OUTPUT_BASE}/hku_campus_00_70s_cascade_${RUN_LABEL}"
REFERENCE="${RUN}/reference"
ARTIFACT_DIR="${RUN}/artifacts"
REFERENCE_CONFIG="${REFERENCE}/gaussian_config_effective.yaml"

[[ -s "$FRONTEND_BAG" ]] || {
  echo "缺少已录制的前端 bag：$FRONTEND_BAG" >&2
  exit 1
}
[[ ${#FRAMES[@]} -gt 0 ]] || { echo "FRAMES 不能为空" >&2; exit 1; }
[[ ${#TEACHER_BUDGETS[@]} -gt 0 ]] || {
  echo "TEACHER_BUDGETS 不能为空" >&2
  exit 1
}
[[ ! -e "$RUN" ]] || { echo "拒绝覆盖已有 run：$RUN" >&2; exit 1; }
mkdir -p "$REFERENCE" "$ARTIFACT_DIR"

# 复制共享配置后，只修改 run 专用副本。路径模板由程序针对每个 frame × K 展开。
cp "${REPO}/config/r3live.yaml" "$REFERENCE_CONFIG"
printf '\n# 本次多 K Oracle teacher reference（run 专用）\n' >> "$REFERENCE_CONFIG"
printf 'oracle_teacher_mode: "capture"\n' >> "$REFERENCE_CONFIG"

printf 'oracle_target_frame_ids: [' >> "$REFERENCE_CONFIG"
printf '%s' "${FRAMES[0]}" >> "$REFERENCE_CONFIG"
for FRAME_ID in "${FRAMES[@]:1}"; do
  printf ', %s' "$FRAME_ID" >> "$REFERENCE_CONFIG"
done
printf ']\n' >> "$REFERENCE_CONFIG"

printf 'oracle_teacher_budgets: [' >> "$REFERENCE_CONFIG"
printf '%s' "${TEACHER_BUDGETS[0]}" >> "$REFERENCE_CONFIG"
for K in "${TEACHER_BUDGETS[@]:1}"; do
  printf ', %s' "$K" >> "$REFERENCE_CONFIG"
done
printf ']\n' >> "$REFERENCE_CONFIG"

printf 'oracle_neighbor_radius: %s\n' "$NEIGHBOR_RADIUS" >> "$REFERENCE_CONFIG"
printf 'oracle_eval_interval: %s\n' "$EVAL_INTERVAL" >> "$REFERENCE_CONFIG"
printf 'oracle_artifact_path_template: "%s/teacher_frame{frame}_k{budget}.pt"\n' \
  "$ARTIFACT_DIR" >> "$REFERENCE_CONFIG"

# 保存代码状态，便于之后判断每个结果由哪一版实现产生。
git -C "$REPO" status --short > "${RUN}/git_status.txt"
git -C "$REPO" diff --binary > "${RUN}/git_diff.patch"
CONFIG_REL=$(realpath --relative-to="$REPO" "$REFERENCE_CONFIG")

echo "本次 reference 配置：$REFERENCE_CONFIG"
echo "本次多 K artifact 模板：${ARTIFACT_DIR}/teacher_frame{frame}_k{budget}.pt"
```

## 2. 运行一次 reference / capture

终端 A 启动独立 ROS master：

```bash
source /opt/ros/noetic/setup.bash
source /root/catkin_gaussian/devel/setup.bash
roscore
```

保留第 1 节变量的终端 B 启动 Gaussian-LIC：

```bash
source "${CATKIN_WS}/devel/setup.bash"
cd "$CATKIN_WS"
export OMP_NUM_THREADS=1

roslaunch gaussian_lic r3live.launch \
  config_path:="$CONFIG_REL" \
  dataset_path:="$REFERENCE" \
  result_path:="$REFERENCE" \
  generate_dataset:=false \
  use_Gaussian_regress:=false \
  enable_spnet:=false \
  enable_ablation_logging:=true \
  experiment_seed:="$SEED" \
  > "${REFERENCE}/roslaunch_gaussian.log" 2>&1
```

确认 Gaussian-LIC 已启动订阅者后，在终端 C 回放固定前端 bag：

```bash
source /root/catkin_gaussian/devel/setup.bash
rosbag play --delay=1 "$FRONTEND_BAG"
```

reference 正常结束后，在终端 B 检查每个 frame × K 的 artifact：

```bash
for FRAME_ID in "${FRAMES[@]}"; do
  for K in "${TEACHER_BUDGETS[@]}"; do
    ARTIFACT="${ARTIFACT_DIR}/teacher_frame${FRAME_ID}_k${K}.pt"
    [[ -s "$ARTIFACT" ]] || {
      echo "teacher 未生成：frame=${FRAME_ID}, K=${K}" >&2
      exit 1
    }
  done
done

[[ -s "${REFERENCE}/oracle_teacher/metrics.csv" ]] || {
  echo "reference metrics 不完整" >&2
  exit 1
}

grep -E '\[OracleTeacher\] teacher snapshot saved|Gaussian-LIC Done!' \
  "${REFERENCE}/gaussian_lic_log.txt"
```

如果某个较晚 insertion 批次直到序列结束仍未达到最大 K，程序会报出类似：

```text
[OracleTeacher] target batch did not reach configured teacher budgets: frame 449, missing_K=[128,256]
```

这时已经达到的较小 K 快照仍会保留，但不要把缺失 K 当作完整实验。应由人工决定降低
K 集合、移除过晚的 insertion 帧，或延长输入序列。

## 3. 为一个选定 K 准备 replay

先在终端 A用 `Ctrl-C` 停止 reference 使用的 `roscore`，再启动一个新的 `roscore`。
然后回到保留第 1 节变量的终端 B。下面只需修改 `REPLAY_BUDGET`；它必须属于
`TEACHER_BUDGETS`。

```bash
# 第一次可设为 64；完成后分别改成 128、256，重复第 3～5 节。
REPLAY_BUDGET=64

case " ${TEACHER_BUDGETS[*]} " in
  *" ${REPLAY_BUDGET} "*) ;;
  *) echo "REPLAY_BUDGET=${REPLAY_BUDGET} 不在 TEACHER_BUDGETS 中" >&2; exit 1 ;;
esac

REPLAY="${RUN}/teacher_replay_k${REPLAY_BUDGET}"
REPLAY_CONFIG="${REPLAY}/gaussian_config_effective.yaml"
[[ ! -e "$REPLAY" ]] || { echo "拒绝覆盖已有 replay：$REPLAY" >&2; exit 1; }
mkdir -p "$REPLAY"

# reference 配置中保留完整 K 集合和同一 artifact 模板；replay 只通过
# oracle_replay_budget 选择其中一个 K，因此不会重新 capture，也不会混用预算。
cp "$REFERENCE_CONFIG" "$REPLAY_CONFIG"
sed -i 's/oracle_teacher_mode: "capture"/oracle_teacher_mode: "replay"/' "$REPLAY_CONFIG"
printf 'oracle_replay_budget: %s\n' "$REPLAY_BUDGET" >> "$REPLAY_CONFIG"
printf 'oracle_replay_attributes: "all"\n' >> "$REPLAY_CONFIG"
REPLAY_CONFIG_REL=$(realpath --relative-to="$REPO" "$REPLAY_CONFIG")

echo "将运行 K=${REPLAY_BUDGET}：$REPLAY"
```

## 4. 运行这个 K 的 cascade replay

终端 A：

```bash
source /opt/ros/noetic/setup.bash
source /root/catkin_gaussian/devel/setup.bash
roscore
```

终端 B：

```bash
source "${CATKIN_WS}/devel/setup.bash"
cd "$CATKIN_WS"
export OMP_NUM_THREADS=1

roslaunch gaussian_lic r3live.launch \
  config_path:="$REPLAY_CONFIG_REL" \
  dataset_path:="$REPLAY" \
  result_path:="$REPLAY" \
  generate_dataset:=false \
  use_Gaussian_regress:=false \
  enable_spnet:=false \
  enable_ablation_logging:=true \
  experiment_seed:="$SEED" \
  > "${REPLAY}/roslaunch_gaussian.log" 2>&1
```

确认订阅者启动后，终端 C 仍然回放同一份 bag：

```bash
source /root/catkin_gaussian/devel/setup.bash
rosbag play --delay=1 "$FRONTEND_BAG"
```

结束后确认每个 insertion 事件都应用了所选 K 的 teacher：

```bash
grep -E '\[OracleTeacher\] replay applied|Gaussian-LIC Done!' \
  "${REPLAY}/gaussian_lic_log.txt"

[[ -s "${REPLAY}/oracle_teacher/metrics.csv" ]] || {
  echo "K=${REPLAY_BUDGET} 的 replay metrics 不完整" >&2
  exit 1
}
```

## 5. 配对并绘制这个 K 的全部待检查关键帧曲线

```bash
python3 "${REPO}/scripts/analyze_oracle_teacher_single_insert.py" \
  --reference-metrics "${REFERENCE}/oracle_teacher/metrics.csv" \
  --replay-metrics "${REPLAY}/oracle_teacher/metrics.csv" \
  --output-dir "${RUN}/comparison_k${REPLAY_BUDGET}"

sed -n '1,20p' "${RUN}/comparison_k${REPLAY_BUDGET}/pairing_report.txt"
```

只有 `pairing_report.txt` 中 `reference_only_rows: 0` 和 `replay_only_rows: 0` 时，两条
轨迹的逐帧曲线才可直接比较。曲线位于
`comparison_k${REPLAY_BUDGET}/frame_XXXX_curves.png`，横轴始终是绝对全局单视图更新次数
`G`。

完成 K=64 后，停止本次 `roscore`，回到第 3 节把 `REPLAY_BUDGET` 改为 128；之后再改为
256。每个 K 都复用同一个 `${REFERENCE}` 和 `${ARTIFACT_DIR}`，但拥有独立的 replay、
日志、metrics 和 comparison 目录。