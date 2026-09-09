#!/usr/bin/env bash
# 当前 2D 光栅器的基线实验：HKU Campus 00，从 bag 起点开始处理 70 秒。
# 本脚本只编排已有二进制，不执行编译；运行前请先在 GPU 实例上完成 catkin build。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CATKIN_WS="$(cd "${REPO_ROOT}/../.." && pwd)"

# 默认值均可通过命令行覆盖，便于后续在不修改脚本的情况下复现实验。
COCO_WS="${COCO_WS:-/root/catkin_coco}"
BAG_PATH="${BAG_PATH:-/root/autodl-tmp/dataset/hku_campus_seq_00.bag}"
GAUSSIAN_CONFIG_REL="${GAUSSIAN_CONFIG_REL:-config/r3live.yaml}"
COCO_CONFIG_PATH="${COCO_CONFIG_PATH:-${COCO_WS}/src/Coco-LIC/config/ct_odometry_r3live.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-/root/autodl-tmp/experiments/2d_rasterizer_baseline_hku70}"
RUN_NAME="${RUN_NAME:-$(date -u +%Y%m%dT%H%M%SZ)}"
BAG_START_SECONDS="${BAG_START_SECONDS:-0}"
BAG_DURATION_SECONDS="${BAG_DURATION_SECONDS:-70}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-7200}"
SEED="${SEED:-20260909}"

usage() {
  echo "Usage: $0 [--bag FILE] [--gaussian-config PACKAGE_RELATIVE] [--coco-config FILE]"
  echo "          [--output-base DIR] [--run-name NAME] [--bag-start SECONDS]"
  echo "          [--bag-duration SECONDS] [--timeout-seconds SECONDS] [--seed N]"
}

while (($#)); do
  case "$1" in
    --bag) BAG_PATH="$2"; shift 2 ;;
    --gaussian-config) GAUSSIAN_CONFIG_REL="$2"; shift 2 ;;
    --coco-config) COCO_CONFIG_PATH="$2"; shift 2 ;;
    --output-base) OUTPUT_BASE="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --bag-start) BAG_START_SECONDS="$2"; shift 2 ;;
    --bag-duration) BAG_DURATION_SECONDS="$2"; shift 2 ;;
    --timeout-seconds) RUN_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# r3live.launch 会在 config_path 前拼接 $(find gaussian_lic)，因此源配置必须使用 package-relative 路径。
[[ "$GAUSSIAN_CONFIG_REL" != /* ]] || {
  echo "--gaussian-config must be package-relative, for example config/r3live.yaml" >&2
  exit 2
}
[[ "$BAG_PATH" == /* ]] || { echo "--bag must be an absolute path" >&2; exit 2; }
[[ "$COCO_WS" == /* ]] || { echo "COCO_WS must be an absolute path" >&2; exit 2; }
[[ "$COCO_CONFIG_PATH" == /* ]] || { echo "--coco-config must be an absolute path" >&2; exit 2; }
[[ "$OUTPUT_BASE" == /* ]] || { echo "--output-base must be an absolute path" >&2; exit 2; }
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && "$RUN_NAME" != "." && "$RUN_NAME" != ".." ]] || {
  echo "--run-name may only contain letters, digits, dot, underscore and hyphen" >&2
  exit 2
}
[[ "$BAG_START_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "Invalid --bag-start" >&2; exit 2; }
[[ "$BAG_DURATION_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "Invalid --bag-duration" >&2; exit 2; }
awk -v duration="$BAG_DURATION_SECONDS" 'BEGIN { exit !(duration > 0) }' || {
  echo "--bag-duration must be greater than zero" >&2
  exit 2
}
[[ "$RUN_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid --timeout-seconds" >&2; exit 2; }
[[ "$SEED" =~ ^-?[0-9]+$ ]] || { echo "Invalid --seed" >&2; exit 2; }

GAUSSIAN_CONFIG_SOURCE="${REPO_ROOT}/${GAUSSIAN_CONFIG_REL}"
GAUSSIAN_EXECUTABLE="${CATKIN_WS}/devel/lib/gaussian_lic/gs_mapping"
COCO_EXECUTABLE="${COCO_WS}/devel/lib/cocolic/odometry_node"

# 创建结果目录以前完成输入、workspace、二进制和 GPU 检查，失败时不留下伪实验目录。
for required_file in \
  "$BAG_PATH" \
  "$GAUSSIAN_CONFIG_SOURCE" \
  "$COCO_CONFIG_PATH" \
  "${CATKIN_WS}/devel/setup.bash" \
  "${COCO_WS}/devel/setup.bash" \
  "$GAUSSIAN_EXECUTABLE" \
  "$COCO_EXECUTABLE"; do
  [[ -f "$required_file" ]] || { echo "Missing required file: $required_file" >&2; exit 1; }
done

# 防止忘记重新编译而误测旧光栅器；任一受本次接口修改影响的源码比节点二进制更新都直接拒绝运行。
STALE_SOURCE="$(find \
  "${REPO_ROOT}/src/diff_surfel_rasterization_2d" \
  "${REPO_ROOT}/src/gaussian.cpp" \
  "${REPO_ROOT}/src/gaussian.h" \
  "${REPO_ROOT}/src/mapping.h" \
  "${REPO_ROOT}/src/pose_optimizer.cpp" \
  "${REPO_ROOT}/src/feature_utils.h" \
  -type f \( -name '*.cu' -o -name '*.cpp' -o -name '*.h' \) \
  -newer "$GAUSSIAN_EXECUTABLE" -print -quit)"
[[ -z "$STALE_SOURCE" ]] || {
  echo "Gaussian-LIC build is stale; rebuild before running. Newer source: $STALE_SOURCE" >&2
  exit 1
}

source "${CATKIN_WS}/devel/setup.bash"
command -v roslaunch >/dev/null || { echo "roslaunch is not available" >&2; exit 1; }
command -v rosrun >/dev/null || { echo "rosrun is not available" >&2; exit 1; }
command -v rosparam >/dev/null || { echo "rosparam is not available" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 1; }
nvidia-smi -L | grep -q GPU || { echo "No GPU detected. Switch to a GPU-enabled instance." >&2; exit 1; }

# 避免连接到遗留 ROS master，使其他节点或参数污染本次计时；脚本退出时也只清理自己启动的进程组。
if rosparam get /run_id >/dev/null 2>&1; then
  echo "An active ROS master already exists. Stop the previous experiment before running this script." >&2
  exit 1
fi

DA3_MODEL_PATH="$(awk '$1 == "da3_model_path:" {print $2; exit}' "$GAUSSIAN_CONFIG_SOURCE")"
DA3_MODEL_PATH="${DA3_MODEL_PATH%\"}"
DA3_MODEL_PATH="${DA3_MODEL_PATH#\"}"
[[ -n "$DA3_MODEL_PATH" && -f "$DA3_MODEL_PATH" ]] || {
  echo "Missing DA3 model referenced by ${GAUSSIAN_CONFIG_SOURCE}: ${DA3_MODEL_PATH}" >&2
  exit 1
}

RESULT_DIR="${OUTPUT_BASE}/${RUN_NAME}"
if [[ -e "$RESULT_DIR" ]]; then
  echo "Experiment directory already exists; refusing to overwrite: $RESULT_DIR" >&2
  exit 1
fi
mkdir -p "$RESULT_DIR"

mapping_pid=""
coco_pid=""
gpu_sampler_pid=""
run_status="preparing"
start_epoch="$(date +%s)"

write_status() {
  local status="$1"
  local detail="${2:-}"
  {
    echo "status: ${status}"
    echo "updated_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "detail: ${detail}"
  } > "${RESULT_DIR}/status.yaml"
}

stop_process_group() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    local stop_deadline=$(( $(date +%s) + 10 ))
    while kill -0 "$pid" 2>/dev/null && (( $(date +%s) < stop_deadline )); do
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  stop_process_group "$gpu_sampler_pid"
  stop_process_group "$coco_pid"
  stop_process_group "$mapping_pid"
}

on_exit() {
  local exit_code=$?
  trap - EXIT INT TERM
  cleanup
  if ((exit_code != 0)); then
    write_status "failed_or_interrupted" "exit_code=${exit_code}; previous_state=${run_status}"
  fi
  exit "$exit_code"
}
trap on_exit EXIT
trap 'exit 130' INT TERM

# 在结果目录内生成真正传给进程的配置副本，不永久修改共享 YAML。
cp "$GAUSSIAN_CONFIG_SOURCE" "${RESULT_DIR}/gaussian_config_effective.yaml"
cp "$COCO_CONFIG_PATH" "${RESULT_DIR}/coco_config_effective.yaml"

set_yaml_scalar() {
  local file_path="$1"
  local key="$2"
  local value="$3"
  local temp_path="${file_path}.tmp"
  awk -v yaml_key="$key" -v yaml_value="$value" '
    BEGIN { found = 0 }
    $0 ~ "^[[:space:]]*" yaml_key "[[:space:]]*:" {
      print yaml_key ": " yaml_value
      found = 1
      next
    }
    { print }
    END { if (!found) print yaml_key ": " yaml_value }
  ' "$file_path" > "$temp_path"
  mv "$temp_path" "$file_path"
}

# baseline 合同：不生成训练数据、不使用回归器、不启用 SPNet，也不做位姿优化。
set_yaml_scalar "${RESULT_DIR}/gaussian_config_effective.yaml" dataset_path "\"${RESULT_DIR}\""
set_yaml_scalar "${RESULT_DIR}/gaussian_config_effective.yaml" generate_dataset false
set_yaml_scalar "${RESULT_DIR}/gaussian_config_effective.yaml" use_Gaussian_regress false
set_yaml_scalar "${RESULT_DIR}/gaussian_config_effective.yaml" enable_spnet false
set_yaml_scalar "${RESULT_DIR}/gaussian_config_effective.yaml" enable_pose_refinement false
set_yaml_scalar "${RESULT_DIR}/gaussian_config_effective.yaml" enable_ablation_logging true
set_yaml_scalar "${RESULT_DIR}/gaussian_config_effective.yaml" experiment_seed "$SEED"

# Coco-LIC 直接读取 bag；通过独立配置副本限定从 bag 起点开始的 70 秒，而不修改外部 workspace。
set_yaml_scalar "${RESULT_DIR}/coco_config_effective.yaml" bag_start "$BAG_START_SECONDS"
set_yaml_scalar "${RESULT_DIR}/coco_config_effective.yaml" bag_durr "$BAG_DURATION_SECONDS"

# launch 文件会把下面的相对路径拼到 package root 后面；realpath 保证传入的是可解析的实际路径。
GAUSSIAN_CONFIG_EFFECTIVE_REL="$(realpath --relative-to="$REPO_ROOT" "${RESULT_DIR}/gaussian_config_effective.yaml")"

cp "${SCRIPT_DIR}/run_2d_rasterizer_baseline_hku70.sh" "${RESULT_DIR}/invocation_script.sh"
git -C "$REPO_ROOT" status --short > "${RESULT_DIR}/git_status.txt"
git -C "$REPO_ROOT" diff --binary > "${RESULT_DIR}/git_diff.patch"
nvidia-smi > "${RESULT_DIR}/gpu_info.txt"
uname -a > "${RESULT_DIR}/system_info.txt"

{
  echo "experiment: 2d_rasterizer_baseline"
  echo "sequence: hku_campus_seq_00"
  echo "bag_path: ${BAG_PATH}"
  echo "bag_start_seconds: ${BAG_START_SECONDS}"
  echo "bag_duration_seconds: ${BAG_DURATION_SECONDS}"
  echo "mode: baseline"
  echo "generate_dataset: false"
  echo "use_Gaussian_regress: false"
  echo "enable_spnet: false"
  echo "enable_pose_refinement: false"
  echo "rasterizer_tile_culling: removed"
  echo "experiment_seed: ${SEED}"
  echo "hypothesis: establish timing and quality reference for the repaired unculled 2D rasterizer"
  echo "changed_factor: none; reference baseline"
  echo "git_commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"
  if [[ -s "${RESULT_DIR}/git_status.txt" ]]; then echo "git_dirty: true"; else echo "git_dirty: false"; fi
  echo "da3_model_path: ${DA3_MODEL_PATH}"
  echo "da3_model_size_bytes: $(stat -c %s "$DA3_MODEL_PATH")"
  echo "da3_model_mtime: $(stat -c %y "$DA3_MODEL_PATH")"
  echo "gaussian_executable_mtime: $(stat -c %y "$GAUSSIAN_EXECUTABLE")"
  echo "coco_executable_mtime: $(stat -c %y "$COCO_EXECUTABLE")"
  echo "started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${RESULT_DIR}/manifest.yaml"

{
  printf '%q ' roslaunch gaussian_lic r3live.launch \
    config_path:="$GAUSSIAN_CONFIG_EFFECTIVE_REL" \
    dataset_path:="$RESULT_DIR" \
    result_path:="$RESULT_DIR" \
    generate_dataset:=false \
    use_Gaussian_regress:=false \
    enable_spnet:=false \
    enable_ablation_logging:=true \
    experiment_seed:="$SEED"
  printf '\n'
  printf '%q ' rosrun cocolic odometry_node \
    _project_path:="${COCO_WS}/src/Coco-LIC" \
    _config_path:="${RESULT_DIR}/coco_config_effective.yaml" \
    _bag_path:="$BAG_PATH" \
    _pasue_time:=-1 \
    _verbose:=false
  printf '\n'
} > "${RESULT_DIR}/commands.txt"

run_status="starting_mapping"
write_status "$run_status" ""
(
  cd "$CATKIN_WS"
  source devel/setup.bash
  exec setsid roslaunch gaussian_lic r3live.launch \
    config_path:="$GAUSSIAN_CONFIG_EFFECTIVE_REL" \
    dataset_path:="$RESULT_DIR" \
    result_path:="$RESULT_DIR" \
    generate_dataset:=false \
    use_Gaussian_regress:=false \
    enable_spnet:=false \
    enable_ablation_logging:=true \
    experiment_seed:="$SEED"
) > "${RESULT_DIR}/roslaunch_gaussian.log" 2>&1 &
mapping_pid=$!

# 等待本次 roslaunch 建立自己的 ROS master；同时检查映射进程是否在启动阶段退出。
master_deadline=$(( $(date +%s) + 60 ))
while ! rosparam get /run_id >/dev/null 2>&1; do
  kill -0 "$mapping_pid" 2>/dev/null || {
    echo "Gaussian-LIC exited before ROS master became ready." >&2
    exit 1
  }
  (( $(date +%s) < master_deadline )) || {
    echo "Timed out waiting for ROS master." >&2
    exit 124
  }
  sleep 1
done

run_status="running"
write_status "$run_status" "processing first ${BAG_DURATION_SECONDS}s of bag"
(
  cd "$COCO_WS"
  source devel/setup.bash
  exec setsid rosrun cocolic odometry_node \
    _project_path:="${COCO_WS}/src/Coco-LIC" \
    _config_path:="${RESULT_DIR}/coco_config_effective.yaml" \
    _bag_path:="$BAG_PATH" \
    _pasue_time:=-1 \
    _verbose:=false
) > "${RESULT_DIR}/coco_odometry.log" 2>&1 &
coco_pid=$!

# 每秒保存设备级利用率和显存，用于解释计时结果；采样器属于本脚本并由 trap 单独回收。
(
  echo "unix_time,gpu_utilization_percent,memory_used_mib"
  while kill -0 "$mapping_pid" 2>/dev/null; do
    sample="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -n 1)"
    echo "$(date +%s),${sample}"
    sleep 1
  done
) > "${RESULT_DIR}/gpu_samples.csv" &
gpu_sampler_pid=$!

run_deadline=$(( $(date +%s) + RUN_TIMEOUT_SECONDS ))
coco_finished=false
while (( $(date +%s) < run_deadline )); do
  if grep -q "Gaussian-LIC Done!" "${RESULT_DIR}/gaussian_lic_log.txt" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$mapping_pid" 2>/dev/null; then
    if wait "$mapping_pid"; then mapping_code=0; else mapping_code=$?; fi
    mapping_pid=""
    echo "Gaussian-LIC exited with code ${mapping_code} without completion marker." >&2
    exit 1
  fi
  if [[ "$coco_finished" == "false" ]] && ! kill -0 "$coco_pid" 2>/dev/null; then
    if wait "$coco_pid"; then coco_code=0; else coco_code=$?; fi
    coco_pid=""
    [[ "$coco_code" -eq 0 ]] || {
      echo "Coco-LIC exited with code ${coco_code}." >&2
      exit "$coco_code"
    }
    coco_finished=true
  fi
  sleep 2
done

if ! grep -q "Gaussian-LIC Done!" "${RESULT_DIR}/gaussian_lic_log.txt" 2>/dev/null; then
  echo "Experiment timed out after ${RUN_TIMEOUT_SECONDS}s without completion marker." >&2
  exit 124
fi

# 完成标志出现后允许 roslaunch 自然退出；超时则由统一 cleanup 仅回收本次进程组。
shutdown_deadline=$(( $(date +%s) + 60 ))
while kill -0 "$mapping_pid" 2>/dev/null && (( $(date +%s) < shutdown_deadline )); do
  sleep 1
done
if kill -0 "$mapping_pid" 2>/dev/null; then
  echo "Gaussian-LIC completed but roslaunch did not exit within 60s; cleaning up its process group." >&2
else
  if wait "$mapping_pid"; then mapping_code=0; else mapping_code=$?; fi
  mapping_pid=""
  [[ "$mapping_code" -eq 0 ]] || {
    echo "Gaussian-LIC returned ${mapping_code} after writing the completion marker." >&2
    exit "$mapping_code"
  }
fi

# 这些文件分别证明运行完成、计时已写出和视觉指标已写出；缺失时不得把 run 标为成功。
for required_output in \
  "${RESULT_DIR}/gaussian_lic_log.txt" \
  "${RESULT_DIR}/ablation/runtime_summary.csv" \
  "${RESULT_DIR}/visual_quality/train/frame_metrics.csv" \
  "${RESULT_DIR}/visual_quality/test/frame_metrics.csv"; do
  [[ -s "$required_output" ]] || {
    echo "Run reached completion marker but required output is missing: $required_output" >&2
    exit 1
  }
done

grep -E "Runtime Statistics|Total Mapping Time:|Forward:|Backward:|Step:|CPU2GPU:|Total Adding Time:|Total Extending Time:|Number of Final Gaussians:|AUTO_TUNE_|ABLATION_NOVEL_" \
  "${RESULT_DIR}/gaussian_lic_log.txt" > "${RESULT_DIR}/key_metrics.txt" || true

end_epoch="$(date +%s)"
{
  echo "finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "wall_seconds: $((end_epoch - start_epoch))"
} >> "${RESULT_DIR}/manifest.yaml"

run_status="completed"
write_status "$run_status" "all required timing and visual metric files are present"
cleanup
mapping_pid=""; coco_pid=""; gpu_sampler_pid=""
echo "Experiment completed: ${RESULT_DIR}"
