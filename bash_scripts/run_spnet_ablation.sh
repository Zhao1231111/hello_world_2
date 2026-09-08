#!/usr/bin/env bash
# SPNet A/B 消融实验编排脚本：对每条序列分别运行开启和关闭 SPNet 的 baseline。
# 任一命令、子进程或未定义变量出错时立即失败，避免产生看似完整的残缺结果。
set -euo pipefail

# 从脚本位置推导仓库和 catkin workspace，避免绑定到固定的仓库绝对路径。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CATKIN_WS="$(cd "${REPO_ROOT}/../.." && pwd)"

# 以下默认值均可由同名环境变量或后续命令行参数覆盖。
COCO_WS="${COCO_WS:-/root/catkin_coco}"
OUTPUT_BASE="${OUTPUT_BASE:-/root/autodl-tmp/experiments/spnet_ablation}"
SEED="${SEED:-20260908}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-10800}"
RUN_NAME="${RUN_NAME:-$(date -u +%Y%m%dT%H%M%SZ)}"
BAG_PATH_OVERRIDE="${BAG_PATH_OVERRIDE:-}"
GAUSSIAN_CONFIG_OVERRIDE="${GAUSSIAN_CONFIG_OVERRIDE:-}"
COCO_CONFIG_OVERRIDE="${COCO_CONFIG_OVERRIDE:-}"
GAUSSIAN_LAUNCH_OVERRIDE="${GAUSSIAN_LAUNCH_OVERRIDE:-}"

# 默认实验矩阵以及各序列对应的 bag、Gaussian-LIC 配置、launch 和 Coco-LIC 配置。
declare -a SELECTED_SEQUENCES=(hku_campus_seq_00 degenerate_seq_00 tuhh_day_04)
declare -A BAG_PATHS=(
  [hku_campus_seq_00]="/root/autodl-tmp/dataset/hku_campus_seq_00.bag"
  [degenerate_seq_00]="/root/autodl-tmp/dataset/degenerate_seq_00.bag"
  [tuhh_day_04]="/root/autodl-tmp/dataset/mcd/tuhh_day_04.bag"
)
declare -A GAUSSIAN_CONFIGS=(
  [hku_campus_seq_00]="config/r3live.yaml"
  [degenerate_seq_00]="config/r3live.yaml"
  [tuhh_day_04]="config/mcd.yaml"
)
declare -A GAUSSIAN_LAUNCHES=(
  [hku_campus_seq_00]="r3live.launch"
  [degenerate_seq_00]="r3live.launch"
  [tuhh_day_04]="mcd.launch"
)
declare -A COCO_CONFIGS=(
  [hku_campus_seq_00]="/config/ct_odometry_r3live.yaml"
  [degenerate_seq_00]="/config/ct_odometry_r3live.yaml"
  [tuhh_day_04]="/config/ct_odometry_mcd.yaml"
)

usage() {
  echo "Usage: $0 [--output-base DIR] [--run-name NAME] [--seed N] [--timeout-seconds N] [--sequence NAME ...]"
  echo "          [--bag-path FILE] [--gaussian-config PACKAGE_RELATIVE] [--coco-config PACKAGE_RELATIVE] [--gaussian-launch FILE]"
}

# 解析命令行。多个 --sequence 会组成自定义序列列表；路径覆盖仅面向单序列调试。
custom_sequences=()
while (($#)); do
  case "$1" in
    --output-base) OUTPUT_BASE="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --timeout-seconds) RUN_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --sequence) custom_sequences+=("$2"); shift 2 ;;
    --bag-path) BAG_PATH_OVERRIDE="$2"; shift 2 ;;
    --gaussian-config) GAUSSIAN_CONFIG_OVERRIDE="$2"; shift 2 ;;
    --coco-config) COCO_CONFIG_OVERRIDE="$2"; shift 2 ;;
    --gaussian-launch) GAUSSIAN_LAUNCH_OVERRIDE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if ((${#custom_sequences[@]})); then
  SELECTED_SEQUENCES=("${custom_sequences[@]}")
fi
# 一组路径覆盖无法无歧义地应用到多条序列，因此要求同时只选择一条序列。
if [[ -n "$BAG_PATH_OVERRIDE$GAUSSIAN_CONFIG_OVERRIDE$COCO_CONFIG_OVERRIDE$GAUSSIAN_LAUNCH_OVERRIDE" ]] \
   && ((${#SELECTED_SEQUENCES[@]} != 1)); then
  echo "Path overrides require exactly one --sequence." >&2
  exit 2
fi

# 在创建结果目录前完成 workspace、ROS 和 GPU 检查，尽早暴露环境问题。
for required in "${CATKIN_WS}/devel/setup.bash" "${COCO_WS}/devel/setup.bash"; do
  [[ -f "$required" ]] || { echo "Missing workspace setup: $required" >&2; exit 1; }
done
source "${CATKIN_WS}/devel/setup.bash"
command -v roslaunch >/dev/null || { echo "roslaunch is not available" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 1; }
nvidia-smi -L | grep -q GPU || { echo "No GPU detected. Switch to a GPU-enabled instance." >&2; exit 1; }

# 每个 run name 对应唯一实验目录，禁止静默覆盖已有实验。
EXPERIMENT_ROOT="${OUTPUT_BASE}/${RUN_NAME}"
if [[ -e "$EXPERIMENT_ROOT" ]]; then
  echo "Experiment directory already exists: $EXPERIMENT_ROOT" >&2
  exit 1
fi
mkdir -p "$EXPERIMENT_ROOT"

mapping_pid=""
coco_pid=""
gpu_pid=""
# 仅终止本脚本记录的进程组；不进行全局 ROS 清理。
cleanup() {
  set +e
  for pid in "$gpu_pid" "$coco_pid" "$mapping_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 执行一条序列的一个 condition。
# 参数依次为：序列名、condition 名以及是否启用 SPNet。
run_one() {
  local sequence="$1"
  local condition="$2"
  local enable_spnet="$3"
  local bag_path="${BAG_PATH_OVERRIDE:-${BAG_PATHS[$sequence]:-}}"
  local config_rel="${GAUSSIAN_CONFIG_OVERRIDE:-${GAUSSIAN_CONFIGS[$sequence]:-}}"
  local launch_file="${GAUSSIAN_LAUNCH_OVERRIDE:-${GAUSSIAN_LAUNCHES[$sequence]:-}}"
  local coco_config="${COCO_CONFIG_OVERRIDE:-${COCO_CONFIGS[$sequence]:-}}"

  # 启动前验证该 condition 所需的输入、配置和模型权重。
  [[ -n "$bag_path" && -n "$config_rel" && -n "$launch_file" && -n "$coco_config" ]] || {
    echo "Unknown sequence: $sequence" >&2; exit 1;
  }
  [[ -f "$bag_path" ]] || { echo "Missing bag: $bag_path" >&2; exit 1; }
  [[ -f "${REPO_ROOT}/${config_rel}" ]] || { echo "Missing config: ${REPO_ROOT}/${config_rel}" >&2; exit 1; }
  [[ -f "${COCO_WS}/src/Coco-LIC${coco_config}" ]] || { echo "Missing Coco-LIC config: ${COCO_WS}/src/Coco-LIC${coco_config}" >&2; exit 1; }
  local da3_model
  da3_model="$(awk '$1 == "da3_model_path:" {print $2; exit}' "${REPO_ROOT}/${config_rel}")"
  [[ -n "$da3_model" && -f "$da3_model" ]] || { echo "Missing DA3 model from ${config_rel}: $da3_model" >&2; exit 1; }
  if [[ "$enable_spnet" == "true" ]]; then
    local spnet_model
    spnet_model="$(awk '$1 == "spnet_model_path:" {print $2; exit}' "${REPO_ROOT}/${config_rel}")"
    [[ -n "$spnet_model" && -f "$spnet_model" ]] || { echo "Missing SPNet model from ${config_rel}: $spnet_model" >&2; exit 1; }
  fi

  # 保存实际配置、命令和 Git 状态，使本次运行可复现并可判断是否直接可比。
  local result_dir="${EXPERIMENT_ROOT}/${sequence}/${condition}"
  mkdir -p "$result_dir"
  cp "${REPO_ROOT}/${config_rel}" "${result_dir}/gaussian_config.yaml"
  cp "${COCO_WS}/src/Coco-LIC${coco_config}" "${result_dir}/coco_config.yaml"
  {
    echo "sequence: ${sequence}"
    echo "condition: ${condition}"
    echo "enable_spnet: ${enable_spnet}"
    echo "use_Gaussian_regress: false"
    echo "enable_ablation_logging: true"
    echo "experiment_seed: ${SEED}"
    echo "bag_path: ${bag_path}"
    echo "gaussian_config: ${config_rel}"
    echo "coco_config: ${coco_config}"
    echo "git_commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"
    if [[ -n "$(git -C "$REPO_ROOT" status --short)" ]]; then echo "git_dirty: true"; else echo "git_dirty: false"; fi
  } > "${result_dir}/manifest.yaml"
  printf '%q ' roslaunch gaussian_lic "$launch_file" \
    config_path:="$config_rel" dataset_path:="$result_dir" result_path:="$result_dir" \
    use_Gaussian_regress:=false enable_spnet:="$enable_spnet" \
    enable_ablation_logging:=true experiment_seed:="$SEED" > "${result_dir}/command.txt"
  printf '\n' >> "${result_dir}/command.txt"

  # 清空上一 condition 的 PID，之后 trap 只会清理本次启动的子进程。
  mapping_pid=""
  coco_pid=""
  gpu_pid=""

  # 先启动 Gaussian-LIC，使 ROS subscriber 在 Coco-LIC 发布数据前准备就绪。
  (
    cd "$CATKIN_WS"
    source devel/setup.bash
    exec setsid roslaunch gaussian_lic "$launch_file" \
      config_path:="$config_rel" \
      dataset_path:="$result_dir" \
      result_path:="$result_dir" \
      use_Gaussian_regress:=false \
      enable_spnet:="$enable_spnet" \
      enable_ablation_logging:=true \
      experiment_seed:="$SEED"
  ) > "${result_dir}/roslaunch_gaussian.log" 2>&1 &
  mapping_pid=$!

  # 给 ROS 节点预留启动时间，并在启动阶段失败时立即终止本次实验。
  sleep 5
  if ! kill -0 "$mapping_pid" 2>/dev/null; then
    echo "Gaussian-LIC exited during startup: ${result_dir}" >&2
    return 1
  fi

  # 启动外部 Coco-LIC frontend，读取 bag 并发布同步后的图像、点云和位姿。
  (
    cd "$COCO_WS"
    source devel/setup.bash
    exec setsid roslaunch cocolic odometry.launch bag_path:="$bag_path" config_path:="$coco_config"
  ) > "${result_dir}/roslaunch_coco.log" 2>&1 &
  coco_pid=$!

  # 每秒记录所有 GPU compute process 的显存总量，供运行级峰值显存统计使用。
  (
    echo "unix_time,memory_used_mb"
    while kill -0 "$mapping_pid" 2>/dev/null; do
      printf '%s,' "$(date +%s)"
      nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{sum += $1} END {print sum + 0}'
      sleep 1
    done
  ) > "${result_dir}/gpu_memory.csv" &
  gpu_pid=$!

  # Coco-LIC 播放完 bag 后可以先退出；唯一成功完成标志是 Gaussian-LIC Done!。
  # 同时保留 Gaussian-LIC 异常退出检查和整个 condition 的有界超时。
  local deadline=$((SECONDS + RUN_TIMEOUT_SECONDS))
  local coco_exit_reported=false
  while ((SECONDS < deadline)); do
    if grep -q "Gaussian-LIC Done!" "${result_dir}/gaussian_lic_log.txt" 2>/dev/null; then
      break
    fi
    if ! kill -0 "$mapping_pid" 2>/dev/null; then
      echo "Gaussian-LIC exited without completion marker: ${result_dir}" >&2
      return 1
    fi
    if [[ "$coco_exit_reported" == "false" ]] && ! kill -0 "$coco_pid" 2>/dev/null; then
      echo "Coco-LIC exited after bag playback; waiting for Gaussian-LIC Done!: ${result_dir}"
      coco_exit_reported=true
    fi
    sleep 2
  done
  if ! grep -q "Gaussian-LIC Done!" "${result_dir}/gaussian_lic_log.txt" 2>/dev/null; then
    echo "Run timed out: ${result_dir}" >&2
    return 124
  fi

  # 完成标志写入后等待 ROS 节点自然关闭；超出宽限期才由 cleanup 回收进程组。
  local shutdown_deadline=$((SECONDS + 30))
  while kill -0 "$mapping_pid" 2>/dev/null && ((SECONDS < shutdown_deadline)); do
    sleep 1
  done
  if kill -0 "$mapping_pid" 2>/dev/null; then
    echo "Gaussian-LIC wrote the completion marker but did not exit within 30 seconds; cleaning up its process group." >&2
  else
    local mapping_status=0
    if wait "$mapping_pid"; then
      mapping_status=0
    else
      mapping_status=$?
    fi
    mapping_pid=""
    if ((mapping_status != 0)); then
      echo "Gaussian-LIC exited with status ${mapping_status} after writing the completion marker: ${result_dir}" >&2
      return "$mapping_status"
    fi
  fi

  # 回收残留的 frontend 和显存采样进程，再进入下一个 condition。
  cleanup
  mapping_pid=""; coco_pid=""; gpu_pid=""
  echo "Completed: ${sequence}/${condition}"
}

# 相邻序列交换 A/B 的运行顺序，以减小固定执行顺序带来的系统性偏差。
for index in "${!SELECTED_SEQUENCES[@]}"; do
  sequence="${SELECTED_SEQUENCES[$index]}"
  if ((index % 2 == 0)); then
    run_one "$sequence" with_spnet true
    run_one "$sequence" without_spnet false
  else
    run_one "$sequence" without_spnet false
    run_one "$sequence" with_spnet true
  fi
done

# 只有所有 condition 均成功后才生成逐帧配对结果和运行级汇总。
python3 "${REPO_ROOT}/scripts/analyze_spnet_ablation.py" --experiment-root "$EXPERIMENT_ROOT"
echo "Experiment complete: $EXPERIMENT_ROOT"
