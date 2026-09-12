#!/usr/bin/env bash
# 单次 Oracle teacher 实验：先录制 Coco-LIC 输出，再让两个 Gaussian-LIC 分支回放同一份输入。
# Coco-LIC 独立重跑仍可能产生数值差异；因此 reference 与 teacher_replay 只能回放这里录制的三路 topic。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CATKIN_WS="$(cd "${REPO_ROOT}/../.." && pwd)"
COCO_WS="${COCO_WS:-/root/catkin_coco}"
BAG_PATH="${BAG_PATH:-/root/autodl-tmp/dataset/hku_campus_seq_00.bag}"
GAUSSIAN_CONFIG_REL="${GAUSSIAN_CONFIG_REL:-config/r3live.yaml}"
COCO_CONFIG_PATH="${COCO_CONFIG_PATH:-${COCO_WS}/src/Coco-LIC/config/ct_odometry_r3live.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-/root/autodl-tmp/experiments/oracle_teacher_single_insert_hku70}"
RUN_NAME="${RUN_NAME:-$(date -u +%Y%m%dT%H%M%SZ)}"
BAG_START_SECONDS="${BAG_START_SECONDS:-0}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-21600}"
SEED="${SEED:-20260911}"
ORACLE_OMP_NUM_THREADS="${ORACLE_OMP_NUM_THREADS:-1}"

# 这些值决定实验语义，必须显式提供，不能把某次消融设置隐藏成默认值。
TARGET_FRAME_ID=""; TEACHER_BUDGET=""; NEIGHBOR_RADIUS=""; EVAL_INTERVAL=""; BAG_DURATION_SECONDS=""
usage() {
  echo "Usage: $0 --target-frame-id N --teacher-budget N --neighbor-radius N"
  echo "          --eval-every-global-updates N --bag-duration SECONDS [options]"
  echo "Options: --bag FILE --gaussian-config PACKAGE_RELATIVE --coco-config FILE"
  echo "         --output-base DIR --run-name NAME --bag-start SECONDS --timeout-seconds SECONDS --seed N"
}
while (($#)); do
  case "$1" in
    --target-frame-id) TARGET_FRAME_ID="$2"; shift 2 ;; --teacher-budget) TEACHER_BUDGET="$2"; shift 2 ;;
    --neighbor-radius) NEIGHBOR_RADIUS="$2"; shift 2 ;; --eval-every-global-updates) EVAL_INTERVAL="$2"; shift 2 ;;
    --bag-duration) BAG_DURATION_SECONDS="$2"; shift 2 ;; --bag) BAG_PATH="$2"; shift 2 ;;
    --gaussian-config) GAUSSIAN_CONFIG_REL="$2"; shift 2 ;; --coco-config) COCO_CONFIG_PATH="$2"; shift 2 ;;
    --output-base) OUTPUT_BASE="$2"; shift 2 ;; --run-name) RUN_NAME="$2"; shift 2 ;;
    --bag-start) BAG_START_SECONDS="$2"; shift 2 ;; --timeout-seconds) RUN_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;; -h|--help) usage; exit 0 ;; *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$TARGET_FRAME_ID" =~ ^[0-9]+$ ]] || { echo "--target-frame-id is required and must be non-negative" >&2; exit 2; }
[[ "$TEACHER_BUDGET" =~ ^[1-9][0-9]*$ ]] || { echo "--teacher-budget is required and must be positive" >&2; exit 2; }
[[ "$NEIGHBOR_RADIUS" =~ ^[0-9]+$ ]] || { echo "--neighbor-radius is required and must be non-negative" >&2; exit 2; }
[[ "$EVAL_INTERVAL" =~ ^[1-9][0-9]*$ ]] || { echo "--eval-every-global-updates is required and must be positive" >&2; exit 2; }
[[ "$BAG_DURATION_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v d="$BAG_DURATION_SECONDS" 'BEGIN { exit !(d > 0) }' || { echo "--bag-duration must be positive" >&2; exit 2; }
[[ "$BAG_START_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ && "$RUN_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ && "$SEED" =~ ^-?[0-9]+$ && "$ORACLE_OMP_NUM_THREADS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid scalar argument" >&2; exit 2; }
[[ "$GAUSSIAN_CONFIG_REL" != /* && "$BAG_PATH" == /* && "$COCO_WS" == /* && "$COCO_CONFIG_PATH" == /* && "$OUTPUT_BASE" == /* ]] || { echo "Invalid path argument" >&2; exit 2; }
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && "$RUN_NAME" != . && "$RUN_NAME" != .. ]] || { echo "Invalid --run-name" >&2; exit 2; }

GAUSSIAN_CONFIG_SOURCE="${REPO_ROOT}/${GAUSSIAN_CONFIG_REL}"
GAUSSIAN_EXECUTABLE="${CATKIN_WS}/devel/lib/gaussian_lic/gs_mapping"
COCO_EXECUTABLE="${COCO_WS}/devel/lib/cocolic/odometry_node"
ANALYZER="${REPO_ROOT}/scripts/analyze_oracle_teacher_single_insert.py"
for required_file in "$BAG_PATH" "$GAUSSIAN_CONFIG_SOURCE" "$COCO_CONFIG_PATH" "${CATKIN_WS}/devel/setup.bash" "${COCO_WS}/devel/setup.bash" "$GAUSSIAN_EXECUTABLE" "$COCO_EXECUTABLE" "$ANALYZER"; do [[ -f "$required_file" ]] || { echo "Missing required file: $required_file" >&2; exit 1; }; done
STALE_SOURCE="$(find "${REPO_ROOT}/src" -type f \( -name '*.cpp' -o -name '*.h' \) -newer "$GAUSSIAN_EXECUTABLE" -print -quit)"
[[ -z "$STALE_SOURCE" ]] || { echo "Gaussian-LIC build is stale; rebuild first: $STALE_SOURCE" >&2; exit 1; }
COCO_STALE_SOURCE="$(find "${COCO_WS}/src/Coco-LIC/src" -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) -newer "$COCO_EXECUTABLE" -print -quit)"
[[ -z "$COCO_STALE_SOURCE" ]] || { echo "Coco-LIC build is stale; rebuild first: $COCO_STALE_SOURCE" >&2; exit 1; }
source "${CATKIN_WS}/devel/setup.bash"
for command_name in roslaunch rosrun rosparam rosbag nvidia-smi; do command -v "$command_name" >/dev/null || { echo "Missing command: $command_name" >&2; exit 1; }; done
nvidia-smi -L | grep -q GPU || { echo "No GPU detected. Switch to a GPU-enabled instance." >&2; exit 1; }
rosparam get /run_id >/dev/null 2>&1 && { echo "An active ROS master already exists. Stop it before running this experiment." >&2; exit 1; }
DA3_MODEL_PATH="$(awk '$1 == "da3_model_path:" {print $2; exit}' "$GAUSSIAN_CONFIG_SOURCE")"; DA3_MODEL_PATH="${DA3_MODEL_PATH%\"}"; DA3_MODEL_PATH="${DA3_MODEL_PATH#\"}"
[[ -f "$DA3_MODEL_PATH" ]] || { echo "Missing DA3 model: $DA3_MODEL_PATH" >&2; exit 1; }
RESULT_ROOT="${OUTPUT_BASE}/${RUN_NAME}"; [[ ! -e "$RESULT_ROOT" ]] || { echo "Refusing to overwrite: $RESULT_ROOT" >&2; exit 1; }; mkdir -p "${RESULT_ROOT}/artifacts"
ARTIFACT_PATH="${RESULT_ROOT}/artifacts/teacher_snapshot.pt"; FRONTEND_BAG_PREFIX="${RESULT_ROOT}/artifacts/cocolic_to_gaussianlic"; FRONTEND_BAG_PATH="${FRONTEND_BAG_PREFIX}.bag"

mapping_pid=""; player_pid=""; coco_pid=""; recorder_pid=""; core_pid=""; active_status_file=""; owned_ros_run_id=""
write_status() { local path="$1" status="$2" detail="${3:-}"; { echo "status: ${status}"; echo "updated_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "detail: ${detail}"; } > "$path"; }
stop_process_group() {
  local pid="$1" signal="${2:-TERM}"; [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -"$signal" -- "-${pid}" 2>/dev/null || kill -"$signal" "$pid" 2>/dev/null || true
    local deadline=$(( $(date +%s) + 15 )); while kill -0 "$pid" 2>/dev/null && (( $(date +%s) < deadline )); do sleep 1; done
    kill -0 "$pid" 2>/dev/null && { kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true; }
  fi
  wait "$pid" 2>/dev/null || true
}
stop_owned_ros_run() {
  [[ -n "$owned_ros_run_id" ]] || return 0
  local marker="/root/.ros/log/${owned_ros_run_id}/"; local pids=(); mapfile -t pids < <(pgrep -f "__log:=${marker}" || true)
  if ((${#pids[@]})); then
    # 节点可能脱离 roslaunch 的进程组；run_id 日志目录仅属于本次启动，可据此精确清理。
    kill -TERM "${pids[@]}" 2>/dev/null || true
    local deadline=$(( $(date +%s) + 15 )); while (( $(date +%s) < deadline )); do local alive=false pid; for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && { alive=true; break; }; done; [[ "$alive" == false ]] && break; sleep 1; done
    for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true; done
  fi
  owned_ros_run_id=""
}
cleanup() {
  # recorder 必须 SIGINT 才会完整写出索引；其余均为本脚本创建的进程组。
  stop_process_group "$recorder_pid" INT; stop_process_group "$player_pid"; stop_process_group "$coco_pid"; stop_process_group "$mapping_pid"; stop_process_group "$core_pid"; stop_owned_ros_run
  mapping_pid=""; player_pid=""; coco_pid=""; recorder_pid=""; core_pid=""
}
on_exit() { local code=$?; trap - EXIT INT TERM; cleanup; if ((code != 0)) && [[ -n "$active_status_file" ]]; then write_status "$active_status_file" failed_or_interrupted "exit_code=${code}"; fi; exit "$code"; }
trap on_exit EXIT; trap 'exit 130' INT TERM
set_yaml_scalar() { local file_path="$1" key="$2" value="$3" temp_path="${1}.tmp"; awk -v k="$key" -v v="$value" 'BEGIN{f=0} $0 ~ "^[[:space:]]*" k "[[:space:]]*:" {print k ": " v;f=1;next} {print} END{if(!f)print k ": " v}' "$file_path" > "$temp_path"; mv "$temp_path" "$file_path"; }
wait_for_master() {
  local label="$1" deadline=$(( $(date +%s) + 60 )); while ! rosparam get /run_id >/dev/null 2>&1; do (( $(date +%s) < deadline )) || { echo "ROS master startup timeout (${label})" >&2; exit 124; }; sleep 1; done
  owned_ros_run_id="$(rosparam get /run_id)"; [[ "$owned_ros_run_id" =~ ^[A-Za-z0-9-]+$ ]] || { echo "Invalid ROS run_id (${label})" >&2; exit 1; }
}

capture_frontend_bag() {
  local capture_dir="${RESULT_ROOT}/frontend_capture" coco_cfg="${RESULT_ROOT}/frontend_capture/coco_config_effective.yaml"; mkdir -p "$capture_dir"; active_status_file="${capture_dir}/status.yaml"; write_status "$active_status_file" starting recording_cocolic_output
  cp "$COCO_CONFIG_PATH" "$coco_cfg"; set_yaml_scalar "$coco_cfg" bag_start "$BAG_START_SECONDS"; set_yaml_scalar "$coco_cfg" bag_durr "$BAG_DURATION_SECONDS"; set_yaml_scalar "$coco_cfg" random_seed "$SEED"
  { printf '%q ' roscore; printf '\n'; printf '%q ' rosbag record -O "$FRONTEND_BAG_PREFIX" /image_for_gs /pose_for_gs /points_for_gs; printf '\n'; printf '%q ' rosrun cocolic odometry_node _project_path:="${COCO_WS}/src/Coco-LIC" _config_path:="$coco_cfg" _bag_path:="$BAG_PATH" _pasue_time:=-1 _verbose:=false; printf '\n'; } > "${capture_dir}/commands.txt"
  (cd "$CATKIN_WS"; source devel/setup.bash; exec setsid roscore) > "${capture_dir}/roscore.log" 2>&1 & core_pid=$!; wait_for_master frontend_capture
  # recorder 先向 master 注册并生成 .active，再启动 Coco-LIC，避免漏掉开头的发布消息。
  (cd "$CATKIN_WS"; source devel/setup.bash; exec setsid rosbag record -O "$FRONTEND_BAG_PREFIX" /image_for_gs /pose_for_gs /points_for_gs) > "${capture_dir}/rosbag_record.log" 2>&1 & recorder_pid=$!
  local deadline=$(( $(date +%s) + 30 )); while [[ ! -e "${FRONTEND_BAG_PATH}.active" ]]; do kill -0 "$recorder_pid" 2>/dev/null || { echo "rosbag recorder exited during startup" >&2; exit 1; }; (( $(date +%s) < deadline )) || { echo "rosbag recorder startup timeout" >&2; exit 124; }; sleep 1; done; sleep 1
  (cd "$COCO_WS"; source devel/setup.bash; export OMP_NUM_THREADS="$ORACLE_OMP_NUM_THREADS"; exec setsid rosrun cocolic odometry_node _project_path:="${COCO_WS}/src/Coco-LIC" _config_path:="$coco_cfg" _bag_path:="$BAG_PATH" _pasue_time:=-1 _verbose:=false) > "${capture_dir}/coco_odometry.log" 2>&1 & coco_pid=$!; write_status "$active_status_file" running recording_cocolic_output
  local coco_code=0; wait "$coco_pid" || coco_code=$?; coco_pid=""; { echo "exit_code: ${coco_code}"; echo "observed_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "${capture_dir}/coco_exit_status.yaml"
  # Coco-LIC 已知会在保存轨迹后的清理阶段 134；只有两项日志证据齐全时才把它当作完成。
  if ((coco_code != 0)); then if grep -q "LoadBag .* with duration " "${capture_dir}/coco_odometry.log" && grep -q "Save trajectory at " "${capture_dir}/coco_odometry.log"; then echo "accepted_after_complete_bag_playback: true" >> "${capture_dir}/coco_exit_status.yaml"; else echo "accepted_after_complete_bag_playback: false" >> "${capture_dir}/coco_exit_status.yaml"; exit "$coco_code"; fi; else echo "accepted_after_complete_bag_playback: true" >> "${capture_dir}/coco_exit_status.yaml"; fi
  # SIGINT 使 rosbag 完整关闭 .active 并写索引；不可直接杀掉 recorder。
  stop_process_group "$recorder_pid" INT; recorder_pid=""; [[ -s "$FRONTEND_BAG_PATH" && ! -e "${FRONTEND_BAG_PATH}.active" ]] || { echo "Frontend input bag was not finalized" >&2; exit 1; }
  rosbag info "$FRONTEND_BAG_PATH" > "${capture_dir}/frontend_input_bag_info.txt"; for topic in /image_for_gs /pose_for_gs /points_for_gs; do grep -q "$topic" "${capture_dir}/frontend_input_bag_info.txt" || { echo "Recorded bag is missing topic: $topic" >&2; exit 1; }; done
  cleanup; write_status "$active_status_file" completed "frontend_input_bag=$(basename "$FRONTEND_BAG_PATH")"; active_status_file=""
}

run_arm() {
  local arm_name="$1" oracle_mode="$2" arm_dir="${RESULT_ROOT}/${1}" gaussian_cfg="${RESULT_ROOT}/${1}/gaussian_config_effective.yaml"; mkdir -p "$arm_dir"; cp "$GAUSSIAN_CONFIG_SOURCE" "$gaussian_cfg"
  set_yaml_scalar "$gaussian_cfg" dataset_path "\"${arm_dir}\""; set_yaml_scalar "$gaussian_cfg" generate_dataset false; set_yaml_scalar "$gaussian_cfg" use_Gaussian_regress false; set_yaml_scalar "$gaussian_cfg" enable_spnet false; set_yaml_scalar "$gaussian_cfg" enable_pose_refinement false; set_yaml_scalar "$gaussian_cfg" enable_train_visual_eval false; set_yaml_scalar "$gaussian_cfg" enable_ablation_logging true; set_yaml_scalar "$gaussian_cfg" experiment_seed "$SEED"; set_yaml_scalar "$gaussian_cfg" oracle_teacher_mode "\"${oracle_mode}\""; set_yaml_scalar "$gaussian_cfg" oracle_target_frame_id "$TARGET_FRAME_ID"; set_yaml_scalar "$gaussian_cfg" oracle_teacher_budget "$TEACHER_BUDGET"; set_yaml_scalar "$gaussian_cfg" oracle_neighbor_radius "$NEIGHBOR_RADIUS"; set_yaml_scalar "$gaussian_cfg" oracle_eval_interval "$EVAL_INTERVAL"; set_yaml_scalar "$gaussian_cfg" oracle_artifact_path "\"${ARTIFACT_PATH}\""
  local gaussian_cfg_rel; gaussian_cfg_rel="$(realpath --relative-to="$REPO_ROOT" "$gaussian_cfg")"; active_status_file="${arm_dir}/status.yaml"; write_status "$active_status_file" starting "oracle_mode=${oracle_mode}; input_bag=$(basename "$FRONTEND_BAG_PATH")"
  { printf '%q ' roslaunch gaussian_lic r3live.launch config_path:="$gaussian_cfg_rel" dataset_path:="$arm_dir" result_path:="$arm_dir" generate_dataset:=false use_Gaussian_regress:=false enable_spnet:=false enable_ablation_logging:=true experiment_seed:="$SEED"; printf '\n'; printf '%q ' rosbag play --delay=1 "$FRONTEND_BAG_PATH"; printf '\n'; } > "${arm_dir}/commands.txt"
  (cd "$CATKIN_WS"; source devel/setup.bash; export OMP_NUM_THREADS="$ORACLE_OMP_NUM_THREADS"; exec setsid roslaunch gaussian_lic r3live.launch config_path:="$gaussian_cfg_rel" dataset_path:="$arm_dir" result_path:="$arm_dir" generate_dataset:=false use_Gaussian_regress:=false enable_spnet:=false enable_ablation_logging:=true experiment_seed:="$SEED") > "${arm_dir}/roslaunch_gaussian.log" 2>&1 & mapping_pid=$!; wait_for_master "$arm_name"
  # delay 只留给订阅者建连，不缩放 bag 的时间；两个 arm 因而获得逐消息一致的输入。
  (cd "$CATKIN_WS"; source devel/setup.bash; exec setsid rosbag play --delay=1 "$FRONTEND_BAG_PATH") > "${arm_dir}/rosbag_play.log" 2>&1 & player_pid=$!; write_status "$active_status_file" running "oracle_mode=${oracle_mode}; input_bag=$(basename "$FRONTEND_BAG_PATH")"
  local deadline=$(( $(date +%s) + RUN_TIMEOUT_SECONDS )) player_code=0
  while (( $(date +%s) < deadline )); do
    grep -q "Gaussian-LIC Done!" "${arm_dir}/gaussian_lic_log.txt" 2>/dev/null && break; kill -0 "$mapping_pid" 2>/dev/null || { echo "Mapping exited without completion (${arm_name})" >&2; exit 1; }
    if [[ -n "$player_pid" ]] && ! kill -0 "$player_pid" 2>/dev/null; then wait "$player_pid" || player_code=$?; player_pid=""; ((player_code == 0)) || { echo "rosbag play failed (${arm_name}), exit=${player_code}" >&2; exit "$player_code"; }; fi; sleep 2
  done
  grep -q "Gaussian-LIC Done!" "${arm_dir}/gaussian_lic_log.txt" 2>/dev/null || { echo "Experiment timeout (${arm_name})" >&2; exit 124; }
  cleanup; for required_output in "${arm_dir}/gaussian_lic_log.txt" "${arm_dir}/oracle_teacher/metrics.csv" "${arm_dir}/visual_quality/train/frame_metrics.csv"; do [[ -s "$required_output" ]] || { echo "Missing output (${arm_name}): $required_output" >&2; exit 1; }; done
  write_status "$active_status_file" completed "oracle_mode=${oracle_mode}"; active_status_file=""; local master_stop_deadline=$(( $(date +%s) + 30 )); while rosparam get /run_id >/dev/null 2>&1 && (( $(date +%s) < master_stop_deadline )); do sleep 1; done; if rosparam get /run_id >/dev/null 2>&1; then echo "ROS master did not stop (${arm_name})" >&2; exit 1; fi
  # rosparam 在 master 已正常关闭时返回非零；显式成功返回，不能把该探测结果冒泡给 set -e。
  return 0
}

git -C "$REPO_ROOT" status --short > "${RESULT_ROOT}/git_status.txt"; git -C "$REPO_ROOT" diff --binary > "${RESULT_ROOT}/git_diff.patch"; cp "${BASH_SOURCE[0]}" "${RESULT_ROOT}/invocation_script.sh"
{ printf '%q ' "$0" --target-frame-id "$TARGET_FRAME_ID" --teacher-budget "$TEACHER_BUDGET" --neighbor-radius "$NEIGHBOR_RADIUS" --eval-every-global-updates "$EVAL_INTERVAL" --bag-duration "$BAG_DURATION_SECONDS" --bag "$BAG_PATH" --gaussian-config "$GAUSSIAN_CONFIG_REL" --coco-config "$COCO_CONFIG_PATH" --output-base "$OUTPUT_BASE" --run-name "$RUN_NAME" --bag-start "$BAG_START_SECONDS" --timeout-seconds "$RUN_TIMEOUT_SECONDS" --seed "$SEED"; printf '\n'; } > "${RESULT_ROOT}/command.txt"
{ echo "experiment: oracle_teacher_single_insert"; echo "target_frame_id: ${TARGET_FRAME_ID}"; echo "teacher_budget: ${TEACHER_BUDGET}"; echo "neighbor_radius: ${NEIGHBOR_RADIUS}"; echo "eval_interval_global_updates: ${EVAL_INTERVAL}"; echo "source_bag_path: ${BAG_PATH}"; echo "frontend_input_source: recorded_cocolic_rosbag"; echo "frontend_input_bag: artifacts/$(basename "$FRONTEND_BAG_PATH")"; echo "frontend_topics: [/image_for_gs, /pose_for_gs, /points_for_gs]"; echo "bag_start_seconds: ${BAG_START_SECONDS}"; echo "bag_duration_seconds: ${BAG_DURATION_SECONDS}"; echo "experiment_seed: ${SEED}"; echo "omp_num_threads: ${ORACLE_OMP_NUM_THREADS}"; echo "git_commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"; echo "started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "da3_model_path: ${DA3_MODEL_PATH}"; echo "da3_model_size_bytes: $(stat -c %s "$DA3_MODEL_PATH")"; } > "${RESULT_ROOT}/manifest.yaml"
capture_frontend_bag; run_arm reference capture; [[ -s "$ARTIFACT_PATH" && -s "${ARTIFACT_PATH}.yaml" ]] || { echo "Reference completed without a valid teacher artifact" >&2; exit 1; }; run_arm teacher_replay replay
python3 "$ANALYZER" --reference-metrics "${RESULT_ROOT}/reference/oracle_teacher/metrics.csv" --replay-metrics "${RESULT_ROOT}/teacher_replay/oracle_teacher/metrics.csv" --output-dir "${RESULT_ROOT}/comparison"
echo "finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RESULT_ROOT}/manifest.yaml"; echo "Experiment completed: ${RESULT_ROOT}"
