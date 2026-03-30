/*
 * 位姿评估日志记录器
 *
 * 该模块负责将每帧的优化前位姿、优化后位姿、时间戳和位姿滤波状态写入 CSV，
 * 供离线 ArUco 真值提取与误差统计脚本自动读取。
 */

#pragma once

#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>

class Camera;

struct PoseEvalFrameStatus
{
    int frame_idx = -1;                  // 原始序列中的帧编号
    bool is_keyframe = false;            // 当前帧是否作为关键帧处理
    bool pose_refine_attempted = false;  // 是否尝试执行了位姿滤波
    bool pose_refine_success = false;    // 位姿滤波是否成功
    int valid_pixels = -1;               // 线性化有效像素数；未尝试时为 -1
    double rmse = std::numeric_limits<double>::quiet_NaN();  // 位姿滤波残差 RMSE；未尝试时为 NaN
    double alpha_coverage_ratio = std::numeric_limits<double>::quiet_NaN(); // alpha 有效覆盖率
    bool rejected_by_guard = false;      // 是否因为稳定性保护提前结束
    bool step_was_clamped = false;       // 是否发生过步长裁剪
    std::string guard_reason;            // 保护触发原因
};

class PoseEvalRecorder
{
public:
    explicit PoseEvalRecorder(const std::string& result_path);

    void recordFrame(const std::shared_ptr<Camera>& camera,
                     const PoseEvalFrameStatus& status);

    bool enabled() const { return enabled_; }
    const std::string& outputPath() const { return output_path_; }

private:
    void writeHeader();

private:
    std::string output_path_;
    std::ofstream file_;
    std::mutex mutex_;
    bool enabled_ = false;
    bool header_written_ = false;
};
