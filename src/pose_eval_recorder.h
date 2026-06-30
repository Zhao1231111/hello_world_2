#pragma once

#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>

class Camera;

struct PoseEvalFrameStatus
{
    int frame_idx = -1;
    bool is_keyframe = false;
    bool pose_refine_attempted = false;
    bool pose_refine_success = false;
    int valid_pixels = -1;  // Remains -1 when refinement is skipped.
    double rmse = std::numeric_limits<double>::quiet_NaN();  // NaN when refinement is skipped.
    double alpha_coverage_ratio = std::numeric_limits<double>::quiet_NaN();
    bool rejected_by_guard = false;
    bool step_was_clamped = false;
    std::string guard_reason;
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
