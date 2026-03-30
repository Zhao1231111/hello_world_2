/*
 * 位姿评估日志记录器实现
 */

#include "pose_eval_recorder.h"

#include "camera.h"

#include <filesystem>
#include <iomanip>
#include <iostream>

namespace
{

/**
 * @brief 将内部使用的世界到相机位姿 T_cw 转成对外统一记录的 T_wc
 */
void convertPoseCwToWc(const Eigen::Matrix3d& R_cw,
                       const Eigen::Vector3d& t_cw,
                       Eigen::Matrix3d& R_wc,
                       Eigen::Vector3d& t_wc)
{
    R_wc = R_cw.transpose();
    t_wc = -R_wc * t_cw;
}

/**
 * @brief 追加一组位姿到 CSV 输出流
 */
void appendPoseCsv(std::ostream& os,
                   const Eigen::Matrix3d& R_cw,
                   const Eigen::Vector3d& t_cw)
{
    Eigen::Matrix3d R_wc = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_wc = Eigen::Vector3d::Zero();
    convertPoseCwToWc(R_cw, t_cw, R_wc, t_wc);

    Eigen::Quaterniond q_wc(R_wc);
    q_wc.normalize();

    os << ','
       << t_wc.x() << ',' << t_wc.y() << ',' << t_wc.z() << ','
       << q_wc.w() << ',' << q_wc.x() << ',' << q_wc.y() << ',' << q_wc.z();
}

}  // namespace

PoseEvalRecorder::PoseEvalRecorder(const std::string& result_path)
{
    std::filesystem::path base_path = result_path.empty() ? std::filesystem::current_path()
                                                          : std::filesystem::path(result_path);
    std::filesystem::create_directories(base_path);
    output_path_ = (base_path / "pose_eval_log.csv").string();

    file_.open(output_path_, std::ios::out | std::ios::trunc);
    if (!file_.is_open())
    {
        std::cerr << "[PoseEval] Failed to open pose log file: " << output_path_ << std::endl;
        return;
    }

    enabled_ = true;
    writeHeader();
}

void PoseEvalRecorder::writeHeader()
{
    if (!enabled_ || header_written_)
    {
        return;
    }

    file_ << "frame_idx,image_name,image_stamp_sec,is_keyframe,"
          << "pose_refine_attempted,pose_refine_success,valid_pixels,rmse,alpha_coverage_ratio,"
          << "rejected_by_guard,step_was_clamped,guard_reason,"
          << "pre_tx,pre_ty,pre_tz,pre_qw,pre_qx,pre_qy,pre_qz,"
          << "post_tx,post_ty,post_tz,post_qw,post_qx,post_qy,post_qz\n";
    file_.flush();
    header_written_ = true;
}

void PoseEvalRecorder::recordFrame(const std::shared_ptr<Camera>& camera,
                                   const PoseEvalFrameStatus& status)
{
    if (!enabled_ || !camera)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // 时间戳和位姿统一采用较高精度输出，便于离线最近邻匹配。
    file_ << std::fixed << std::setprecision(9);
    file_ << status.frame_idx << ','
          << camera->image_name_ << ','
          << camera->image_stamp_sec_ << ','
          << (status.is_keyframe ? 1 : 0) << ','
          << (status.pose_refine_attempted ? 1 : 0) << ','
          << (status.pose_refine_success ? 1 : 0) << ','
          << status.valid_pixels << ','
          << status.rmse << ','
          << status.alpha_coverage_ratio << ','
          << (status.rejected_by_guard ? 1 : 0) << ','
          << (status.step_was_clamped ? 1 : 0) << ','
          << status.guard_reason;

    appendPoseCsv(file_, camera->R_cw_pred_, camera->t_cw_pred_);
    appendPoseCsv(file_, camera->R_cw_, camera->t_cw_);
    file_ << '\n';
    file_.flush();
}
