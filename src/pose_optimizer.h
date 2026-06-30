#pragma once

#include <memory>
#include <string>
#include <Eigen/Dense>
#include <torch/torch.h>

class Camera;
class GaussianModel;

namespace pose_opt
{

struct PoseFilterParams
{
    int max_iterations = 5;
    double img_point_cov = 100.0;
    double init_cov = 0.01;
    double conv_thresh_rot = 0.001;  // Degrees.
    double conv_thresh_pos = 0.001;  // Centimeters.
    int min_valid_pixels = 256;
    double alpha_threshold = 0.2;
    double huber_delta = 0.05;
    bool rerender_each_iter = true;
    double max_step_rot_deg = 0.3;
    double max_step_trans_cm = 3.0;
    double rmse_increase_tolerance_ratio = 0.01;
    double min_alpha_coverage_ratio = 0.005;
    int alpha_erode_radius = 1;  // Pixels.
};

struct PoseFilterResult
{
    bool success = false;
    Eigen::Matrix3d R_cw = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_cw = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 6, 6> post_cov = Eigen::Matrix<double, 6, 6>::Identity();
    int iterations = 0;
    int valid_pixels = 0;
    double rmse = 0.0;
    double alpha_coverage_ratio = 0.0;
    torch::Tensor valid_mask;
    torch::Tensor residual_rgb;
    bool rejected_by_guard = false;
    std::string guard_reason;
    bool step_was_clamped = false;
};

Eigen::Matrix3d Exp(const Eigen::Vector3d& ang);

Eigen::Vector3d Log(const Eigen::Matrix3d& R);

Eigen::Matrix3d skew(const Eigen::Vector3d& v);

// Uses camera prediction fields as the prior and returns a refined pose estimate.
PoseFilterResult refineCurrentFramePose(
    const std::shared_ptr<Camera>& viewpoint_camera,
    const std::shared_ptr<GaussianModel>& pc,
    const PoseFilterParams& params
);

}  // namespace pose_opt
