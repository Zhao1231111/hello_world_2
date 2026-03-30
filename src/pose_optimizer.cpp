/*
 * 当前帧 2DGS-IEKF 位姿优化器实现
 */

#include "pose_optimizer.h"

#include "camera.h"
#include "gaussian.h"
#include "diff_surfel_rasterization_2d/renderer_2d.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{

/**
 * @brief 将 6x6 Torch 张量转换为 Eigen 矩阵
 */
Eigen::Matrix<double, 6, 6> tensorToEigenMatrix6(const torch::Tensor& tensor)
{
    Eigen::Matrix<double, 6, 6> matrix = Eigen::Matrix<double, 6, 6>::Zero();
    auto tensor_cpu = tensor.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64)).contiguous();
    const auto* ptr = tensor_cpu.data_ptr<double>();
    for (int row = 0; row < 6; ++row)
    {
        for (int col = 0; col < 6; ++col)
        {
            matrix(row, col) = ptr[row * 6 + col];
        }
    }
    return matrix;
}

/**
 * @brief 将 6 维 Torch 张量转换为 Eigen 向量
 */
Eigen::Matrix<double, 6, 1> tensorToEigenVector6(const torch::Tensor& tensor)
{
    Eigen::Matrix<double, 6, 1> vector = Eigen::Matrix<double, 6, 1>::Zero();
    auto tensor_cpu = tensor.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64)).contiguous();
    const auto* ptr = tensor_cpu.data_ptr<double>();
    for (int row = 0; row < 6; ++row)
    {
        vector(row) = ptr[row];
    }
    return vector;
}

/**
 * @brief 计算“先验减当前估计”的 6 维误差，用于先验约束项
 *
 * [公式] eta = (T_pred boxminus T_current)
 */
Eigen::Matrix<double, 6, 1> computePriorMinusCurrent(
    const std::shared_ptr<Camera>& viewpoint_camera,
    const Eigen::Matrix3d& current_R_cw,
    const Eigen::Vector3d& current_t_cw)
{
    Eigen::Matrix<double, 6, 1> eta = Eigen::Matrix<double, 6, 1>::Zero();
    eta.block<3, 1>(0, 0) = pose_opt::Log(viewpoint_camera->R_cw_pred_ * current_R_cw.transpose());
    eta.block<3, 1>(3, 0) = viewpoint_camera->t_cw_pred_ - current_t_cw;
    return eta;
}

/**
 * @brief 对协方差做最小正则化，防止求逆时数值退化
 */
Eigen::Matrix<double, 6, 6> regularizeCovariance(
    const Eigen::Matrix<double, 6, 6>& covariance,
    double init_cov)
{
    Eigen::Matrix<double, 6, 6> regularized = covariance;
    if (!regularized.allFinite())
    {
        regularized = Eigen::Matrix<double, 6, 6>::Identity() * init_cov;
    }

    regularized += Eigen::Matrix<double, 6, 6>::Identity() * 1e-9;
    return regularized;
}

/**
 * @brief 构造当前帧滤波使用的背景颜色
 */
torch::Tensor buildPoseBackground(const std::shared_ptr<GaussianModel>& pc)
{
    if (pc->white_background_)
    {
        return torch::ones({3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    }
    return torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
}

/**
 * @brief 对单轮位姿增量做旋转/平移步长裁剪，避免一次更新过大直接跳飞
 */
bool clampDeltaXi(
    Eigen::Matrix<double, 6, 1>& delta_xi,
    double max_step_rot_deg,
    double max_step_trans_cm)
{
    bool clamped = false;
    constexpr double kPi = 3.14159265358979323846;

    const double max_rot_rad = max_step_rot_deg > 0.0
                                   ? max_step_rot_deg * kPi / 180.0
                                   : std::numeric_limits<double>::infinity();
    const double max_trans_m = max_step_trans_cm > 0.0
                                   ? max_step_trans_cm / 100.0
                                   : std::numeric_limits<double>::infinity();

    const double rot_norm = delta_xi.block<3, 1>(0, 0).norm();
    if (rot_norm > max_rot_rad && rot_norm > 1e-12)
    {
        delta_xi.block<3, 1>(0, 0) *= (max_rot_rad / rot_norm);
        clamped = true;
    }

    const double trans_norm = delta_xi.block<3, 1>(3, 0).norm();
    if (trans_norm > max_trans_m && trans_norm > 1e-12)
    {
        delta_xi.block<3, 1>(3, 0) *= (max_trans_m / trans_norm);
        clamped = true;
    }

    return clamped;
}

}  // namespace

namespace pose_opt
{

Eigen::Matrix3d skew(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d m;
    m << 0.0, -v(2), v(1),
         v(2), 0.0, -v(0),
        -v(1), v(0), 0.0;
    return m;
}

Eigen::Matrix3d Exp(const Eigen::Vector3d& ang)
{
    const double ang_norm = ang.norm();
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    if (ang_norm < 1e-12)
    {
        return I + skew(ang);
    }

    const Eigen::Matrix3d K = skew(ang / ang_norm);
    return I + std::sin(ang_norm) * K + (1.0 - std::cos(ang_norm)) * K * K;
}

Eigen::Vector3d Log(const Eigen::Matrix3d& R)
{
    const double cos_theta = std::clamp(0.5 * (R.trace() - 1.0), -1.0, 1.0);
    const double theta = std::acos(cos_theta);
    const Eigen::Vector3d vee(R(2, 1) - R(1, 2),
                              R(0, 2) - R(2, 0),
                              R(1, 0) - R(0, 1));
    if (theta < 1e-12)
    {
        return 0.5 * vee;
    }
    return 0.5 * theta / std::sin(theta) * vee;
}

PoseFilterResult refineCurrentFramePose(
    const std::shared_ptr<Camera>& viewpoint_camera,
    const std::shared_ptr<GaussianModel>& pc,
    const PoseFilterParams& params)
{
    PoseFilterResult result;
    result.R_cw = viewpoint_camera->R_cw_pred_;
    result.t_cw = viewpoint_camera->t_cw_pred_;
    result.post_cov = regularizeCovariance(viewpoint_camera->pose_cov_, params.init_cov);

    if (!pc || !pc->is_init_ || pc->getXYZ().size(0) <= 0)
    {
        return result;
    }

    auto background = buildPoseBackground(pc);
    auto working_camera = std::make_shared<Camera>(*viewpoint_camera);
    working_camera->setPoseFromCw(viewpoint_camera->R_cw_pred_, viewpoint_camera->t_cw_pred_);

    const Eigen::Matrix<double, 6, 6> prior_cov = regularizeCovariance(viewpoint_camera->pose_cov_, params.init_cov);
    const Eigen::Matrix<double, 6, 6> prior_info = prior_cov.inverse();

    Eigen::Matrix<double, 6, 6> last_system = prior_info;
    bool accepted_once = false;

    PoseLinearizationSettings2D linearization_settings;
    linearization_settings.alpha_threshold = params.alpha_threshold;
    linearization_settings.huber_delta = params.huber_delta;
    linearization_settings.alpha_erode_radius = params.alpha_erode_radius;

    auto fillResultFromLinearization = [&result](const PoseLinearization2D& linearization)
    {
        result.valid_pixels = linearization.valid_pixels;
        result.rmse = linearization.rmse;
        result.alpha_coverage_ratio = linearization.alpha_coverage_ratio;
        result.valid_mask = linearization.valid_mask;
        result.residual_rgb = linearization.residual_rgb;
    };

    auto evaluateCamera = [&](const std::shared_ptr<Camera>& camera) -> PoseLinearization2D
    {
        auto render_pkg = render_2d(
            camera,
            pc,
            background,
            1.0f,
            false,
            false,
            torch::Tensor(),
            true);

        return linearize_pose_2d(
            camera,
            pc,
            camera->original_image_,
            background,
            render_pkg,
            linearization_settings);
    };

    const int max_linearization_rounds = params.rerender_each_iter ? params.max_iterations : std::min(params.max_iterations, 1);
    for (int iter = 0; iter < max_linearization_rounds; ++iter)
    {
        auto current_linearization = evaluateCamera(working_camera);
        result.iterations = iter + 1;
        fillResultFromLinearization(current_linearization);

        if (current_linearization.valid_pixels < params.min_valid_pixels)
        {
            result.rejected_by_guard = true;
            result.guard_reason = "insufficient_valid_pixels";
            break;
        }

        if (current_linearization.alpha_coverage_ratio < params.min_alpha_coverage_ratio)
        {
            result.rejected_by_guard = true;
            result.guard_reason = "insufficient_alpha_coverage";
            break;
        }

        const Eigen::Matrix<double, 6, 6> jtj =
            tensorToEigenMatrix6(current_linearization.jtj) / params.img_point_cov;
        const Eigen::Matrix<double, 6, 1> jtr =
            tensorToEigenVector6(current_linearization.jtr) / params.img_point_cov;
        const Eigen::Matrix<double, 6, 1> eta =
            computePriorMinusCurrent(viewpoint_camera, working_camera->R_cw_, working_camera->t_cw_);

        // [公式] A = P^{-1} + JTJ,  b = P^{-1} * eta + JTr
        const Eigen::Matrix<double, 6, 6> system = prior_info + jtj;
        const Eigen::Matrix<double, 6, 1> rhs = prior_info * eta + jtr;

        Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(system);
        if (ldlt.info() != Eigen::Success)
        {
            result.rejected_by_guard = true;
            result.guard_reason = "ill_conditioned_system";
            break;
        }

        Eigen::Matrix<double, 6, 1> delta_xi = ldlt.solve(rhs);
        if (ldlt.info() != Eigen::Success || !delta_xi.allFinite())
        {
            result.rejected_by_guard = true;
            result.guard_reason = "nan_delta";
            break;
        }

        if (clampDeltaXi(delta_xi, params.max_step_rot_deg, params.max_step_trans_cm))
        {
            result.step_was_clamped = true;
        }

        // [公式] T_{cw}^{+} = Exp(delta_xi^\wedge) T_{cw}
        const Eigen::Matrix3d dR = Exp(delta_xi.block<3, 1>(0, 0));
        auto candidate_camera = std::make_shared<Camera>(*working_camera);
        const Eigen::Matrix3d next_R_cw = dR * candidate_camera->R_cw_;
        const Eigen::Vector3d next_t_cw = dR * candidate_camera->t_cw_ + delta_xi.block<3, 1>(3, 0);
        candidate_camera->setPoseFromCw(next_R_cw, next_t_cw);

        auto candidate_linearization = evaluateCamera(candidate_camera);
        if (candidate_linearization.valid_pixels < params.min_valid_pixels)
        {
            result.rejected_by_guard = true;
            result.guard_reason = "insufficient_valid_pixels";
            break;
        }

        if (candidate_linearization.alpha_coverage_ratio < params.min_alpha_coverage_ratio)
        {
            result.rejected_by_guard = true;
            result.guard_reason = "insufficient_alpha_coverage";
            break;
        }

        const double current_rmse_guard = std::max(current_linearization.rmse, 1e-9);
        const double candidate_rmse_guard = candidate_linearization.rmse;
        if (candidate_rmse_guard > current_rmse_guard * (1.0 + params.rmse_increase_tolerance_ratio))
        {
            result.rejected_by_guard = true;
            result.guard_reason = "rmse_increase";
            break;
        }

        working_camera = candidate_camera;
        accepted_once = true;
        last_system = system;
        fillResultFromLinearization(candidate_linearization);

        const double delta_rot_deg = delta_xi.block<3, 1>(0, 0).norm() * 57.29577951308232;
        const double delta_pos_cm = delta_xi.block<3, 1>(3, 0).norm() * 100.0;
        if (delta_rot_deg < params.conv_thresh_rot && delta_pos_cm < params.conv_thresh_pos)
        {
            break;
        }

    }

    if (!accepted_once)
    {
        return result;
    }

    result.success = true;
    result.R_cw = working_camera->R_cw_;
    result.t_cw = working_camera->t_cw_;
    result.post_cov = last_system.inverse();
    return result;
}

}  // namespace pose_opt
