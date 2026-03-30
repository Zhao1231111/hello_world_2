/*
 * 当前帧 2DGS-IEKF 位姿优化器
 *
 * 该模块只负责 6 维 SE(3) 外层滤波求解：
 * 1. 由 renderer_2d 输出当前帧的 JTJ / JTr
 * 2. 结合位姿先验协方差进行 IEKF / Gauss-Newton 更新
 * 3. 将后验位姿与后验协方差写回 Camera
 *
 * 注意：
 * - 这里不再使用 OpenCV 灰度 patch，也不再假设视觉测量来自外部点云重投影。
 * - 所有残差均来自 2DGS 的 RGB 渲染图与当前帧图像的差异。
 */

#pragma once

#include <memory>
#include <string>
#include <Eigen/Dense>
#include <torch/torch.h>

class Camera;
class GaussianModel;

namespace pose_opt
{

/**
 * @brief 当前帧位姿滤波配置
 */
struct PoseFilterParams
{
    int max_iterations = 5;            // IEKF 最大迭代次数
    double img_point_cov = 100.0;      // 每个 RGB 通道的测量方差 sigma^2
    double init_cov = 0.01;            // 初始位姿协方差，用于新关键帧
    double conv_thresh_rot = 0.001;    // 收敛阈值，单位为度
    double conv_thresh_pos = 0.001;    // 收敛阈值，单位为厘米
    int min_valid_pixels = 256;        // 最少有效像素数
    double alpha_threshold = 0.2;      // 位姿观测 alpha 阈值
    double huber_delta = 0.05;         // Huber 阈值
    bool rerender_each_iter = true;    // 每轮位姿更新后是否重新渲染并线性化
    double max_step_rot_deg = 0.3;     // 单轮位姿更新允许的最大旋转步长，单位度
    double max_step_trans_cm = 3.0;    // 单轮位姿更新允许的最大平移步长，单位厘米
    double rmse_increase_tolerance_ratio = 0.01;  // 候选解 RMSE 允许相对恶化比例
    double min_alpha_coverage_ratio = 0.005;      // 位姿优化要求的最小 alpha 覆盖率
    int alpha_erode_radius = 1;        // 对 alpha 掩码做腐蚀时的半径，单位像素
};

/**
 * @brief 当前帧位姿滤波结果
 */
struct PoseFilterResult
{
    bool success = false;
    Eigen::Matrix3d R_cw = Eigen::Matrix3d::Identity();   // 后验旋转
    Eigen::Vector3d t_cw = Eigen::Vector3d::Zero();       // 后验平移
    Eigen::Matrix<double, 6, 6> post_cov = Eigen::Matrix<double, 6, 6>::Identity(); // 后验协方差
    int iterations = 0;             // 实际执行的迭代轮数
    int valid_pixels = 0;           // 最终线性化使用的有效像素数
    double rmse = 0.0;              // 最终线性化的加权残差 RMSE
    double alpha_coverage_ratio = 0.0;  // 最终接受位姿对应的 alpha 有效覆盖率
    torch::Tensor valid_mask;       // 当前帧有效像素掩码
    torch::Tensor residual_rgb;     // 当前帧残差图
    bool rejected_by_guard = false; // 是否因为稳定性保护而提前结束
    std::string guard_reason;       // 保护触发原因
    bool step_was_clamped = false;  // 本帧是否出现过步长裁剪
};

/**
 * @brief so(3) 指数映射
 */
Eigen::Matrix3d Exp(const Eigen::Vector3d& ang);

/**
 * @brief SO(3) 对数映射
 */
Eigen::Vector3d Log(const Eigen::Matrix3d& R);

/**
 * @brief 反对称矩阵
 */
Eigen::Matrix3d skew(const Eigen::Vector3d& v);

/**
 * @brief 对当前帧执行一次 6 维 IEKF 位姿更新
 *
 * 输入相机中的 `R_cw_pred_ / t_cw_pred_ / pose_cov_` 被视为本轮滤波的先验；
 * 若成功，将返回后验位姿与后验协方差，但不会在函数内部直接修改传入相机。
 */
PoseFilterResult refineCurrentFramePose(
    const std::shared_ptr<Camera>& viewpoint_camera,
    const std::shared_ptr<GaussianModel>& pc,
    const PoseFilterParams& params
);

}  // namespace pose_opt
