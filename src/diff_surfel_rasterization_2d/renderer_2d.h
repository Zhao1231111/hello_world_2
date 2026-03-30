/*
 * Copyright (C) 2025
 * 2D Gaussian Splatting Renderer for Gaussian-LIC
 *
 * This file provides the high-level rendering interface for 2D Gaussian Splatting,
 * mirroring the structure of the 3DGS renderer in rasterizer/renderer.cpp.
 */

#pragma once

#include <tuple>
#include <memory>
#include <torch/torch.h>

// Forward declarations
class Camera;
class GaussianModel;

/**
 * @brief 2D高斯光栅化器的渲染结果
 * 
 * 这是一个数据结构，用于保存前向传播（渲染）的结果。
 * 它不仅包含最终的渲染图像，还包含了反向传播计算梯度时所需的中间变量，
 * 以及2DGS特有的深度、法线等辅助输出。
 */
struct RenderResult2D
{
    // === 主要输出 ===
    torch::Tensor rendered_image;     // 渲染的RGB图像 (3, H, W)
    torch::Tensor visibility_mask;    // 可见性掩码 (N,)，标记哪些高斯点可见
    
    // === 2DGS特有的辅助输出 ===
    torch::Tensor rendered_depth;     // 渲染的深度图 (H, W)
    torch::Tensor rendered_alpha;     // 渲染的Alpha图 (H, W)
    torch::Tensor rendered_normal;    // 渲染的法线图 (3, H, W)
    torch::Tensor rendered_distortion; // 畸变图，用于正则化 (H, W)
    torch::Tensor rendered_median_depth; // 中值深度图 (H, W)
    torch::Tensor surf_normal; // 伪表面法线图 (3, H, W)
    
    // === 用于反向传播的中间张量 ===
    torch::Tensor radii;              // 投影半径 (N,)
    torch::Tensor screenspace_points; // 屏幕空间点坐标 (N, 3)，用于梯度计算
    torch::Tensor screenspace_points_mask; // 可选：从局部子集散射回全局的掩码
    torch::Tensor screenspace_points_local; // 可选：局部子集的屏幕坐标（含 grad）

    
    // === 内部缓冲区（用于反向传播）===
    torch::Tensor geom_buffer;
    torch::Tensor binning_buffer;
    torch::Tensor img_buffer;
};

/**
 * @brief 当前帧位姿线性化设置
 *
 * 这里的扰动量用于在 2DGS 的 CUDA 渲染路径上构造数值 Jacobian，
 * 从而组装高斯-牛顿/IEKF 所需的 \f$\mathbf{J}^\top\mathbf{J}\f$ 与
 * \f$\mathbf{J}^\top\mathbf{r}\f$。
 */
struct PoseLinearizationSettings2D
{
    double rot_epsilon = 1e-4;          // 旋转扰动步长，单位 rad
    double trans_epsilon = 1e-4;        // 平移扰动步长，单位 m
    double alpha_threshold = 0.2;       // 只有 alpha 足够大的像素才参与位姿观测
    double huber_delta = 0.05;          // Huber 权重阈值
    double gradient_epsilon = 1e-5;     // 图像梯度掩码的最小阈值
    bool use_central_difference = true; // 是否使用中心差分提高线性化精度
    int alpha_erode_radius = 1;         // 对 alpha 掩码做腐蚀时的半径，去掉边界半透明区域
};

/**
 * @brief 当前帧位姿线性化结果
 *
 * `jtj` 与 `jtr` 会被外层 IEKF 直接转换为 6 维位姿增量的法方程。
 */
struct PoseLinearization2D
{
    torch::Tensor jtj;            // (6, 6) double，信息矩阵近似
    torch::Tensor jtr;            // (6) double，信息向量
    torch::Tensor valid_mask;     // (H, W) bool，参与位姿观测的像素掩码
    torch::Tensor residual_rgb;   // (3, H, W) float，当前帧 RGB 残差
    double rmse = 0.0;            // 有效像素上的加权 RMSE
    int valid_pixels = 0;         // 有效像素数量
    int alpha_pixels = 0;         // alpha 掩码中保留下来的像素数
    double alpha_coverage_ratio = 0.0;  // alpha 掩码像素占整张图的比例
};

/**
 * @brief 2D高斯点云渲染核心函数
 *
 * 这是2D Gaussian Splatting渲染系统的核心接口函数，负责将2D高斯点云
 * 从任意视角渲染成2D图像。与3DGS的主要区别：
 * 1. 使用2D缩放参数而非3D
 * 2. 使用变换矩阵(transMat)而非3D协方差
 * 3. 额外输出深度、法线、畸变等辅助图
 *
 * @param viewpoint_camera 相机对象，包含内参、外参和图像尺寸等信息
 * @param pc 高斯模型指针，包含所有待渲染的高斯点参数
 * @param bg_color 背景颜色张量，形状为(3,)，RGB值范围[0,1]
 * @param scaling_modifier 缩放因子修改器，用于控制渲染尺度，默认1.0
 *
 * @return RenderResult2D 包含渲染结果和中间变量的结构体
 */
RenderResult2D render_2d(
    const std::shared_ptr<Camera>& viewpoint_camera,
    const std::shared_ptr<GaussianModel>& pc,
    const torch::Tensor& bg_color,
    float scaling_modifier = 1.0f,
    bool use_tile_culling = true,
    bool debug_mode = false,
    const torch::Tensor& render_mask = torch::Tensor(),
    bool compute_extras = true
);

/**
 * @brief 基于当前 2DGS 渲染结果构造相机位姿线性化结果
 *
 * 该函数保持地图参数固定，通过对 6 维位姿做小扰动重新渲染，
 * 在 GPU 上直接组装当前帧的 \f$\mathbf{J}^\top\mathbf{J}\f$ 和
 * \f$\mathbf{J}^\top\mathbf{r}\f$，供外层 IEKF 更新使用。
 */
PoseLinearization2D linearize_pose_2d(
    const std::shared_ptr<Camera>& viewpoint_camera,
    const std::shared_ptr<GaussianModel>& pc,
    const torch::Tensor& gt_image,
    const torch::Tensor& bg_color,
    const RenderResult2D& base_render,
    const PoseLinearizationSettings2D& settings
);
