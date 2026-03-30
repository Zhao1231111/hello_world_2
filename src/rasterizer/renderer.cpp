/*
 * Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion
 * Copyright (C) 2025 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "renderer.h"

/**
 * @brief 高斯点云渲染主函数
 *
 * 将3D高斯点云从指定视角渲染成2D图像。这是Gaussian Splatting的核心渲染函数，
 * 负责将所有高斯分布投影到图像平面并进行alpha混合生成最终图像。
 *
 * @param viewpoint_camera 当前视角的相机参数
 * @param pc 高斯模型，包含所有高斯点的参数
 * @param bg_color 背景颜色 (R, G, B)
 * @param use_trained_exposure 是否使用训练的曝光参数
 * @param no_color 是否跳过颜色计算（用于梯度调试）
 * @param scaling_modifier 缩放因子修改器，用于控制高斯大小
 *
 * @return std::tuple 包含5个元素的元组：
 *         - rendered_image: 最终渲染的RGB图像 (3, H, W)
 *         - rendered_final_T: 最终的透射率图 (H, W)，表示每个像素的累积透明度
 *         - screenspace_points: 投影后的2D点坐标 (N, 3)，用于梯度计算
 *         - radii > 0: 可见性掩码 (N,)，标记哪些高斯点在当前视角可见
 *         - radii: 每个高斯点的投影半径 (N,)，用于可见性判断
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
render(const std::shared_ptr<Camera>& viewpoint_camera,
       std::shared_ptr<GaussianModel> pc,
       torch::Tensor& bg_color,
       bool use_trained_exposure,
       bool no_color,
       float scaling_modifier)
{
    // === 准备2D投影点 ===
    // 创建一个与3D点坐标同形状的张量，用于存储投影后的2D坐标
    // requires_grad=true 表示需要计算梯度，用于反向传播
    auto screenspace_points = torch::zeros_like(pc->getXYZ(),
        torch::TensorOptions().dtype(pc->getXYZ().dtype()).requires_grad(true).device(torch::kCUDA));

    // === 计算相机视场角参数 ===
    // 将弧度角转换为正切值，用于投影计算
    float tanfovx = std::tan(viewpoint_camera->FoVx_ * 0.5f);  // 水平视场角的一半正切值
    float tanfovy = std::tan(viewpoint_camera->FoVy_ * 0.5f);  // 垂直视场角的一半正切值

    // === 设置渲染配置 ===
    bool prefiltered = false;  // 是否预过滤（暂时未使用）
    bool debug = false;        // 调试模式

    // 配置光栅化器参数，定义渲染视口、变换矩阵等
    GaussianRasterizationSettings raster_settings(
        viewpoint_camera->image_height_,    // 图像高度
        viewpoint_camera->image_width_,     // 图像宽度
        tanfovx,                            // 水平视场角参数
        tanfovy,                            // 垂直视场角参数
        viewpoint_camera->limx_neg_,        // 视口x轴负边界
        viewpoint_camera->limx_pos_,        // 视口x轴正边界
        viewpoint_camera->limy_neg_,        // 视口y轴负边界
        viewpoint_camera->limy_pos_,        // 视口y轴正边界
        bg_color,                           // 背景颜色
        scaling_modifier,                   // 高斯缩放修改器
        viewpoint_camera->world_view_transform_,    // 世界到相机坐标变换
        viewpoint_camera->full_proj_transform_,     // 完整投影变换矩阵
        pc->sh_degree_,                     // 球谐函数阶数
        viewpoint_camera->camera_center_,   // 相机中心位置
        prefiltered,                        // 预过滤标志
        debug,                              // 调试标志
        no_color,                           // 无颜色标志
        pc->lambda_erank_                   // 排序正则化参数
    );

    // 创建光栅化器实例
    GaussianRasterizer rasterizer(raster_settings);

    // === 准备高斯点参数 ===
    // 获取所有高斯点的各种属性，这些都是可优化的参数
    auto means3D = pc->getXYZ();           // 3D位置坐标 (N, 3)
    auto means2D = screenspace_points;     // 2D投影坐标 (N, 3)，初始为零
    auto opacity = pc->getOpacity();       // 不透明度，经过sigmoid激活 (N, 1)，范围[0,1]
    auto scales = pc->getScaling();        // 缩放因子，经过exp激活 (N, 3)，范围[0,∞)
    auto rotations = pc->getRotation();    // 旋转四元数，经过归一化 (N, 4)
    torch::Tensor dc = pc->getFeaturesDc();       // DC颜色特征 (N, 1, 3)，球谐函数0阶系数
    torch::Tensor shs = pc->getFeaturesRest();    // 高阶颜色特征 (N, 15, 3)，球谐函数1-3阶系数
    torch::Tensor colors_precomp;          // 预计算颜色（可选，用于调试）
    torch::Tensor cov3D_precomp;           // 预计算3D协方差矩阵（可选）

    // === 执行光栅化渲染 ===
    // 调用CUDA光栅化器进行前向渲染，将3D高斯点渲染成2D图像
    auto rasterizer_result = rasterizer.forward(
        means3D,           // 3D位置
        means2D,           // 2D投影位置（输出）
        opacity,           // 不透明度
        dc,                // DC颜色特征
        shs,               // 高阶颜色特征
        colors_precomp,    // 预计算颜色
        scales,            // 缩放参数
        rotations,         // 旋转参数
        cov3D_precomp      // 预计算协方差
    );

    // === 提取渲染结果 ===
    auto rendered_image = std::get<0>(rasterizer_result);     // 最终RGB图像 (3, H, W)
    auto radii = std::get<1>(rasterizer_result);              // 投影半径 (N,)，用于可见性判断
    auto rendered_final_T = std::get<2>(rasterizer_result);   // 最终透射率 (H, W)

    // === 返回渲染结果 ===
    // 返回5个关键输出，供训练循环和梯度计算使用
    return std::make_tuple(
        rendered_image,        // 渲染的RGB图像
        rendered_final_T,      // 透射率图（用于梯度计算）
        screenspace_points,    // 投影后的2D点（用于梯度计算）
        radii > 0,             // 可见性掩码：半径>0表示可见
        radii                  // 原始半径值
    );
}