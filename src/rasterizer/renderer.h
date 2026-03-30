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

#pragma once

#include <tuple>
#include <torch/torch.h>

#include "camera.h"      // 相机类定义
#include "gaussian.h"    // 高斯模型类定义
#include "rasterizer.h"  // 光栅化器类定义

// 前向声明，避免包含整个头文件
class GaussianModel;

/**
 * @brief 高斯点云渲染核心函数
 *
 * 这是Gaussian Splatting渲染系统的核心接口函数，负责将3D高斯点云
 * 从任意视角渲染成2D图像。该函数封装了完整的渲染流程：
 * 1. 相机参数设置和视场角计算
 * 2. 光栅化器配置和初始化
 * 3. 高斯点参数准备
 * 4. CUDA加速的光栅化渲染
 * 5. 结果打包和返回
 *
 * 该函数是可微分的，支持PyTorch的自动梯度计算，
 * 是训练过程中前向传播的核心组件。
 *
 * @param viewpoint_camera 相机对象，包含内参、外参和图像尺寸等信息
 * @param pc 高斯模型指针，包含所有待渲染的高斯点参数
 * @param bg_color 背景颜色张量，形状为(3,)，RGB值范围[0,1]
 * @param use_trained_exposure 是否使用训练好的曝光参数（预留接口，当前版本未使用）
 * @param no_color 调试选项，设为true时跳过颜色计算，只计算几何信息
 * @param scaling_modifier 缩放因子修改器，乘以所有高斯点的缩放参数，用于控制渲染尺度
 *
 * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 *         返回包含5个元素的元组：
 *         0: rendered_image     - 最终渲染的RGB图像，形状(3, H, W)
 *         1: rendered_final_T   - 最终透射率图，形状(H, W)，每个像素的累积透明度
 *         2: screenspace_points - 投影后的2D点坐标，形状(N, 3)，用于梯度计算
 *         3: visibility_mask    - 可见性掩码，形状(N,)，布尔值表示每个高斯点是否可见
 *         4: radii              - 投影半径，形状(N,)，用于可见性和LOD计算
 *
 * @note 该函数是线程安全的，可以在多线程训练中使用
 * @note 所有输出张量都在CUDA设备上，需要手动转移到CPU进行可视化
 * @note 函数内部使用CUDA加速，调用前确保CUDA上下文已初始化
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
render(const std::shared_ptr<Camera>& viewpoint_camera,
       std::shared_ptr<GaussianModel> pc,
       torch::Tensor& bg_color,
       bool use_trained_exposure = false,
       bool no_color = false,
       float scaling_modifier = 1.0);