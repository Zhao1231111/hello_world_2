/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/torch.h>
#include <cstdio>
#include <tuple>
#include <string>
	
/**
 * @brief CUDA 高斯光栅化前向传播函数
 * 
 * 接收一组3D高斯球的参数和相机信息，将它们渲染成一张2D图像。
 * 
 * @param background 背景颜色张量 (3)
 * @param means3D 高斯球中心的世界坐标 (N, 3)
 * @param colors 预计算的颜色 (N, 3)，如果提供了SH系数则此项可忽略
 * @param opacity 高斯球的不透明度 (N, 1)
 * @param scales 高斯球的缩放因子 (N, 3)
 * @param rotations 高斯球的旋转四元数 (N, 4)
 * @param scale_modifier 全局缩放系数 (float)
 * @param transMat_precomp 预计算的变换矩阵 (N, 9)
 * @param viewmatrix 视图矩阵 (4, 4)
 * @param projmatrix 投影矩阵 (4, 4)
 * @param tan_fovx x方向视场角的正切值 (float)
 * @param tan_fovy y方向视场角的正切值 (float)
 * @param image_height 图像高度 (int)
 * @param image_width 图像宽度 (int)
 * @param sh 球谐函数(SH)系数 (N, K, 3)，用于计算视角相关颜色
 * @param degree SH阶数 (int)
 * @param campos 相机中心的世界坐标 (3)
 * @param prefiltered 是否启用预过滤 (bool)
 * @param debug 是否启用调试模式 (bool)
 * 
 * @return std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 * 返回一个元组，包含：
 * - (int) 光栅化的高斯点数量
 * - (Tensor) out_color: 渲染出的图像 (C, H, W)
 * - (Tensor) out_all: 包含深度、alpha等信息的完整渲染缓冲区
 * - (Tensor) radii: 每个高斯点在屏幕空间的半径 (N)
 * - (Tensor) geom_buffer: 用于反向传播的几何信息
 * - (Tensor) binning_buffer: 用于反向传播的分箱信息
 * - (Tensor) img_buffer: 用于反向传播的图像信息
 */
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool use_tile_culling,
	const bool debug);

/**
 * @brief CUDA 高斯光栅化反向传播函数
 * 
 * 计算前向传播过程中所有可学习参数的梯度。
 * 
 * @param background 背景颜色张量 (3)
 * @param means3D 高斯球中心的世界坐标 (N, 3)
 * @param radii 每个高斯点在屏幕空间的半径 (N)，由前向传播返回
 * @param colors 预计算的颜色 (N, 3)
 * @param scales 高斯球的缩放因子 (N, 3)
 * @param rotations 高斯球的旋转四元数 (N, 4)
 * @param scale_modifier 全局缩放系数 (float)
 * @param transMat_precomp 预计算的变换矩阵 (N, 9)
 * @param viewmatrix 视图矩阵 (4, 4)
 * @param projmatrix 投影矩阵 (4, 4)
 * @param tan_fovx x方向视场角的正切值 (float)
 * @param tan_fovy y方向视场角的正切值 (float)
 * @param dL_dout_color 渲染图像的颜色梯度 (C, H, W)
 * @param dL_dout_others 其他渲染缓冲区的梯度 (C, H, W)
 * @param sh 球谐函数(SH)系数 (N, K, 3)
 * @param degree SH阶数 (int)
 * @param campos 相机中心的世界坐标 (3)
 * @param geomBuffer 从前向传播获取的几何信息
 * @param R 瓦片大小 (int)
 * @param binningBuffer 从前向传播获取的分箱信息
 * @param imageBuffer 从前向传播获取的图像信息
 * @param debug 是否启用调试模式 (bool)
 * 
 * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 * 返回一个元组，包含以下参数的梯度：
 * - dL_dmeans3D: 3D位置的梯度
 * - dL_dcolors: 颜色的梯度
 * - dL_dopacity: 不透明度的梯度
 * - dL_dscales: 缩放的梯度
 * - dL_drotations: 旋转的梯度
 * - dL_dsh: SH系数的梯度
 * - dL_dmeans2D: 2D位置的梯度
 * - dL_dtransMat: 变换矩阵的梯度
 * - dL_dnormal: 法线的梯度 (新增)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
	 const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug);
		
/**
 * @brief Adam optimizer update step (CUDA)
 */
void adamUpdate(
    torch::Tensor &param,
    torch::Tensor &param_grad,
    torch::Tensor &exp_avg,
    torch::Tensor &exp_avg_sq,
    torch::Tensor &visible,
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N,
    const uint32_t M);

/**
 * @brief Mark visible points (CUDA)
 * @param means3D 高斯球中心的世界坐标 (N, 3)
 * @param viewmatrix 视图矩阵 (4, 4)
 * @param projmatrix 投影矩阵 (4, 4)
 * @return torch::Tensor 一个布尔张量 (N)，标记每个点是否可见
 */
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);
