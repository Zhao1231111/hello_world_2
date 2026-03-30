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

/**
 * @file forward.h
 * @brief 高斯散射(Gaussian Splatting)前向传播头文件
 *
 * 该文件定义了CUDA光栅化器中的前向传播相关函数和常量，包括：
 * - 预处理函数：将3D高斯点投影到2D并计算渲染参数
 * - 渲染函数：执行基于tile的光栅化渲染
 * - 辅助函数：计算不透明度贡献和tile剔除优化
 * - 相关的宏定义和常量
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <functional>

// 顺序处理tile的阈值，小于此值的tile将按顺序处理而不是并行处理
#define SEQUENTIAL_TILE_THRESH (32)
// 不透明度阈值，用于剔除不透明度过低的高斯点
#define OPACITY_THRESHOLD (1.f / 255.f)
// CUDA warp大小，通常为32个线程
#define WARP_SIZE (32)
// warp掩码，用于warp级别的同步操作
constexpr uint32_t WARP_MASK = 0xFFFFFFFFU;

/**
 * @brief 评估高斯点对像素的不透明度贡献因子
 * @param dx 像素x坐标与高斯点中心的x距离
 * @param dy 像素y坐标与高斯点中心的y距离
 * @param co 圆锥矩阵和不透明度参数 (xx, xy, yy, opacity)
 * @return 高斯分布的指数部分：0.5 * x^T * Σ⁻¹ * x
 */
__device__ inline float evaluate_opacity_factor(const float dx, const float dy, const float4 co)
{
	return 0.5f * (co.x * dx * dx + co.z * dy * dy) + co.y * dx * dy;
}

/**
 * @brief 计算高斯点在矩形tile区域内的最大贡献值
 * 该函数用于tile剔除优化，计算高斯点对tile的最大可能贡献
 * @param co 圆锥矩阵和不透明度参数 (xx, xy, yy, opacity)
 * @param mean 高斯点在图像平面上的位置 (x, y)
 * @param rect_min tile的左上角坐标
 * @param rect_max tile的右下角坐标
 * @param max_pos 输出：tile内贡献最大的位置
 * @return tile内高斯分布的最大贡献值（对数形式）
 */
__device__ inline float max_contrib_power_rect_gaussian_float(const float4 co,  const float2 mean, const glm::vec2 rect_min, const glm::vec2 rect_max, glm::vec2& max_pos)
{
	// 计算高斯点中心与矩形左边界的x距离，如果为正表示高斯点在矩形左侧
	const float x_min_diff = rect_min.x - mean.x;
	// 判断高斯点是否在矩形左侧（x_left = 1表示在左侧，0表示在右侧或内部）
	const float x_left = x_min_diff > 0.0f;
	// 另一种等价的判断方式：mean.x < rect_min.x
	// const float x_left = mean.x < rect_min.x;
	// 检查x方向是否超出范围：如果在左侧或右侧，not_in_x_range > 0
	const float not_in_x_range = x_left + (mean.x > rect_max.x);

	// 计算高斯点中心与矩形上边界的y距离，如果为正表示高斯点在矩形上方
	const float y_min_diff = rect_min.y - mean.y;
	// 判断高斯点是否在矩形上方（y_above = 1表示在上方，0表示在下方或内部）
	const float y_above =  y_min_diff > 0.0f;
	// 另一种等价的判断方式：mean.y < rect_min.y
	// const float y_above = mean.y < rect_min.y;
	// 检查y方向是否超出范围：如果在上方或下方，not_in_y_range > 0
	const float not_in_y_range = y_above + (mean.y > rect_max.y);

	// 默认最大贡献位置为高斯点中心
	max_pos = {mean.x, mean.y};
	// 默认最大贡献值为0
	float max_contrib_power = 0.0f;
	// 计算矩形tile的尺寸
	glm::vec2 size = {rect_max.x - rect_min.x, rect_max.y - rect_min.y};

	// 如果高斯点完全在矩形内，最大贡献位置就是高斯点中心，贡献值为0（已在上面初始化）
	// 如果高斯点部分或完全超出矩形范围，需要在矩形边界上找到最大贡献点
	if ((not_in_y_range + not_in_x_range) > 0.0f)
	{
		// 根据高斯点位置选择最近的矩形边界点作为起始搜索点
		// 如果在左侧(x_left=1)，选择左边界；否则选择右边界
		const float px = x_left * rect_min.x + (1.0f - x_left) * rect_max.x;
		// 如果在上方(y_above=1)，选择上边界；否则选择下边界
		const float py = y_above * rect_min.y + (1.0f - y_above) * rect_max.y;

		// 计算搜索方向：符号与距离一致，确保向矩形内部搜索
		const float dx = copysign(float(size.x), x_min_diff);
		const float dy = copysign(float(size.y), y_min_diff);

		// 计算从边界点到高斯点中心的向量
		const float diffx = mean.x - px;
		const float diffy = mean.y - py;

		// 计算归一化因子（使用CUDA快速倒数指令优化性能）
		// rcp_dxdxcox = 1.0 / (搜索范围宽度² × 圆锥矩阵xx分量)
		const float rcp_dxdxcox = __frcp_rn(size.x * size.x * co.x); // = 1.0 / (dx*dx*co.x)
		// rcp_dydycoz = 1.0 / (搜索范围高度² × 圆锥矩阵yy分量)
		const float rcp_dydycoz = __frcp_rn(size.y * size.y * co.z); // = 1.0 / (dy*dy*co.z)

		// 在边界上搜索最大贡献点的位置参数tx（限制在[0,1]范围内）
		// tx只在y方向超出范围时有效，否则为0
		const float tx = not_in_y_range * __saturatef((dx * co.x * diffx + dx * co.y * diffy) * rcp_dxdxcox);
		// 在边界上搜索最大贡献点的位置参数ty（限制在[0,1]范围内）
		// ty只在x方向超出范围时有效，否则为0
		const float ty = not_in_x_range * __saturatef((dy * co.y * diffx + dy * co.z * diffy) * rcp_dydycoz);

		// 计算最终的最大贡献位置：边界起始点 + 参数 × 搜索方向
		max_pos = {px + tx * dx, py + ty * dy};

		// 计算最大贡献位置相对于高斯点中心的偏移向量
		const float2 max_pos_diff = {mean.x - max_pos.x, mean.y - max_pos.y};
		// 计算该位置的高斯分布贡献值（指数部分的0.5倍）
		max_contrib_power = evaluate_opacity_factor(max_pos_diff.x, max_pos_diff.y, co);
	}

	// 返回最大贡献值（对数形式，用于tile剔除判断）
	return max_contrib_power;
}

/**
 * @brief FORWARD命名空间
 * 包含高斯散射(Gaussian Splatting)前向传播相关的CUDA函数声明
 * 主要包括预处理和渲染两大功能模块
 */
namespace FORWARD
{
	/**
	 * @brief 为每个高斯点执行光栅化前的预处理步骤
	 * 该函数将3D高斯点投影到2D图像平面，计算协方差矩阵、颜色等渲染参数
	 */
	/**
	 * @param P 高斯点总数
	 * @param D 球谐函数的度数
	 * @param M 球谐函数的最大系数数量
	 * @param orig_points 原始3D高斯点位置数组
	 * @param scales 高斯点缩放参数数组
	 * @param scale_modifier 全局缩放修正因子
	 * @param rotations 高斯点旋转参数（四元数）
	 * @param opacities 高斯点不透明度数组
	 * @param dc 直流颜色分量
	 * @param shs 球谐函数系数
	 * @param clamped 颜色值截断标记数组
	 * @param cov3D_precomp 预计算的3D协方差矩阵（可选）
	 * @param colors_precomp 预计算的颜色（可选）
	 * @param viewmatrix 视图变换矩阵
	 * @param projmatrix 投影变换矩阵
	 * @param cam_pos 相机位置
	 * @param W 图像宽度
	 * @param H 图像高度
	 * @param focal_x x方向焦距
	 * @param focal_y y方向焦距
	 * @param tan_fovx x方向视场角正切值
	 * @param tan_fovy y方向视场角正切值
	 * @param limx_neg x方向边界负限
	 * @param limx_pos x方向边界正限
	 * @param limy_neg y方向边界负限
	 * @param limy_pos y方向边界正限
	 * @param radii 输出：高斯点的半径数组
	 * @param points_xy_image 输出：高斯点在图像平面上的位置
	 * @param depths 输出：高斯点的深度值
	 * @param cov3Ds 输出：3D协方差矩阵
	 * @param colors 输出：高斯点的RGB颜色
	 * @param conic_opacity 输出：圆锥矩阵和不透明度
	 * @param grid tile网格尺寸
	 * @param tiles_touched 输出：每个高斯点影响的tile数量
	 * @param prefiltered 是否已经预过滤
	 * @param no_color 是否跳过颜色计算
	 */
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* dc,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const float limx_neg,
		const float limx_pos,
		const float limy_neg,
		const float limy_pos,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		bool no_color);

	/**
	 * @brief 主要的栅格化渲染方法
	 * 该函数基于tile的光栅化，对每个tile执行alpha混合，将高斯点按深度排序渲染到图像上
	 * @param grid CUDA网格维度
	 * @param block CUDA块维度
	 * @param ranges 每个tile中高斯点范围 [start, end)
	 * @param point_list 高斯点列表（按tile分组）
	 * @param per_tile_bucket_offset 每个tile的bucket偏移量
	 * @param bucket_to_tile bucket到tile的映射
	 * @param sampled_T 采样透明度T值（用于反向传播）
	 * @param sampled_ar 采样累积颜色（用于反向传播）
	 * @param W 图像宽度
	 * @param H 图像高度
	 * @param points_xy_image 高斯点在图像平面上的位置
	 * @param depth 高斯点深度值
	 * @param features 高斯点颜色特征
	 * @param conic_opacity 圆锥矩阵和不透明度
	 * @param n_contrib 每个像素贡献的高斯点数量
	 * @param max_contrib tile中的最大贡献者数量
	 * @param bg_color 背景颜色
	 * @param out_color 输出颜色缓冲区
	 * @param out_final_T 输出最终透明度T
	 * @param no_color 是否跳过颜色渲染
	 */
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const uint32_t* per_tile_bucket_offset, uint32_t* bucket_to_tile,
		float* sampled_T, float* sampled_ar,
		int W, int H,
		const float2* points_xy_image,
		const float* depth,
		const float* features,
		const float4* conic_opacity,
		uint32_t* n_contrib,
		uint32_t* max_contrib,
		const float* bg_color,
		float* out_color,
		float* out_final_T,
		bool no_color);
}


#endif