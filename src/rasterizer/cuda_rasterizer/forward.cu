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

#include "forward.h"
#include "auxiliary.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

/**
 * @brief 从球谐函数(Spherical Harmonics)计算高斯点的颜色
 * @param idx 高斯点的索引
 * @param deg 球谐函数的度数
 * @param max_coeffs 最大系数数量
 * @param means 3D高斯点位置数组
 * @param campos 相机位置
 * @param dc 直流分量(DC)颜色数据
 * @param shs 球谐函数系数
 * @param clamped 记录颜色值是否被截断的数组
 * @return 计算得到的RGB颜色
 * 
 * 与2d的区别：3d单独记录了直流分量，2d则将直流分量存储于shs[0]中
 */
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* dc, const float* shs, bool* clamped)
{
	// 计算从相机到高斯点的方向向量并归一化
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	// 获取直流颜色分量和球谐系数
	glm::vec3* direct_color = ((glm::vec3*)dc) + idx;
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// 球谐函数的0阶分量
	glm::vec3 result = SH_C0 * direct_color[0];

	// 1阶球谐函数
	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[0] + SH_C1 * z * sh[1] - SH_C1 * x * sh[2];

		// 2阶球谐函数
		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[3] +
				SH_C2[1] * yz * sh[4] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[5] +
				SH_C2[3] * xz * sh[6] +
				SH_C2[4] * (xx - yy) * sh[7];

			// 3阶球谐函数
			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[8] +
					SH_C3[1] * xy * z * sh[9] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[10] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[11] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[12] +
					SH_C3[5] * z * (xx - yy) * sh[13] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[14];
			}
		}
	}

	// TODO：这个偏移量有什么用？
	result += 0.5f;

	// 记录颜色值是否被截断
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

/**
 * @brief 计算高斯点在图像平面上的2D协方差矩阵
 * 通过将3D协方差矩阵投影到2D图像平面上，用于后续的光栅化
 * @param mean 3D高斯点位置
 * @param focal_x x方向焦距
 * @param focal_y y方向焦距
 * @param tan_fovx x方向视场角的正切值
 * @param tan_fovy y方向视场角的正切值
 * @param limx_neg x方向负边界
 * @param limx_pos x方向正边界
 * @param limy_neg y方向负边界
 * @param limy_pos y方向正边界
 * @param cov3D 3D协方差矩阵
 * @param viewmatrix 视图变换矩阵
 * @return 2D协方差矩阵的三个分量(xx, xy, yy)
 * 
 * 如果高斯中心在视场范围外，则将其篡改到视场范围内
 */
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy,
							   float limx_neg, float limx_pos, float limy_neg, float limy_pos, const float* cov3D, const float* viewmatrix)
{
	// 将3D点变换到视图坐标系
	float3 t = transformPoint4x3(mean, viewmatrix);

	// 注释掉的代码：原来的视场角限制方式
	// const float limx = 1.3f * tan_fovx;
	// const float limy = 1.3f * tan_fovy;
	// const float txtz = t.x / t.z;
	// const float tytz = t.y / t.z;
	// t.x = min(limx, max(-limx, txtz)) * t.z;
	// t.y = min(limy, max(-limy, tytz)) * t.z;

	// 篡改高斯中心在视图坐标系下的位置，限制在视场范围内
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx_pos, max(limx_neg, txtz)) * t.z;
	t.y = min(limy_pos, max(limy_neg, tytz)) * t.z;

	// 计算Jacobian矩阵（从视图坐标到屏幕坐标的变换）
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	// 从视图变换矩阵中提取旋转部分
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	// 计算变换矩阵T = W * J
	glm::mat3 T = W * J;

	// 构建3D协方差矩阵
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	// 计算投影后的2D协方差矩阵：cov = T^T * Vrk * T
	glm::mat3 cov = glm::transpose(T) * Vrk * T;

	// 添加正则化项以提高数值稳定性
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;

	// 返回2D协方差矩阵的上三角部分(xx, xy, yy)
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

/**
 * @brief 从高斯点的缩放和旋转参数计算3D协方差矩阵
 * 根据高斯分布的定义，协方差矩阵 = R * S * S^T * R^T，其中R是旋转矩阵，S是对角缩放矩阵
 * @param scale 高斯点的三个轴向缩放参数
 * @param mod 缩放修正因子
 * @param rot 四元数表示的旋转参数
 * @param cov3D 输出3D协方差矩阵（6个元素：xx, xy, xz, yy, yz, zz）
 */
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// 构建缩放矩阵S（对角矩阵）
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// 提取四元数分量
	glm::vec4 q = rot;
	float r = q.x;  // w分量
	float x = q.y;  // x分量
	float y = q.z;  // y分量
	float z = q.w;  // z分量

	// 从四元数构建旋转矩阵R
	// 使用标准四元数到旋转矩阵的转换公式
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	// 计算变换矩阵M = S * R
	glm::mat3 M = S * R;

	// 计算协方差矩阵Sigma = M^T * M
	// 这是因为高斯分布的协方差矩阵等于变换矩阵的平方
	glm::mat3 Sigma = glm::transpose(M) * M;

	// 存储协方差矩阵的上三角部分（对称矩阵）
	cov3D[0] = Sigma[0][0];  // xx
	cov3D[1] = Sigma[0][1];  // xy
	cov3D[2] = Sigma[0][2];  // xz
	cov3D[3] = Sigma[1][1];  // yy
	cov3D[4] = Sigma[1][2];  // yz
	cov3D[5] = Sigma[2][2];  // zz
}

/**
 * @brief 基于tile的剔除算法，每个线程处理一个高斯点，计算该高斯点影响的tile数量。
 * 我们已知在3σ范围下该高斯占据的tile坐标，现在我们遍历所有tile，并且查看在此tile中该高斯的概率密度的最大值。我们只将这个值大于阈值的tile认为需要被渲染
 * 该函数使用并行计算来快速确定某个高斯点在哪些tile上有显著贡献，避免渲染不必要的tile
 * @param active 高斯点是否激活（可见且有效）
 * @param co_init 圆锥矩阵和不透明度参数 (xx, xy, yy, opacity)
 * @param xy_init 高斯点在图像平面上的位置
 * @param opacity_power_threshold_init 不透明度阈值（用于剔除判断）
 * @param rect_min_init 高斯点影响区域的最小tile坐标（左上角）
 * @param rect_max_init 高斯点影响区域的最大tile坐标（右下角）
 * @return 高斯点影响的有效tile数量
 */
__device__ inline int computeTilebasedCullingTileCount(const bool active, const float4 co_init, const float2 xy_init,
													   const float opacity_power_threshold_init,
													   const uint2 rect_min_init,  const uint2 rect_max_init)
{
	// 计算高斯点影响的总tile数量
	const int32_t tile_count_init = (rect_max_init.y - rect_min_init.y) * (rect_max_init.x - rect_min_init.x);
	int tile_count = 0;

	// 如果高斯点有效，先处理少量tile（顺序处理）
	if (active)
	{
		const uint32_t rect_width = (rect_max_init.x - rect_min_init.x); // 以一个tile的宽度为单位计算宽度
		// 顺序处理前SEQUENTIAL_TILE_THRESH个tile
		for (int tile_idx = 0; tile_idx < tile_count_init && tile_idx < SEQUENTIAL_TILE_THRESH; tile_idx++)
		{
			// 计算当前tile的坐标
			//使用行优先遍历tile，从左到右，从上到下
			const int y = (tile_idx / rect_width) + rect_min_init.y;
			const int x = (tile_idx % rect_width) + rect_min_init.x;

			// 计算tile的像素范围
			const glm::vec2 tile_min = {x * BLOCK_X, y * BLOCK_Y}; //tile_min是tile的左上角坐标
			const glm::vec2 tile_max = {(x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1}; //tile_max是tile的右下角坐标

			// 计算高斯点对该tile的最大贡献度
			glm::vec2 max_pos;
			float max_opac_factor = max_contrib_power_rect_gaussian_float(co_init, xy_init, tile_min, tile_max, max_pos);

			// 如果贡献度超过阈值，计数加1
			tile_count += (max_opac_factor <= opacity_power_threshold_init);
		}
	}

	// 获取当前线程在warp中的位置信息
	const uint32_t lane_idx = cg::this_thread_block().thread_rank() % WARP_SIZE;
	const uint32_t warp_idx = cg::this_thread_block().thread_rank() / WARP_SIZE;

	// 判断是否需要协作计算剩余的tile
	const int32_t compute_cooperatively = active && tile_count_init > SEQUENTIAL_TILE_THRESH;
	const uint32_t remaining_threads = __ballot_sync(WARP_MASK, compute_cooperatively);

	// 如果没有线程需要协作计算，直接返回结果
	if (remaining_threads == 0)
		return tile_count;

	// 协作处理剩余的tile（warp级并行）
	const uint32_t n_remaining_threads = __popc(remaining_threads);
	for (int n = 0; n < n_remaining_threads && n < WARP_SIZE; n++)
	{
		// 找到下一个需要协作的线程
		const uint32_t i = __fns(remaining_threads, 0, n+1);

		// 通过warp shuffle获取其他线程的数据
		const uint2 rect_min = make_uint2(__shfl_sync(WARP_MASK, rect_min_init.x, i), __shfl_sync(WARP_MASK, rect_min_init.y, i));
		const uint2 rect_max = make_uint2(__shfl_sync(WARP_MASK, rect_max_init.x, i), __shfl_sync(WARP_MASK, rect_max_init.y, i));
		const float2 xy = { __shfl_sync(WARP_MASK, xy_init.x, i), __shfl_sync(WARP_MASK, xy_init.y, i) };

		const float4 co =
		{
			__shfl_sync(WARP_MASK, co_init.x, i),
			__shfl_sync(WARP_MASK, co_init.y, i),
			__shfl_sync(WARP_MASK, co_init.z, i),
			__shfl_sync(WARP_MASK, co_init.w, i),
		};
		const float opacity_power_threshold = __shfl_sync(WARP_MASK, opacity_power_threshold_init, i);

		// 计算剩余需要处理的tile数量
		const uint32_t rect_width = (rect_max.x - rect_min.x);
		const uint32_t rect_tile_count = (rect_max.y - rect_min.y) * rect_width;
		const uint32_t remaining_rect_tile_count = rect_tile_count - SEQUENTIAL_TILE_THRESH;

		// 计算迭代次数（每个warp处理一批tile）
		const int32_t n_iterations = (remaining_rect_tile_count + WARP_SIZE - 1) / WARP_SIZE;
		for (int it = 0; it < n_iterations; it++)
		{
			// 计算当前处理的tile索引
			const int tile_idx = it * WARP_SIZE + lane_idx + SEQUENTIAL_TILE_THRESH;
			const int active_curr_it = tile_idx < rect_tile_count;

			// 计算tile坐标
			const int y = (tile_idx / rect_width) + rect_min.y;
			const int x = (tile_idx % rect_width) + rect_min.x;

			const glm::vec2 tile_min = {x * BLOCK_X, y * BLOCK_Y};
			const glm::vec2 tile_max = {(x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1};

			// 计算贡献度
			glm::vec2 max_pos;
			const float max_opac_factor = max_contrib_power_rect_gaussian_float(co, xy, tile_min, tile_max, max_pos);

			// 判断该tile是否受到显著影响
			const uint32_t tile_contributes = active_curr_it && max_opac_factor <= opacity_power_threshold;

			// warp内投票统计贡献的tile数量
			const uint32_t contributes_ballot = __ballot_sync(WARP_MASK, tile_contributes);
			const uint32_t n_contribute = __popc(contributes_ballot);

			// 只有当前处理线程的数据线程才累加计数
			tile_count += (i == lane_idx) * n_contribute;
		}
	}

	return tile_count;
}

/**
 * @brief CUDA预处理内核，为每个高斯点计算渲染所需的参数，每个线程处理一个高斯点
 * 该内核将3D高斯点投影到2D图像平面，计算协方差矩阵、颜色等渲染参数
 * @tparam C 颜色通道数（通常为3，RGB）
 * @param P 高斯点总数
 * @param D 球谐函数的度数
 * @param M 球谐函数的最大系数数量
 * @param orig_points 原始3D高斯点位置
 * @param scales 高斯点缩放参数
 * @param scale_modifier 全局缩放修正因子
 * @param rotations 高斯点旋转参数（四元数）
 * @param opacities 高斯点不透明度
 * @param dc 直流颜色分量
 * @param shs 球谐函数系数
 * @param clamped 颜色截断标记
 * @param cov3D_precomp 预计算的3D协方差矩阵（可选）
 * @param colors_precomp 预计算的颜色（可选）
 * @param viewmatrix 视图变换矩阵
 * @param projmatrix 投影变换矩阵
 * @param cam_pos 相机位置
 * @param W 图像宽度
 * @param H 图像高度
 * @param tan_fovx x方向视场角正切值
 * @param tan_fovy y方向视场角正切值
 * @param focal_x x方向焦距
 * @param focal_y y方向焦距
 * @param limx_neg x方向边界负限
 * @param limx_pos x方向边界正限
 * @param limy_neg y方向边界负限
 * @param limy_pos y方向边界正限
 * @param radii 输出：高斯点的半径
 * @param points_xy_image 输出：高斯点在图像平面上的位置
 * @param depths 输出：高斯点的深度值
 * @param cov3Ds 输出：3D协方差矩阵
 * @param rgb 输出：高斯点的RGB颜色
 * @param conic_opacity 输出：圆锥矩阵和不透明度
 * @param grid tile网格尺寸
 * @param tiles_touched 输出：每个高斯点影响的tile数量
 * @param prefiltered 是否已经预过滤
 * @param no_color 是否跳过颜色计算
 * 
 * @note 作者设置active而不是直接返回，是因为computeTilebasedCullingTileCount需要整个warp都参与计算，当前线程不能提前返回。
 */ 

template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
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
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	const float limx_neg,
	const float limx_pos,
	const float limy_neg,
	const float limy_pos,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered, bool no_color) 
{
	// 获取当前线程处理的全局高斯点索引，是第几个线程就处理第几个高斯点
	auto idx = cg::this_grid().thread_rank();
	bool active = true;

	// 如果索引超出范围，标记为非活跃状态
	if (idx >= P)
	{
		active = false;
		idx = P - 1;  // 防止数组越界
	}

	// 初始化输出数组，如果高斯点被剔除，则其半径和覆盖的tile数量为0
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// 检查高斯点是否在视锥体内
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
	{
		active = false;
	}

	// 获取原始3D点坐标并进行投影变换，得到ndc坐标，还是归一化平面坐标？
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);  // 透视除法
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// 计算3D协方差矩阵
	computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
	const float* cov3D = cov3Ds + idx * 6;

	// 将3D协方差矩阵投影到2D
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy,
							 limx_neg, limx_pos, limy_neg, limy_pos, cov3D, viewmatrix);

	// 计算协方差矩阵行列式，判断矩阵是否可逆
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
	{
		active = false;
	}

	// 计算圆锥矩阵（协方差矩阵的逆矩阵）
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// 构建圆锥矩阵和不透明度参数
	const float4 co = { conic.x, conic.y, conic.z, opacities[idx] };

	// 如果不透明度太低，标记为非活跃
	if (co.w < OPACITY_THRESHOLD)
	{
		active = false;
	}

	// 如果整个warp都被剔除，提前返回
	if (__ballot_sync(WARP_MASK, active) == 0)
		return;

	// 计算高斯点的有效半径
	// lambda1是协方差矩阵的最大特征值，代表高斯分布的主轴长度
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(lambda1));  // 3倍标准差作为半径

	// 将NDC坐标转换为像素坐标
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

	// 计算高斯点影响的tile范围，rect_min和rect_max是tile范围的左上角和右下角坐标
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);

	// 计算不透明度阈值（对数形式）
	const float opacity_factor_threshold = logf(co.w / OPACITY_THRESHOLD);

	// 计算该高斯点影响的有效tile数量（基于tile剔除）
	const int tile_count = computeTilebasedCullingTileCount(active, co, point_image,
															opacity_factor_threshold, rect_min, rect_max);

	// 如果没有有效tile或高斯点不活跃，提前返回
	if (tile_count == 0 || !active)
		return;

	// 计算高斯点颜色（如果需要）
	if (!no_color)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, dc, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// 设置输出参数
	depths[idx] = p_view.z;                    // 深度值
	radii[idx] = my_radius;                    // 半径
	points_xy_image[idx] = point_image;        // 图像平面位置
	conic_opacity[idx] = co;                   // 圆锥矩阵和不透明度
	tiles_touched[idx] = tile_count;           // 影响的tile数量
}

/**
 * @brief CUDA渲染内核，实现基于tile的光栅化渲染，一个block处理一个tile，一个线程处理一个像素
 * 该内核对每个tile执行alpha混合，将高斯点按深度排序渲染到图像上
 * 具体流程：
 * 一些其他的事：维护bucket_to_tile
 * 依据BLOCK_SIZE将高斯点分为多个批次，即一个批次处理BLOCK_SIZE个高斯点，即分批协作加载 
  
    1.大家暂停计算。
    2.每人负责搬运 1 个高斯点：线程 0 去搬第 0 个点，线程 1 去搬第 1 个点... 放入 Shared Memory。
    3.这样一次就搬运了 256 个点进“高速缓存”。
    4.同步 (__syncthreads())：等大家都搬完了。
    5.现在，大家从 Shared Memory 里读取这 256 个点进行计算。
    6.算完这 256 个，再重复步骤 1 搬下一批。

	注意，由于 shared memory 有限，我们不能一次性搬运所有高斯点，所以需要分批搬运。
	这就像切香肠：
    切一块（256个高斯）。
    吃掉（计算）。
    再切一块。
    再吃掉。

	计算时，我们会每32个高斯点采样一次T值和累积颜色（用于反向传播）

 * @tparam CHANNELS 颜色通道数（通常为3，RGB）
 * @param ranges 每个tile中高斯点范围 [start, end)
 * @param point_list 高斯点列表（按tile分组）
 * @param per_tile_bucket_offset 每个tile的bucket偏移量
 * @param bucket_to_tile bucket到tile的映射
 * @param sampled_T 采样透明度T值（用于反向传播）
 * @param sampled_ar 采样累积颜色（用于反向传播）
 * @param W 图像宽度
 * @param H 图像高度
 * @param points_xy_image 高斯点在图像平面上的位置
 * @param depths 高斯点深度值
 * @param features 高斯点颜色特征
 * @param conic_opacity 圆锥矩阵和不透明度
 * @param n_contrib 每个像素贡献的高斯点数量
 * @param max_contrib tile中的最大贡献者数量
 * @param bg_color 背景颜色
 * @param out_color 输出颜色缓冲区
 * @param out_final_T 输出最终透明度T
 * @param no_color 是否跳过颜色渲染
 */ 
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const uint32_t* __restrict__ per_tile_bucket_offset, uint32_t* __restrict__ bucket_to_tile,
	float* __restrict__ sampled_T, float* __restrict__ sampled_ar,
	int W, int H,
	const float2* __restrict__ points_xy_image, 
	const float* __restrict__ depths,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	uint32_t* __restrict__ n_contrib,
	uint32_t* __restrict__ max_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_final_T,
	bool no_color) 
{
	// 确定当前处理的tile及其像素范围
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;  // 计算水平方向的tile数量

	// 计算当前tile的像素范围
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };

	// 计算当前线程处理的像素坐标
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;  // 计算像素在一维数组中的索引
	float2 pixf = { (float)pix.x, (float)pix.y };  // 像素坐标（浮点数）

	// 检查当前线程是否对应有效像素
	bool inside = pix.x < W&& pix.y < H;
	// 已完成的线程可以帮助数据获取，但不进行光栅化
	bool done = !inside;

	// 获取当前tile需要处理的高斯点范围
	uint32_t tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
	uint2 range = ranges[tile_id];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);  // 一次处理BLOCK_SIZE个高斯，那么处理完所有高斯需要多少个批次？
	int toDo = range.y - range.x;  // 剩余需要处理的高斯点数量

	// 维护bucket_to_tile，即bucket与tile的映射关系，index为bucket的索引，value为tile的索引
	uint32_t bbm = 0;  // bucket before me
	if (!no_color)
	{
		// 获取当前tile的bucket偏移量
		bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];

		// 维护bucket到tile的映射关系
		// 对于当前tile，会新添加num_buckets个bucket，我们要更新bucket_to_tile数组，将新添加的bucket映射到当前tile
		int num_buckets = (toDo + 31) / 32;  // 每个bucket包含32个高斯点，向上取整计算当前tile需要的bucket数量
		// 跨步循环
		for (int i = 0; i < (num_buckets + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i)
		{
			int bucket_idx = i * BLOCK_SIZE + block.thread_rank();
			if (bucket_idx < num_buckets)
			{
				bucket_to_tile[bbm + bucket_idx] = tile_id;
			}
		}
	}

	// 为批量获取的数据分配共享内存
	__shared__ int collected_id[BLOCK_SIZE];              // 高斯点ID
	__shared__ float2 collected_xy[BLOCK_SIZE];           // 高斯点位置
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE]; // 圆锥矩阵和不透明度

	// 初始化渲染变量
	float T = 1.0f;           // 当前透明度（从1.0开始，表示完全透明）
	uint32_t contributor = 0; // 当前贡献者计数
	uint32_t last_contributor = 0; // 最后一个贡献者
	uint32_t contributor_real = 0; // 实际颜色贡献者计数
	float C[CHANNELS] = { 0 };     // 累积颜色

	// 按批次处理所有高斯点，直到完成或范围结束
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// 如果整个block都完成了光栅化，提前结束
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// 集体从 global memory 获取当前批次的高斯点数据到 shared memory
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();  // 同步确保所有线程都完成了数据获取

		// 处理当前批次中的每个高斯点
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// 每32个高斯点采样一次T值和累积颜色（用于反向传播）
			if (j % 32 == 0 && !no_color)
			{
				sampled_T[(bbm * BLOCK_SIZE) + block.thread_rank()] = T;
				for (int ch = 0; ch < CHANNELS; ++ch)
				{
					sampled_ar[(bbm * BLOCK_SIZE * CHANNELS) + ch * BLOCK_SIZE + block.thread_rank()] = C[ch];
				}
				++bbm;
			}

			// 更新贡献者计数
			contributor++;

			// 使用圆锥矩阵进行重采样（参考"Surface Splatting"论文，Zwicker et al., 2001）
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };  // 计算像素到高斯点中心的偏移
			float4 con_o = collected_conic_opacity[j];

			// 计算高斯分布的power值（负指数部分）
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f) { continue; }  // 如果power > 0，贡献可忽略，此时这个高斯点会在相机后方

			// 计算alpha值（3D Gaussian Splatting论文公式2）
			// 将高斯不透明度与其从均值的指数衰减相乘
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f) { continue; }  // alpha太小，跳过

			// 测试新的透明度T值
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;  // 像素基本不透明，后续高斯点贡献可忽略
				continue;
			}

			// 累积颜色（3D Gaussian Splatting论文公式3）
			if (!no_color)
			{
				for (int ch = 0; ch < CHANNELS; ch++)
					C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
				contributor_real++;
			}

			// 更新透明度和最后一个贡献者
			T = test_T;
			last_contributor = contributor;
		}
	}

	// 所有处理有效像素的线程将最终渲染结果写入帧缓冲区和辅助缓冲区
	if (inside)
	{
		out_final_T[pix_id] = T;  // 输出最终透明度T
		if (!no_color)
		{
			n_contrib[pix_id] = last_contributor;  // 输出贡献者数量
			for (int ch = 0; ch < CHANNELS; ch++)
				out_color[ch * H * W + pix_id] = C[ch];  // 输出累积颜色
		}
	}
	if (no_color) { return; }

	// 对最后一个贡献者进行block级最大值规约
	// typedef cub::BlockReduce<uint32_t, BLOCK_SIZE> BlockReduce;
	typedef cub::BlockReduce<uint32_t, BLOCK_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_Y> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	last_contributor = BlockReduce(temp_storage).Reduce(last_contributor, cub::Max());

	// block的主线程将tile的最大贡献者数量写入输出
	if (block.thread_rank() == 0)
	{
		max_contrib[tile_id] = last_contributor;
	}
}


/**
 * @brief FORWARD类的渲染接口函数，启动CUDA渲染内核
 * 该函数是C++到CUDA的接口，负责调用renderCUDA内核进行实际渲染
 */
void FORWARD::render( const dim3 grid, dim3 block, const uint2* ranges,
	const uint32_t* point_list,
	const uint32_t* per_tile_bucket_offset, uint32_t* bucket_to_tile,
	float* sampled_T, float* sampled_ar,
	int W, int H,
	const float2* means2D,
	const float* depths,
	const float* colors,
	const float4* conic_opacity,
	uint32_t* n_contrib,
	uint32_t* max_contrib,
	const float* bg_color,
	float* out_color,
	float* out_final_T,
	bool no_color)
{
	// 启动渲染CUDA内核，使用模板参数NUM_CHAFFELS（应该是NUM_CHANNELS的拼写错误）
	renderCUDA<NUM_CHAFFELS> <<<grid, block>>> (
		ranges,
		point_list,
		per_tile_bucket_offset, bucket_to_tile,
		sampled_T, sampled_ar,
		W, H,
		means2D,
		depths,
		colors,
		conic_opacity,
		n_contrib,
		max_contrib,
		bg_color,
		out_color,
		out_final_T,
		no_color);
}

/**
 * @brief FORWARD类的预处理接口函数，启动CUDA预处理内核
 * 该函数是C++到CUDA的接口，负责调用preprocessCUDA内核进行预处理计算
 */
void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
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
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered, bool no_color)
{
	// 启动预处理CUDA内核，使用模板参数NUM_CHAFFELS（应该是NUM_CHANNELS的拼写错误）
	// 内核配置：每个block 256个线程，总共(P + 255) / 256个block
	preprocessCUDA<NUM_CHAFFELS> <<<(P + 255) / 256, 256>>> (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		dc,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		limx_neg,
		limx_pos,
		limy_neg,
		limy_pos,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered, no_color);
}