/**
 * @file forward.h
 * @brief 2D高斯光栅化前向传播头文件
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cmath> // For fabsf
#include <cfloat> // For FLT_MIN
#include "auxiliary.h"

namespace FORWARD
{
	/**
	 * @brief 高斯渲染预处理（坐标投影、颜色转换、包围盒计算等）
	 */
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec2* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* transMat_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		// float* isovals,
		// float3* normals,
		float* transMats,
		float* colors,
		float4* normal_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		bool use_tile_culling);

	/**
	 * @brief 核心渲染函数：逐 Tile 进行 alpha 混合合成图像
	 */
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		float focal_x, float focal_y,
		const float2* points_xy_image,
		const float* features,
		const float* transMats,
		const float* depths,
		const float4* normal_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_others);
}

// ========================== my code ==========================
/**
 * @brief 判断一个中心在原点的正方形AABB与一个四边形是否相交 (CUDA C++)
 * 
 * 算法针对大部分情况都相交的场景进行了优化，并处理了因投影变换产生的数值不稳定问题。
 *
 * @param half_size 正方形AABB的半边长 (side_length / 2)
 * @param q         待测试的四边形
 * @return          true 如果相交, false 如果不相交
 */
__device__ __forceinline__ bool checkIntersectionSquare(float half_size, const Quad& q) {
    // =======================================================================
    // 阶段 1: AABB轴检测 (快速拒绝) & 顶点包含检测 (快速接受)
    // =======================================================================
    
    float q_min_x = q.v[0].x;
    float q_max_x = q.v[0].x;
    float q_min_y = q.v[0].y;
    float q_max_y = q.v[0].y;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x = q.v[i].x;
        const float y = q.v[i].y;

        // 使用L-infinity范数 (Chebyshev distance) 判断点是否在正方形内
        if (fmaxf(fabsf(x), fabsf(y)) <= half_size) {
            return true; // 快速接受
        }

        q_min_x = fminf(q_min_x, x);
        q_max_x = fmaxf(q_max_x, x);
        q_min_y = fminf(q_min_y, y);
        q_max_y = fmaxf(q_max_y, y);
    }

    if (q_min_x > half_size || q_max_x < -half_size || 
        q_min_y > half_size || q_max_y < -half_size) {
        return false; // 快速拒绝
    }

    // =======================================================================
    // 阶段 2 & 3: 四边形法线轴的精确SAT (包含退化边过滤)
    // =======================================================================
    
    const float DEGENERATE_THRESHOLD = 1e-6f;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 p1 = q.v[i];
        const float2 p2 = q.v[(i + 1) & 3];
        
        const float edge_x = p2.x - p1.x;
        const float edge_y = p2.y - p1.y;

        const float manhattan_len = fabsf(edge_x) + fabsf(edge_y);
        if (manhattan_len < DEGENERATE_THRESHOLD) {
            continue;
        }
        
        const float axis_x = -edge_y;
        const float axis_y = edge_x;

        // --- 关键优化点 ---
        // 1. 计算AABB在轴上的投影半径。
        //    利用正方形的对称性，公式被简化。
        const float aabb_radius = half_size * (fabsf(axis_x) + fabsf(axis_y));

        // 2. 将四边形投影到该轴上
        float q_proj_min, q_proj_max;
        const float proj0 = q.v[0].x * axis_x + q.v[0].y * axis_y;
        q_proj_min = q_proj_max = proj0;
        
        #pragma unroll(3) // 少量循环，可以手动展开或提示编译器
        for (int j = 1; j < 4; ++j) {
            const float proj = q.v[j].x * axis_x + q.v[j].y * axis_y;
            q_proj_min = fminf(q_proj_min, proj);
            q_proj_max = fmaxf(q_proj_max, proj);
        }

        // 3. 检查投影是否分离
        if (q_proj_min > aabb_radius || q_proj_max < -aabb_radius) {
            return false;
        }
    }

    return true;
}

/**
 * @brief 判断一个中心在原点的 *圆* 与一个四边形是否相交 (CUDA C++)
 * 
 * 该算法利用了圆的对称性，比AABB-Quad的SAT算法更直接和高效。
 *
 * @param radius_sq 圆半径的平方 (radius * radius)，传入平方值以避免开方
 * @param q         待测试的四边形
 * @return          true 如果相交, false 如果不相交
 */
__device__ __forceinline__ bool checkIntersectionCircle(float radius_sq, const Quad& q) {
    // =======================================================================
    // 阶段 1: 顶点检测 (快速接受)
    // 检查是否有任何四边形的顶点在圆内。
    // =======================================================================
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float dist_sq = q.v[i].x * q.v[i].x + q.v[i].y * q.v[i].y;
        if (dist_sq <= radius_sq) {
            return true;
        }
    }

    // =======================================================================
    // 阶段 2: 圆心在四边形内检测 (次级快速接受)
    // 适用于四边形包围圆的情况。
    // 使用叉积判断原点(0,0)是否在所有边的同一侧。
    // =======================================================================
    // 我们检查原点相对于边 v[i]->v[i+1] 的位置
    // cross_product = (p2.x - p1.x) * (p.y - p1.y) - (p2.y - p1.y) * (p.x - p1.x)
    // 由于 p=(0,0), p.x和p.y为0, 公式简化为:
    // cross_product = -(p2.x - p1.x) * p1.y + (p2.y - p1.y) * p1.x
    // cross_product = p1.x*p2.y - p1.y*p2.x
    const float cross0 = q.v[0].x * q.v[1].y - q.v[0].y * q.v[1].x;
    const float cross1 = q.v[1].x * q.v[2].y - q.v[1].y * q.v[2].x;
    const float cross2 = q.v[2].x * q.v[3].y - q.v[2].y * q.v[3].x;
    const float cross3 = q.v[3].x * q.v[0].y - q.v[3].y * q.v[0].x;

    // 如果所有叉积结果同号 (全>0 或 全<0)，则原点在四边形内部
    if ((cross0 > 0 && cross1 > 0 && cross2 > 0 && cross3 > 0) ||
        (cross0 < 0 && cross1 < 0 && cross2 < 0 && cross3 < 0)) {
        return true;
    }

    // =======================================================================
    // 阶段 3: 边距检测 (最终精确检测)
    // 计算原点到四边形每条线段的最短距离。
    // =======================================================================
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 p1 = q.v[i];
        const float2 p2 = q.v[(i + 1) & 3];

        const float2 edge_vec   = {p2.x - p1.x, p2.y - p1.y};
        const float2 p1_to_origin = {-p1.x, -p1.y};

        const float edge_len_sq = edge_vec.x * edge_vec.x + edge_vec.y * edge_vec.y;
        
        // 避免除以零，对于退化边，其端点已在阶段1检查过
        if (edge_len_sq < 1e-12f) {
            continue;
        }

        // 将原点投影到边所在的直线上，计算投影参数t
        // t = dot(p1_to_origin, edge_vec) / |edge_vec|^2
        float t = (p1_to_origin.x * edge_vec.x + p1_to_origin.y * edge_vec.y) / edge_len_sq;

        // 将t限制在[0, 1]区间，确保我们得到的是线段上的最近点
        t = fmaxf(0.0f, fminf(1.0f, t));

        // 计算线段上离原点最近点的坐标
        const float closest_x = p1.x + t * edge_vec.x;
        const float closest_y = p1.y + t * edge_vec.y;

        // 检查该最近点的距离平方是否小于等于半径平方
        const float closest_dist_sq = closest_x * closest_x + closest_y * closest_y;
        if (closest_dist_sq <= radius_sq) {
            return true;
        }
    }

    // 所有检查都未发现相交
    return false;
}

// ====================================================================================
// Tile Culling Helpers (Shared between Forward and Rasterizer)
// ====================================================================================

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef WARP_MASK
#define WARP_MASK 0xFFFFFFFF
#endif

#define SEQUENTIAL_TILE_THRESH 8

__device__ inline float3 shfl_float3(float3 v, int srcLane) {
	return {
		__shfl_sync(WARP_MASK, v.x, srcLane),
		__shfl_sync(WARP_MASK, v.y, srcLane),
		__shfl_sync(WARP_MASK, v.z, srcLane)
	};
}

// Project screen pixel p to local 2D Gaussian plane (u, v)
__device__ inline float2 project_to_local(float2 pix, float3 Tu, float3 Tv, float3 Tw) {
	float3 k = pix.x * Tw - Tu;
	float3 l = pix.y * Tw - Tv;
	float3 p = cross(k, l);
	if (p.z == 0.0f) return { 0.0f, 0.0f };
	return { p.x / p.z, p.y / p.z };
}

// Check if a tile overlaps with the 2D Gaussian in its local space
// tile_pos: integer tile coordinates
// cutoff: radius of gaussian in local space
__device__ inline bool check_tile_overlap(
	const int2 tile_pos,
	const float3 Tu, const float3 Tv, const float3 Tw,
	const float cutoff)
{
	float min_x = tile_pos.x * BLOCK_X;
	float min_y = tile_pos.y * BLOCK_Y;
	float max_x = min_x + BLOCK_X;
	float max_y = min_y + BLOCK_Y;

	Quad q;
	// Construct quad from projected tile corners
	q.v[0] = project_to_local({ min_x, min_y }, Tu, Tv, Tw);
	q.v[1] = project_to_local({ max_x, min_y }, Tu, Tv, Tw);
	q.v[2] = project_to_local({ max_x, max_y }, Tu, Tv, Tw);
	q.v[3] = project_to_local({ min_x, max_y }, Tu, Tv, Tw);

	// Check intersection with Gaussian's local bounding box ([-cutoff, cutoff]^2)
	return checkIntersectionCircle(cutoff * cutoff, q);
	// return checkIntersectionSquare(cutoff, q);
}

#endif
