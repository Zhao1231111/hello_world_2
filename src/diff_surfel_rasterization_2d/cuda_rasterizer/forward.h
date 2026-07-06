

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cmath>
#include <cfloat>
#include "auxiliary.h"

namespace FORWARD
{

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
		float* transMats,
		float* colors,
		float4* normal_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		bool use_tile_culling);


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


__device__ __forceinline__ bool checkIntersectionSquare(float half_size, const Quad& q) {

    float q_min_x = q.v[0].x;
    float q_max_x = q.v[0].x;
    float q_min_y = q.v[0].y;
    float q_max_y = q.v[0].y;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x = q.v[i].x;
        const float y = q.v[i].y;

        if (fmaxf(fabsf(x), fabsf(y)) <= half_size) {
            return true;
        }

        q_min_x = fminf(q_min_x, x);
        q_max_x = fmaxf(q_max_x, x);
        q_min_y = fminf(q_min_y, y);
        q_max_y = fmaxf(q_max_y, y);
    }

    if (q_min_x > half_size || q_max_x < -half_size ||
        q_min_y > half_size || q_max_y < -half_size) {
        return false;
    }


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

        const float aabb_radius = half_size * (fabsf(axis_x) + fabsf(axis_y));

        float q_proj_min, q_proj_max;
        const float proj0 = q.v[0].x * axis_x + q.v[0].y * axis_y;
        q_proj_min = q_proj_max = proj0;

        #pragma unroll(3)
        for (int j = 1; j < 4; ++j) {
            const float proj = q.v[j].x * axis_x + q.v[j].y * axis_y;
            q_proj_min = fminf(q_proj_min, proj);
            q_proj_max = fmaxf(q_proj_max, proj);
        }

        if (q_proj_min > aabb_radius || q_proj_max < -aabb_radius) {
            return false;
        }
    }

    return true;
}


__device__ __forceinline__ bool checkIntersectionCircle(float radius_sq, const Quad& q) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float dist_sq = q.v[i].x * q.v[i].x + q.v[i].y * q.v[i].y;
        if (dist_sq <= radius_sq) {
            return true;
        }
    }

    const float cross0 = q.v[0].x * q.v[1].y - q.v[0].y * q.v[1].x;
    const float cross1 = q.v[1].x * q.v[2].y - q.v[1].y * q.v[2].x;
    const float cross2 = q.v[2].x * q.v[3].y - q.v[2].y * q.v[3].x;
    const float cross3 = q.v[3].x * q.v[0].y - q.v[3].y * q.v[0].x;

    if ((cross0 > 0 && cross1 > 0 && cross2 > 0 && cross3 > 0) ||
        (cross0 < 0 && cross1 < 0 && cross2 < 0 && cross3 < 0)) {
        return true;
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 p1 = q.v[i];
        const float2 p2 = q.v[(i + 1) & 3];

        const float2 edge_vec   = {p2.x - p1.x, p2.y - p1.y};
        const float2 p1_to_origin = {-p1.x, -p1.y};

        const float edge_len_sq = edge_vec.x * edge_vec.x + edge_vec.y * edge_vec.y;

        if (edge_len_sq < 1e-12f) {
            continue;
        }

        float t = (p1_to_origin.x * edge_vec.x + p1_to_origin.y * edge_vec.y) / edge_len_sq;

        t = fmaxf(0.0f, fminf(1.0f, t));

        const float closest_x = p1.x + t * edge_vec.x;
        const float closest_y = p1.y + t * edge_vec.y;

        const float closest_dist_sq = closest_x * closest_x + closest_y * closest_y;
        if (closest_dist_sq <= radius_sq) {
            return true;
        }
    }

    return false;
}


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

__device__ inline float2 project_to_local(float2 pix, float3 Tu, float3 Tv, float3 Tw) {
	float3 k = pix.x * Tw - Tu;
	float3 l = pix.y * Tw - Tv;
	float3 p = cross(k, l);
	if (p.z == 0.0f) return { 0.0f, 0.0f };
	return { p.x / p.z, p.y / p.z };
}

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
	q.v[0] = project_to_local({ min_x, min_y }, Tu, Tv, Tw);
	q.v[1] = project_to_local({ max_x, min_y }, Tu, Tv, Tw);
	q.v[2] = project_to_local({ max_x, max_y }, Tu, Tv, Tw);
	q.v[3] = project_to_local({ min_x, max_y }, Tu, Tv, Tw);

	return checkIntersectionCircle(cutoff * cutoff, q);
}

#endif
