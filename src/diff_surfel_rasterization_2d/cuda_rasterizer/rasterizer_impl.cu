/**
 * @file rasterizer_impl.cu
 * @brief 2D高斯光栅化器调度逻辑的完整实现
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

/**
 * @brief CUDA内核：执行粗略视锥体剔除检查
 * 
 * 标记所有在当前相机视锥体内的高斯点。
 */
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	const uint32_t idx = static_cast<uint32_t>(cg::this_grid().thread_rank());
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

namespace
{
// 一个完整 CUDA warp 的 lane 数量；当前 kernel 固定以 256 线程启动，block 内不存在半个 warp。
constexpr uint32_t kWarpSize = 32;
// 小包围盒由 owner lane 顺序展开，大包围盒再交给整个 warp 协作，避免调度开销反而占主导。
constexpr int32_t kSequentialTileThreshold = 8;
}

/**
 * @brief CUDA内核：为 AABB 覆盖的每个 Gaussian-Tile 对生成键值。
 *
 * 每个有效高斯的输出数量已经由 preprocess 精确记录为 AABB 内 tile 数量。这里不再做
 * Ray-Splat 精确剔除，只保留 warp 协作展开，用于平衡少数大屏幕高斯的写入负载。
 */
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	const int* radii,
	const int2* bbox_extents,
	dim3 grid)
{
	const uint32_t idx = static_cast<uint32_t>(cg::this_grid().thread_rank());
	const bool active = idx < P && radii[idx] > 0;

	// 在任何 lane 分支退出之前记录当前执行 mask；后续所有 ballot/shuffle 都使用同一 mask。
	// 这样即使未来启动配置改成非整 warp block，也不会把不存在或已退出的 lane 写入 full mask。
	const uint32_t warp_mask = __activemask();
	if (__ballot_sync(warp_mask, active) == 0)
		return;

	uint32_t off = 0;
	uint32_t offset_to = 0;
	uint2 rect_min = make_uint2(0, 0);
	uint2 rect_max = make_uint2(0, 0);
	if (active)
	{
		off = (idx == 0) ? 0 : offsets[idx - 1];
		offset_to = offsets[idx];
		getRect(points_xy[idx], bbox_extents[idx], rect_min, rect_max, grid);
	}

	const uint32_t rect_width = (rect_max.x - rect_min.x);
	const int32_t tile_count_init = (rect_max.y - rect_min.y) * rect_width;

	// 小 AABB 先由 owner lane 直接写入前几个 tile。
	if (active)
	{
		for (int tile_idx = 0;
			tile_idx < tile_count_init && tile_idx < kSequentialTileThreshold && off < offset_to;
			++tile_idx)
		{
			const int y = (tile_idx / rect_width) + rect_min.y;
			const int x = (tile_idx % rect_width) + rect_min.x;
			uint64_t key = y * grid.x + x;
			key <<= 32;
			key |= *((uint32_t*)&depths[idx]);
			gaussian_keys_unsorted[off] = key;
			gaussian_values_unsorted[off] = idx;
			++off;
		}
	}

	// lane 0 没有更低 lane。显式分支还避免了旧写法在 lane 0 上右移 32 位的未定义行为。
	const uint32_t lane_idx = cg::this_thread_block().thread_rank() % kWarpSize;
	const uint32_t lane_mask_allprev_excl = lane_idx == 0 ? 0u : ((1u << lane_idx) - 1u);
	const bool compute_cooperatively = active && tile_count_init > kSequentialTileThreshold;
	const uint32_t remaining_threads = __ballot_sync(warp_mask, compute_cooperatively);

	// 所有小 AABB 已经完整写入，不需要用哨兵补齐输出。
	if (remaining_threads == 0)
		return;

	// Warp 依次帮助每个大 AABB 的 owner lane 展开剩余 tile。
	const uint32_t n_remaining_threads = __popc(remaining_threads);
	for (uint32_t n = 0; n < n_remaining_threads; ++n)
	{
		const int owner_lane = __fns(remaining_threads, 0, n + 1);
		const uint32_t idx_i = __shfl_sync(warp_mask, idx, owner_lane);
		uint32_t off_i = __shfl_sync(warp_mask, off, owner_lane);
		const uint32_t offset_to_i = __shfl_sync(warp_mask, offset_to, owner_lane);
		const uint2 rect_min_i = make_uint2(
			__shfl_sync(warp_mask, rect_min.x, owner_lane),
			__shfl_sync(warp_mask, rect_min.y, owner_lane));
		const uint2 rect_max_i = make_uint2(
			__shfl_sync(warp_mask, rect_max.x, owner_lane),
			__shfl_sync(warp_mask, rect_max.y, owner_lane));

		const uint32_t rect_width_i = (rect_max_i.x - rect_min_i.x);
		const uint32_t tile_count_i = (rect_max_i.y - rect_min_i.y) * rect_width_i;
		const uint32_t remaining_tile_count = tile_count_i - kSequentialTileThreshold;
		const int32_t n_iterations = (remaining_tile_count + kWarpSize - 1) / kWarpSize;

		for (int it = 0; it < n_iterations; ++it)
		{
			const uint32_t tile_idx = it * kWarpSize + lane_idx + kSequentialTileThreshold;
			const bool write = tile_idx < tile_count_i;
			const uint32_t write_ballot = __ballot_sync(warp_mask, write);
			const uint32_t n_writes = __popc(write_ballot);
			const uint32_t write_offset = off_i + __popc(write_ballot & lane_mask_allprev_excl);

			if (write && write_offset < offset_to_i)
			{
				const int y = (tile_idx / rect_width_i) + rect_min_i.y;
				const int x = (tile_idx % rect_width_i) + rect_min_i.x;
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx_i]);
				gaussian_keys_unsorted[write_offset] = key;
				gaussian_values_unsorted[write_offset] = idx_i;
			}
			off_i += n_writes;
		}
	}
}

/**
 * @brief CUDA内核：在排序列表中识别每个 Tile 的范围（Start/End 索引）
 */
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// 读取当前高斯的 Tile ID
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	
	// 如果是列表首位，则是该 Tile 的起点
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		// 检查 Tile ID 是否与前一个高斯点不同
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile) 
		{
			// ID 发生变化，标记上一个 Tile 的终点和当前 Tile 的起点
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	
	// 如果是列表末位，标记最后一个 Tile 的终点
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

/**
 * @brief 初始化几何状态指针
 * 
 * 将连续内存块切割并分配给 GeometryState 的各个成员变量。
 * 
 * @param chunk 内存块起始指针
 * @param P 高斯点总数
 */
CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.bbox_extents, P, 128);
	obtain(chunk, geom.transMat, P * 9, 128);
	obtain(chunk, geom.normal_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	
	// 使用 CUB 库执行并行前缀和，计算每个点在排序列表中的全局偏移
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N * 3, 128);
	obtain(chunk, img.n_contrib, N * 2, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

/**
 * @brief 2D高斯光栅化算子前向传播主入口
 * 
 * 协调预处理、排序、分块识别和最终渲染过程。
 * 
 * @param geometryBuffer 几何缓冲区分配回调
 * @param binningBuffer 分块缓冲区分配回调
 * @param imageBuffer 图像缓冲区分配回调
 * @param P 高斯点总数
 * @param D SH 阶数
 * @param M SH 系数数量
 * @param background 背景颜色
 * @param width 图像宽度
 * @param height 图像高度
 * @param means3D 3D位置指针
 * @param shs 球谐系数指针
 * @param colors_precomp 预计算颜色指针
 * @param opacities 不透明度指针
 * @param scales 2D缩放指针
 * @param scale_modifier 缩放修改器
 * @param rotations 旋转四元数指针
 * @param transMat_precomp 预计算变换矩阵指针
 * @param viewmatrix 视图矩阵
 * @param projmatrix 投影矩阵
 * @param cam_pos 相机位置
 * @param tan_fovx 水平FOV正切
 * @param tan_fovy 垂直FOV正切
 * @param prefiltered 是否已经过滤标志
 * @param out_color [输出] 渲染图像
 * @param out_others [输出] 其他辅助张量 (depth, alpha, normal, etc.)
 * @param radii [输出] 投影半径
 * @param debug 是否开启调试
 * 
 * @return int 成功进行渲染的高斯实例总数
 */
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* transMat_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* out_others,
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	// 1. 设置渲染网格和块尺寸
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// 动态分配图像级缓冲区空间
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// 2. 预处理：坐标映射、投影变换计算、AABB 包围盒计算
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec2*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.bbox_extents,
		geomState.depths,
		geomState.transMat,
		geomState.rgb,
		geomState.normal_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	// 3. 计算每个 Tile 的覆盖点数量前缀和，从而确定全局高斯点实例数
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	// 为分块排序分配缓冲区
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// 4. 生成排序键：[ Tile ID | Depth ]，实现分块且块内有序
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		geomState.bbox_extents,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	const float* transMat_ptr = transMat_precomp != nullptr ? transMat_precomp : geomState.transMat;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		focal_x, focal_y,
		geomState.means2D,
		feature_ptr,
		transMat_ptr,
		geomState.depths,
		geomState.normal_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_others), debug)

	return num_rendered;
}

/**
 * @brief 2D高斯光栅化算子反向传播主入口
 * 
 * 根据每个像素生成的梯度，反求出各高斯参数（均值、SH、不透明度、缩放、旋转）的梯度。
 */
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* transMat_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_depths,
	float* dL_dmean2D,
	float* dL_dnormal,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dtransMat,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	const float* depth_ptr = geomState.depths;
	const float* transMat_ptr = (transMat_precomp != nullptr) ? transMat_precomp : geomState.transMat;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		focal_x, focal_y,
		background,
		geomState.means2D,
		geomState.normal_opacity,
		color_ptr,
		transMat_ptr,
		depth_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_depths,
		dL_dtransMat,
		(float3*)dL_dmean2D,
		dL_dnormal,
		dL_dopacity,
		dL_dcolor), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	// const float* transMat_ptr = (transMat_precomp != nullptr) ? transMat_precomp : geomState.transMat;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec2*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		transMat_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D, // gradient inputs
		dL_dnormal,		     // gradient inputs
		dL_dtransMat,
		dL_dcolor,
		dL_dsh,
		(glm::vec3*)dL_dmean3D,
		(glm::vec2*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}
