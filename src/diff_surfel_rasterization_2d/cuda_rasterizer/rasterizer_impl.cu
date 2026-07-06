

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


__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}


__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	const float* transMats,
	const float4* normal_opacity,
	dim3 grid,
	bool use_tile_culling)
{
	auto idx = cg::this_grid().thread_rank();
	bool active = true;
	if (idx >= P)
	{
		active = false;
		idx = P - 1;
	}

	if (active && radii[idx] <= 0) { active = false; }

	if (__ballot_sync(WARP_MASK, active) == 0) { return; }

	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
	const uint32_t offset_to = offsets[idx];

	if (cg::this_grid().thread_rank() >= P) {
		off = offset_to;
	}

	uint2 rect_min, rect_max;
	getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

	float3 Tu, Tv, Tw;
	float cutoff = 0.0f;
	if (active) {
		Tu = {transMats[9 * idx + 0], transMats[9 * idx + 1], transMats[9 * idx + 2]};
		Tv = {transMats[9 * idx + 3], transMats[9 * idx + 4], transMats[9 * idx + 5]};
		Tw = {transMats[9 * idx + 6], transMats[9 * idx + 7], transMats[9 * idx + 8]};
		float opacity = normal_opacity[idx].w;
		#if TIGHTBBOX
			cutoff = sqrtf(max(9.f + 2.f * logf(opacity), 0.000001));
		#else
			cutoff = 3.0f;
		#endif
	}

	const uint32_t rect_width = (rect_max.x - rect_min.x);
	const int32_t tile_count_init = (rect_max.y - rect_min.y) * rect_width;

	if (active)
	{
		for (int tile_idx = 0; tile_idx < tile_count_init && tile_idx < SEQUENTIAL_TILE_THRESH && off < offset_to; tile_idx++)
		{
			const int y = (tile_idx / rect_width) + rect_min.y;
			const int x = (tile_idx % rect_width) + rect_min.x;

			bool overlap = true;
			if (use_tile_culling)
			{
				overlap = check_tile_overlap({ x, y }, Tu, Tv, Tw, cutoff);
			}

			if (overlap)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}

	const uint32_t lane_idx = cg::this_thread_block().thread_rank() % WARP_SIZE;
	unsigned int lane_mask_allprev_excl = 0xFFFFFFFFU >> (WARP_SIZE - lane_idx);

	const int32_t compute_cooperatively = active && tile_count_init > SEQUENTIAL_TILE_THRESH;
	const uint32_t remaining_threads = __ballot_sync(WARP_MASK, compute_cooperatively);

	if (remaining_threads == 0)
	{
		while (off < offset_to)
		{
			uint64_t key = (uint32_t)-1;
			key <<= 32;
			const float depth = FLT_MAX;
			key |= *((uint32_t*)&depth);
			gaussian_keys_unsorted[off] = key;
			gaussian_values_unsorted[off] = static_cast<uint32_t>(-1);
			off++;
		}
		return;
	}

	uint32_t n_remaining_threads = __popc(remaining_threads);
	for (int n = 0; n < n_remaining_threads && n < WARP_SIZE; n++)
	{
		int i = __fns(remaining_threads, 0, n + 1);

		uint32_t idx_i = __shfl_sync(WARP_MASK, idx, i);
		uint32_t off_i = __shfl_sync(WARP_MASK, off, i);
		const uint32_t offset_to_i = __shfl_sync(WARP_MASK, offset_to, i);

		const uint2 rect_min_i = make_uint2(__shfl_sync(WARP_MASK, rect_min.x, i), __shfl_sync(WARP_MASK, rect_min.y, i));
		const uint2 rect_max_i = make_uint2(__shfl_sync(WARP_MASK, rect_max.x, i), __shfl_sync(WARP_MASK, rect_max.y, i));

		const float3 Tu_i = shfl_float3(Tu, i);
		const float3 Tv_i = shfl_float3(Tv, i);
		const float3 Tw_i = shfl_float3(Tw, i);
		const float cutoff_i = __shfl_sync(WARP_MASK, cutoff, i);

		const uint32_t rect_width_i = (rect_max_i.x - rect_min_i.x);
		const uint32_t tile_count_i = (rect_max_i.y - rect_min_i.y) * rect_width_i;
		const uint32_t remaining_tile_count = tile_count_i - SEQUENTIAL_TILE_THRESH;
		const int32_t n_iterations = (remaining_tile_count + WARP_SIZE - 1) / WARP_SIZE;

		for (int it = 0; it < n_iterations; it++)
		{
			int tile_idx = it * WARP_SIZE + lane_idx + SEQUENTIAL_TILE_THRESH;
			int y = (tile_idx / rect_width_i) + rect_min_i.y;
			int x = (tile_idx % rect_width_i) + rect_min_i.x;

			bool write = false;
			if (tile_idx < tile_count_i) {
				bool overlap_i = true;
				if (use_tile_culling)
				{
					overlap_i = check_tile_overlap({ x, y }, Tu_i, Tv_i, Tw_i, cutoff_i);
				}
				write = overlap_i;
			}

			const uint32_t write_ballot = __ballot_sync(WARP_MASK, write);
			const uint32_t n_writes = __popc(write_ballot);
			const uint32_t write_offset = off_i + __popc(write_ballot & lane_mask_allprev_excl);

			if (write && write_offset < offset_to_i)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx_i]);
				gaussian_keys_unsorted[write_offset] = key;
				gaussian_values_unsorted[write_offset] = idx_i;
			}
			off_i += n_writes;
			off += (i == lane_idx) * n_writes;
		}
	}

	while (off < offset_to)
	{
		uint64_t key = (uint32_t)-1;
		key <<= 32;
		const float depth = FLT_MAX;
		key |= *((uint32_t*)&depth);
		gaussian_keys_unsorted[off] = key;
		gaussian_values_unsorted[off] = static_cast<uint32_t>(-1);
		off++;
	}
}


__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;

	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}

	if (idx == L - 1)
		ranges[currtile].y = L;
}

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


CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.transMat, P * 9, 128);
	obtain(chunk, geom.normal_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);

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
	const bool use_tile_culling,
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

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

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
		geomState.depths,
		geomState.transMat,
		geomState.rgb,
		geomState.normal_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		use_tile_culling
	), debug)

	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		geomState.transMat,
		geomState.normal_opacity,
		tile_grid,
		use_tile_culling)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

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
		(float3*)dL_dmean2D,
		dL_dnormal,
		dL_dtransMat,
		dL_dcolor,
		dL_dsh,
		(glm::vec3*)dL_dmean3D,
		(glm::vec2*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}
