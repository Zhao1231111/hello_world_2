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

// 辅助函数，用于在CPU上找到最高有效位（MSB）的下一位
// 用于基数排序确定需要的位数
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
 * @brief preprocess Kernel 已经精确计算了每个高斯会覆盖哪些tile，但只记录了数量 tiles_touched，此函数将这个过程重复了一次，将每个tile_id 填入长列表中
 * 
 * 这个Kernel的任务是将每个高斯点“复制”多次，每次对应它覆盖的一个Tile。
 * 对于每个覆盖的Tile，生成一个 (Key, Value) 对：
 * - Key: 高位是Tile ID，低位是深度 Depth。这样排序后，同一Tile的高斯点会聚在一起，且按深度排序。
 * - Value: 高斯点的原始索引 ID。
 * 
 * 此外，为了性能优化，这里还执行了更精细的“基于Tile的剔除”：
 * 即使一个高斯点的包围盒覆盖了某个Tile，但如果它在该Tile内的实际贡献（Alpha）非常小，也会被剔除。
 * 
 * @param P 高斯点总数
 * @param points_xy 高斯点在屏幕空间的2D坐标
 * @param conic_opacity 椭圆参数和不透明度
 * @param depths 高斯点的深度
 * @param offsets 每个高斯点在输出数组中的写入起始偏移量（基于tiles_touched的前缀和计算得到）
 * @param gaussian_keys_unsorted 输出：未排序的键数组
 * @param gaussian_values_unsorted 输出：未排序的值数组
 * @param radii 高斯点的半径
 * @param grid Tile网格的维度
 * @param rects (未使用)
 */
__global__ void duplicateWithKeys( 
	int P, 
	const float2* points_xy, 
	const float4* __restrict__ conic_opacity, 
	const float* depths, 
	const uint32_t* offsets, 
	uint64_t* gaussian_keys_unsorted, 
	uint32_t* gaussian_values_unsorted, 
	int* radii, 
	dim3 grid, 
	int2* rects) 
{
	auto idx = cg::this_grid().thread_rank();
	bool active =  true;
	if (idx >= P) 
	{
		active = false;
		idx = P - 1;
	}
	// 如果半径非正，说明该高斯点无效或已被粗略剔除
	if (radii[idx] <= 0) { active = false; }

	// 如果整个 Warp 的线程都不活跃，则直接返回
	if (__ballot_sync(WARP_MASK, active) == 0) { return; }

	// 获取当前高斯点在输出数组中的起始偏移量
	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
	// 获取下一个高斯点的偏移量，两者之差即为当前高斯点可能覆盖的Tile最大数量
	const uint32_t offset_to = offsets[idx];

	// 计算高斯点的屏幕空间包围盒对应的Tile范围
	uint2 rect_min, rect_max;
	getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

	const float2 xy = points_xy[idx];
	const float4 co = conic_opacity[idx];
	const float opacity_factor_threshold = logf(co.w / OPACITY_THRESHOLD);
	const uint32_t rect_width = (rect_max.x - rect_min.x);
	// 计算该高斯点覆盖的Tile总数（基于包围盒）
	const int32_t tile_count_init = (rect_max.y - rect_min.y) * rect_width;

	if (active) 
	{
		// 策略：对于覆盖Tile较少的高斯点，直接由当前线程顺序处理。
		// SEQUENTIAL_TILE_THRESH 是阈值。
		for (int tile_idx = 0; tile_idx < tile_count_init && tile_idx < SEQUENTIAL_TILE_THRESH && off < offset_to; tile_idx++) 
		{
			const int y = (tile_idx / rect_width) + rect_min.y;
			const int x = (tile_idx % rect_width) + rect_min.x;
			const glm::vec2 tile_min = {x * BLOCK_X, y * BLOCK_Y};
			const glm::vec2 tile_max = {(x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1};
			
			// 精确剔除：检查高斯在当前Tile内的最大贡献是否超过阈值
			glm::vec2 max_pos;
			float max_opac_factor = max_contrib_power_rect_gaussian_float(co, xy, tile_min, tile_max, max_pos);
			if (max_opac_factor <= opacity_factor_threshold) 
			{
				// 如果通过剔除测试，生成键值对
				uint64_t key = y * grid.x + x; // 高32位：Tile ID
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]); // 低32位：深度（用于排序）
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}

	// --- 负载均衡 (Load Balancing) ---
	// 如果某个高斯点覆盖的Tile非常多，单线程处理太慢，会阻塞整个Warp。
	// 这里使用协作组 (Cooperative Groups) 让整个 Warp 帮助处理这个“大”高斯点。

	const uint32_t lane_idx = cg::this_thread_block().thread_rank() % WARP_SIZE;
	unsigned int lane_mask_allprev_excl = 0xFFFFFFFFU >> (WARP_SIZE - lane_idx);
	// 判断是否需要协作处理
	const int32_t compute_cooperatively = active && tile_count_init > SEQUENTIAL_TILE_THRESH;
	// 统计warp中有多少线程需要协作
	const uint32_t remaining_threads = __ballot_sync(WARP_MASK, compute_cooperatively);

	// 如果没有线程需要协作，则填充剩余空间并返回
	if (remaining_threads == 0) 
	{ 
		// 填充无效数据，确保后续排序正常（虽然理论上 off 应该等于 offset_to 如果都处理完了）
		// 但这里 off < offset_to 可能发生是因为 tile_count_init 是基于包围盒的，而 offsets 是基于 tile_count_init 累加的
		// 但实际生成的键值对可能因为精确剔除而变少。
		// 实际上这里的逻辑是：offset_to 是分配的空间上限。
		while (off < offset_to) 
		{
			uint64_t key = (uint32_t) -1;
			key <<= 32;
			const float depth = FLT_MAX;
			key |= *((uint32_t*)&depth);
			gaussian_keys_unsorted[off] = key;
			gaussian_values_unsorted[off] = static_cast<uint32_t>(-1);
			off++;
		}
		return; 
	}

	// 处理需要协作的线程
	uint32_t n_remaining_threads = __popc(remaining_threads);
	for (int n = 0; n < n_remaining_threads && n < WARP_SIZE; n++) 
	{
		// 找到第 n 个需要协作的线程 ID (i)
		int i = __fns(remaining_threads, 0, n + 1); 
		// 广播该线程的数据给 Warp 中的所有线程
		uint32_t idx_i = __shfl_sync(WARP_MASK, idx, i);
		uint32_t off_i = __shfl_sync(WARP_MASK, off, i);
		const uint32_t offset_to_i = __shfl_sync(WARP_MASK, offset_to, i);

		const uint2 rect_min_i = make_uint2(__shfl_sync(WARP_MASK, rect_min.x, i), __shfl_sync(WARP_MASK, rect_min.y, i));
		const uint2 rect_max_i = make_uint2(__shfl_sync(WARP_MASK, rect_max.x, i), __shfl_sync(WARP_MASK, rect_max.y, i));
		const float2 xy_i = { __shfl_sync(WARP_MASK, xy.x, i), __shfl_sync(WARP_MASK, xy.y, i) };
		const float4 co_i = 
		{
			__shfl_sync(WARP_MASK, co.x, i),
			__shfl_sync(WARP_MASK, co.y, i),
			__shfl_sync(WARP_MASK, co.z, i),
			__shfl_sync(WARP_MASK, co.w, i),
		};
		const float opacity_factor_threshold_i = __shfl_sync(WARP_MASK, opacity_factor_threshold, i);
		const uint32_t rect_width_i = (rect_max_i.x - rect_min_i.x);
		const uint32_t tile_count_i = (rect_max_i.y - rect_min_i.y) * rect_width_i;

		// 计算剩余需要处理的 Tile
		const uint32_t remaining_tile_count = tile_count_i - SEQUENTIAL_TILE_THRESH;
		// Warp 循环迭代次数
		const int32_t n_iterations = (remaining_tile_count + WARP_SIZE - 1) / WARP_SIZE;
		
		for (int it = 0; it < n_iterations; it++) 
		{
			// Warp 中的每个线程领取一个 Tile 进行处理
			int tile_idx = it * WARP_SIZE + lane_idx + SEQUENTIAL_TILE_THRESH;
			int y = (tile_idx / rect_width_i) + rect_min_i.y;
			int x = (tile_idx % rect_width_i) + rect_min_i.x;
			const glm::vec2 tile_min = {x * BLOCK_X, y * BLOCK_Y};
			const glm::vec2 tile_max = {(x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1};
			glm::vec2 max_pos;
			
			// 每个线程判断自己负责的 Tile 是否通过剔除
			bool write = (tile_idx < tile_count_i) && (max_contrib_power_rect_gaussian_float(co_i, xy_i, tile_min, tile_max, max_pos) <= opacity_factor_threshold_i);
			
			// 投票：有多少线程需要写入
			const uint32_t write_ballot = __ballot_sync(WARP_MASK, write);
			const uint32_t n_writes = __popc(write_ballot);
			// 计算写入偏移：基地址 + 前面所有线程的写入量
			const uint32_t write_offset = off_i + __popc(write_ballot & lane_mask_allprev_excl);
			
			if (write && write_offset < offset_to_i) 
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx_i]);
				gaussian_keys_unsorted[write_offset] = key;
				gaussian_values_unsorted[write_offset] = idx_i;
			}
			// 更新全局偏移量（广播给下一次迭代）
			off_i += n_writes;
			// 如果当前是该高斯点的 owner 线程，更新自己的 off
			off += (i == lane_idx) * n_writes;
		}
	}

	// 填充剩余未使用的空间
	while (off < offset_to) 
	{
		uint64_t key = (uint32_t) -1;
		key <<= 32;
		const float depth = FLT_MAX;
		key |= *((uint32_t*)&depth);
		gaussian_keys_unsorted[off] = key;
		gaussian_values_unsorted[off] = static_cast<uint32_t>(-1);
		off++;
	}
}

// 识别每个Tile在排序后的列表中的起始和结束位置
// 输入必须是按 (TileID, Depth) 排序后的键列表
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L) return;

	// 读取当前 Key 的 Tile ID (高32位)
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	bool valid_tile = currtile != (uint32_t) -1;

	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		// 读取前一个 Key 的 Tile ID
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		// 如果 Tile ID 发生变化，说明上一个 Tile 结束，当前 Tile 开始
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx; // 上一个 Tile 的结束位置
			if (valid_tile) 
				ranges[currtile].x = idx; // 当前 Tile 的起始位置
		}
	}
	// 处理列表末尾
	if (idx == L - 1 && valid_tile)
		ranges[currtile].y = L;
}

// 计算每个 Tile 需要多少个 Bucket 来存储状态 (用于反向传播或其他辅助数据)
// 这里假设每个 Bucket 对应一个 Warp 处理的大小 (32)
__global__ void perTileBucketCount(int T, uint2* ranges, uint32_t* bucketCount) 
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= T)
		return;
	
	uint2 range = ranges[idx];
	int num_splats = range.y - range.x;
	// 向上取整计算需要的 bucket 数量
	int num_buckets = (num_splats + 31) / 32;
	bucketCount[idx] = (uint32_t) num_buckets;
}

// --- 状态类辅助函数：从显存块中分配各个缓冲区 ---

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N, size_t M)
{
	ImageState img;
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, M, 128);
	int* dummy;
	int* wummy;
	cub::DeviceScan::InclusiveSum(nullptr, img.scan_size, dummy, wummy, N);
	obtain(chunk, img.contrib_scan, img.scan_size, 128);

	obtain(chunk, img.max_contrib, N, 128);
	obtain(chunk, img.pixel_colors, N * NUM_CHAFFELS, 128);
	obtain(chunk, img.bucket_count, N, 128);
	obtain(chunk, img.bucket_offsets, N, 128);
	cub::DeviceScan::InclusiveSum(nullptr, img.bucket_count_scan_size, img.bucket_count, img.bucket_count, N);
	obtain(chunk, img.bucket_count_scanning_space, img.bucket_count_scan_size, 128);

	return img;
}

CudaRasterizer::SampleState CudaRasterizer::SampleState::fromChunk(char *& chunk, size_t C) {
	SampleState sample;
	obtain(chunk, sample.bucket_to_tile, C * BLOCK_SIZE, 128);
	obtain(chunk, sample.T, C * BLOCK_SIZE, 128);
	obtain(chunk, sample.ar, NUM_CHAFFELS * C * BLOCK_SIZE, 128);
	return sample;
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

__global__ void zero(int N, int* space)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx >= N)
		return;
	space[idx] = 0;
}

__global__ void set(int N, uint32_t* where, int* space)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx >= N)
		return;

	int off = (idx == 0) ? 0 : where[idx-1];

	space[off] = 1;
}

/**
 * @brief 前向渲染主函数。协调所有的CUDA内核执行。
 * 
 * 流程：
 * 1. 分配 Geometry, Image 缓冲区。
 * 2. **Preprocess**: 将3D高斯投影到2D，计算协方差、颜色、包围盒等。
 * 3. **Scan**: 计算所有高斯点覆盖的Tile总数 (InclusiveSum)。
 * 4. 分配 Binning 缓冲区。
 * 5. **DuplicateWithKeys**: 为每个高斯点在每个覆盖的Tile中生成一个 (TileID|Depth, GaussianID) 键值对。
 * 6. **Sort**: 对键值对进行基数排序，使同一Tile的高斯点聚集，并按深度排序。
 * 7. **IdentifyTileRanges**: 确定每个Tile在排序列表中的起止位置。
 * 8. 分配 Sample 缓冲区 (如果需要)。
 * 9. **Render**: 执行最终的光栅化渲染。
 * 
 * 输出：
 * 1. num_rendered: 总渲染实例数
 * 2. bucket_sum: 总桶数
 */
std::tuple<int,int> CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	std::function<char* (size_t)> sampleBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* dc,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const float limx_neg,
	const float limx_pos,
	const float limy_neg,
	const float limy_pos,
	const bool prefiltered,
	float* out_color,
	float* out_final_T,
	int* radii,
	bool debug, bool no_color) 
{
	if (NUM_CHAFFELS != 3 && colors_precomp == nullptr) 
	{ 
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!"); 
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	dim3 block(BLOCK_X, BLOCK_Y, 1); // 定义block大小为BLOCK_X*BLOCK_Y=16*16=256 threads
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1); // 定义tile_grid大小为(width/BLOCK_X, height/BLOCK_Y)

	// 1. 分配 Geometry 缓冲区， 此阶段主要是将高斯点从3D坐标系转换到2D坐标系，并计算协方差、颜色、包围盒等。
	size_t geom_chunk_size = required<GeometryState>(P);
	char* geom_chunkptr = geometryBuffer(geom_chunk_size); // 返回一个大小为geom_chunk_size的内存块，用于存储GeometryState，此内存块的起始地址被赋值给geom_chunkptr
	GeometryState geomState = GeometryState::fromChunk(geom_chunkptr, P); // 从geom_chunkptr中读取GeometryState，此函数会根据P的大小分配内存，并返回一个已占有内存的GeometryState对象

	// geomState 内部成员有 depths, clamped, internal_radii, means2D, cov3D, rgb, conic_opacity等高斯变量，每个都是一个大数组
	// 核函数被调用时，每个线程都会将自己负责的高斯点对应的变量值赋值给geomState的对应成员的大数组的对应位置

	// 2. 分配 Image 缓冲区，此阶段进行渲染
	size_t img_chunk_size = required<ImageState>(width * height, tile_grid.x * tile_grid.y);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height, tile_grid.x * tile_grid.y);

	// 3. 执行预处理 Kernel (在 forward.cu 中)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		dc,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		limx_neg,
		limx_pos,
		limy_neg,
		limy_pos,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered, 
		no_color
	), debug)

	// 4. 计算前缀和 point_offsets ，并确定总的渲染实例数 num_rendered (一个高斯点覆盖 N 个 Tile 算作 N 个实例)
	// 通过preprocess Kernel，我们已知了每个高斯的tiles_touched，我们将其累加得到point_offsets，即每个高斯点在“最终长列表”里的起始位置和结束位置
	/*
	例如：
		tiles_touched: [2, 1, 4]
		point_offsets (前缀和): [2, 3, 7]

		高斯 0：我要写 2 个键值对。写在长列表的下标 [0, 2)。
		高斯 1：我要写 1 个键值对。写在长列表的下标 [2, 3)。
		高斯 2：我要写 4 个键值对。写在长列表的下标 [3, 7)
	*/
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	// 5. 分配 Binning 缓冲区 (用于排序)
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// 6. 生成键值对 Kernel，每个线程处理一个高斯点，生成一个 (TileID|Depth, GaussianID) 键值对
	duplicateWithKeys <<<(P + 255) / 256, 256>>> (
		P,
		geomState.means2D,
		geomState.conic_opacity,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid,
		nullptr)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);  // 确定 Tile ID 需要的位数

	// 7. 对键值对进行排序 (按 Tile ID 主序，Depth 次序)
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// 8. 识别 Tile 范围 Kernel
	if (num_rendered > 0)
		identifyTileRanges <<<(num_rendered + 255) / 256, 256>>> (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// 9. 分配 Sample 缓冲区 (用于辅助数据)
	SampleState sampleState;
	unsigned int bucket_sum = 0;
	if (!no_color) 
	{
		int num_tiles = tile_grid.x * tile_grid.y;
		perTileBucketCount<<<(num_tiles + 255) / 256, 256>>>(num_tiles, imgState.ranges, imgState.bucket_count);
		CHECK_CUDA(cub::DeviceScan::InclusiveSum(imgState.bucket_count_scanning_space, imgState.bucket_count_scan_size, imgState.bucket_count, imgState.bucket_offsets, num_tiles), debug)
		CHECK_CUDA(cudaMemcpy(&bucket_sum, imgState.bucket_offsets + num_tiles - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost), debug);
		// create a state to store. size is number is the total number of buckets * block_size
		size_t sample_chunk_size = required<SampleState>(bucket_sum);
		char* sample_chunkptr = sampleBuffer(sample_chunk_size);
		sampleState = SampleState::fromChunk(sample_chunkptr, bucket_sum);
	}

	// 10. 执行最终渲染 Kernel (在 forward.cu 中)
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		imgState.bucket_offsets, 
		sampleState.bucket_to_tile,
		sampleState.T, 
		sampleState.ar,
		width, height,
		geomState.means2D,
		geomState.depths,
		feature_ptr,
		geomState.conic_opacity,
		imgState.n_contrib,
		imgState.max_contrib,
		background,
		out_color, out_final_T, no_color), debug)

	if (!no_color) 
	{
		// out_color -> imgState.pixel_colors
		CHECK_CUDA(cudaMemcpy(imgState.pixel_colors, out_color, sizeof(float) * width * height * NUM_CHAFFELS, cudaMemcpyDeviceToDevice), debug);
	}
	return std::make_tuple(num_rendered, bucket_sum);
}

void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R, int B,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* dc,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const float limx_neg,
	const float limx_pos,
	const float limy_neg,
	const float limy_pos,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	char* sample_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_ddc,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	const float lambda_erank,
	bool debug) 
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 block(BLOCK_X, BLOCK_Y, 1);
	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	
	// 恢复各个 State 对象
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height, tile_grid.x * tile_grid.y);
	SampleState sampleState = SampleState::fromChunk(sample_buffer, B);

	const float* feature_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	
	// 执行反向渲染 Kernel (在 backward.cu 中)
	// 计算对 2D 属性（颜色、不透明度、圆锥参数等）的梯度
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height, R, B,
		imgState.bucket_offsets,
		sampleState.bucket_to_tile,
		sampleState.T,
		sampleState.ar,
		background,
		geomState.means2D,
		geomState.depths,
		geomState.conic_opacity,
		feature_ptr,
		imgState.n_contrib,
		imgState.max_contrib,
		imgState.pixel_colors,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug)

	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	
	// 执行反向预处理 Kernel (在 backward.cu 中)
	// 将 2D 梯度反向传播回 3D 属性（位置、旋转、缩放、SH系数等）
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		dc,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		limx_neg,
		limx_pos,
		limy_neg,
		limy_pos,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_ddc,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		lambda_erank), debug)
}

