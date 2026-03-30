/**
 * @file forward.cu
 * @brief 2D高斯光栅化前向传播内核实现
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

/**
 * @brief 将 SH 系数转换为 RGB 颜色（设备函数）
 *
 * 根据查看方向和球谐函数(SH)系数计算高斯点的颜色。
 *
 * @param idx 高斯点的索引
 * @param deg 球谐函数的阶数 (0, 1, 2, 3)
 * @param max_coeffs 每个高斯点最大的 SH 系数数量 ((deg+1)^2)
 * @param means 所有高斯点的 3D 中心位置
 * @param campos 摄像机在世界坐标系下的位置
 * @param shs 指向 SH 系数数据的指针
 * @param clamped [输出] 标记 RGB 分量是否因为小于0而被钳制（用于反向传播）
 * @return glm::vec3 计算出的 RGB 颜色值 [0, 1]
 */
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

/**
 * @brief 计算从切平面到图像平面的 2D 到 2D 变换矩阵 (2DGS核心)
 * 
 * 该函数构建了一个 3x3 的单应矩阵（Homography），将 2D 高斯所在的切平面直接投影到像素坐标系。
 * 对应 2DGS 论文 Eq. (6)-(7)。
 * 
 * @param p_orig 高斯点的 3D 中心位置
 * @param scale 2D 缩放参数 (sx, sy)
 * @param mod 缩放修改因子
 * @param rot 旋转四元数 (w, x, y, z)
 * @param projmatrix 4x4 投影矩阵
 * @param viewmatrix 4x4 视图矩阵
 * @param W 图像宽度
 * @param H 图像高度
 * @param T [输出] 计算得到的 3x3 变换矩阵
 * @param normal [输出] 相机坐标系下的切平面法线
 */
__device__ void compute_transmat(
	const float3& p_orig,
	const glm::vec2 scale,
	float mod,
	const glm::vec4 rot,
	const float* projmatrix,
	const float* viewmatrix,
	const int W,
	const int H, 
	glm::mat3 &T,
	float3 &normal
) {
	// 1. 构建局部坐标系到世界坐标系的变换
	glm::mat3 R = quat_to_rotmat(rot);
	glm::mat3 S = scale_to_mat(scale, mod);
	glm::mat3 L = R * S; // 包含旋转和缩放的基向量

	// 2. 将高斯平面中心映射到世界坐标，基向量 L[0], L[1] 定义了平面
	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
	);

	// 3. 世界坐标到 NDC 坐标的投影矩阵
	glm::mat4 world2ndc = glm::mat4(
		projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
		projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
		projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
		projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
	);

	// 4. NDC 到像素坐标的变换（包含平移和缩放）
	glm::mat3x4 ndc2pix = glm::mat3x4(
		glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
		glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
		glm::vec4(0.0, 0.0, 0.0, 1.0)
	);

	// 5. 组合所有变换得到最终的 2D 投影矩阵 T
	T = glm::transpose(splat2world) * world2ndc * ndc2pix;
	
	// 6. 计算法线方向（相机坐标系下），用于后续光照或对齐计算
	normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);

}

/**
 * @brief 计算 2D 高斯的轴向包围盒 (AABB) 和中心深度
 * 
 * 通过解析变换矩阵 T，确定高斯点在屏幕上覆盖的像素区域。
 * 
 * @param T 3x3 变换矩阵
 * @param cutoff 截断距离（通常为 3 sigma）
 * @param point_image [输出] 投影后的像素中心坐标
 * @param extent [输出] 包围盒的半径（半宽和半高）
 * @return bool 如果计算成功返回 true，否则返回 false（如投影退化）
 */
__device__ bool compute_aabb(
	glm::mat3 T, 
	float cutoff,
	float2& point_image,
	float2& extent
) {
	// 使用解析方法求解二次曲线在屏幕上的范围
	glm::vec3 t = glm::vec3(cutoff * cutoff, cutoff * cutoff, -1.0f);
	float d = glm::dot(t, T[2] * T[2]);
	if (d == 0.0) return false;
	glm::vec3 f = (1 / d) * t;

	// 计算屏幕空间中心
	glm::vec2 p = glm::vec2(
		glm::dot(f, T[0] * T[2]),
		glm::dot(f, T[1] * T[2])
	);

	// 计算 AABB 范围
	glm::vec2 h0 = p * p - 
		glm::vec2(
			glm::dot(f, T[0] * T[0]),
			glm::dot(f, T[1] * T[1])
		);

	// 限制最小范围防止数值不稳定
	glm::vec2 h = sqrt(max(glm::vec2(1e-4, 1e-4), h0));
	point_image = {p.x, p.y};
	extent = {h.x, h.y};
	return true;
}

// Count number of tiles touched by Gaussian using warp-level parallelism
// Adapted from 3DGS computeTilebasedCullingTileCount
__device__ inline int computeTilebasedCullingTileCount2D(
	const bool active,
	const float3 Tu_init, const float3 Tv_init, const float3 Tw_init,
	const float cutoff_init,
	const uint2 rect_min_init, const uint2 rect_max_init)
{
	const int32_t tile_count_init = (rect_max_init.y - rect_min_init.y) * (rect_max_init.x - rect_min_init.x);
	int tile_count = 0;

	// Sequential phase for small number of tiles
	if (active)
	{
		const uint32_t rect_width = (rect_max_init.x - rect_min_init.x);
		for (int tile_idx = 0; tile_idx < tile_count_init && tile_idx < SEQUENTIAL_TILE_THRESH; tile_idx++)
		{
			const int y = (tile_idx / rect_width) + rect_min_init.y;
			const int x = (tile_idx % rect_width) + rect_min_init.x;
			if (check_tile_overlap({ x, y }, Tu_init, Tv_init, Tw_init, cutoff_init))
				tile_count++;
		}
	}

	const uint32_t lane_idx = cg::this_thread_block().thread_rank() % WARP_SIZE;
	const int32_t compute_cooperatively = active && tile_count_init > SEQUENTIAL_TILE_THRESH;
	const uint32_t remaining_threads = __ballot_sync(WARP_MASK, compute_cooperatively);

	if (remaining_threads == 0)
		return tile_count;

	// Cooperative phase for large number of tiles
	const uint32_t n_remaining_threads = __popc(remaining_threads);
	for (int n = 0; n < n_remaining_threads && n < WARP_SIZE; n++)
	{
		// Find the N-th thread that needs help
		int leader_idx = -1;
		int temp_mask = remaining_threads;
		for (int k = 0; k <= n; k++) {
			leader_idx = __ffs(temp_mask) - 1; 
			temp_mask &= ~(1 << leader_idx); 
		}
		const uint32_t i = leader_idx;

		// Broadcast leader's data to the warp
		const uint2 rect_min = make_uint2(__shfl_sync(WARP_MASK, rect_min_init.x, i), __shfl_sync(WARP_MASK, rect_min_init.y, i));
		const uint2 rect_max = make_uint2(__shfl_sync(WARP_MASK, rect_max_init.x, i), __shfl_sync(WARP_MASK, rect_max_init.y, i));
		
		const float3 Tu = shfl_float3(Tu_init, i);
		const float3 Tv = shfl_float3(Tv_init, i);
		const float3 Tw = shfl_float3(Tw_init, i);
		const float cutoff = __shfl_sync(WARP_MASK, cutoff_init, i);

		const uint32_t rect_width = (rect_max.x - rect_min.x);
		const uint32_t rect_tile_count = (rect_max.y - rect_min.y) * rect_width;
		const uint32_t remaining_rect_tile_count = rect_tile_count - SEQUENTIAL_TILE_THRESH;

		const int32_t n_iterations = (remaining_rect_tile_count + WARP_SIZE - 1) / WARP_SIZE;
		
		// Warp iterates over remaining tiles
		for (int it = 0; it < n_iterations; it++)
		{
			const int tile_idx = it * WARP_SIZE + lane_idx + SEQUENTIAL_TILE_THRESH;
			const int active_curr_it = tile_idx < rect_tile_count;

			const int y = (tile_idx / rect_width) + rect_min.y;
			const int x = (tile_idx % rect_width) + rect_min.x;

			bool overlaps = false;
			if (active_curr_it) {
				overlaps = check_tile_overlap({ x, y }, Tu, Tv, Tw, cutoff);
			}

			const uint32_t contributes_ballot = __ballot_sync(WARP_MASK, overlaps);
			const uint32_t n_contribute = __popc(contributes_ballot);

			// Only the leader adds to its count
			tile_count += (i == lane_idx) * n_contribute;
		}
	}

	return tile_count;
}

/**
 * @brief 前向预处理内核：为光栅化准备每个高斯点的数据
 * 
 * 遍历每个高斯点，执行裁剪、坐标变换、颜色计算、AABB 计算。
 */
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,      // 原始 3D 中心位置
	const glm::vec2* scales,       // 2D 缩放
	const float scale_modifier,    // 缩放修正
	const glm::vec4* rotations,    // 旋转四元数
	const float* opacities,        // 不透明度
	const float* shs,              // 球谐系数
	bool* clamped,                 // 颜色裁剪标记
	const float* transMat_precomp, // 预计算变换矩阵（可选）
	const float* colors_precomp,   // 预计算颜色（可选）
	const float* viewmatrix,       // 视图矩阵
	const float* projmatrix,       // 投影矩阵
	const glm::vec3* cam_pos,      // 相机位置
	const int W, int H,            // 图像尺寸
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,  // 焦距
	int* radii,                    // [输出] 屏幕投影半径（像素）
	float2* points_xy_image,       // [输出] 屏幕中心坐标
	float* depths,                 // [输出] 相机坐标系下的深度
	float* transMats,              // [输出] 计算出的变换矩阵 T
	float* rgb,                    // [输出] 计算出的 RGB 颜色
	float4* normal_opacity,        // [输出] 合并的法向(xyz)和不透明度(w)
	const dim3 grid,               // Tile 网格尺寸
	uint32_t* tiles_touched,       // [输出] 覆盖的 Tile 数量
	bool prefiltered,
	bool use_tile_culling)              
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// 初始化为 0，如果未通过裁剪，则后续不会处理此点
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// 1. 视锥体剔除：检查点是否在相机可见范围内
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	
	// 2. 计算变换矩阵 T：将高斯从切平面投影到屏幕
	glm::mat3 T;
	float3 normal;
	if (transMat_precomp == nullptr)
	{
		compute_transmat(((float3*)orig_points)[idx], scales[idx], scale_modifier, rotations[idx], projmatrix, viewmatrix, W, H, T, normal);
		float3 *T_ptr = (float3*)transMats;
		T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
		T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
		T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};
	} else {
		// 使用预计算矩阵
		glm::vec3 *T_ptr = (glm::vec3*)transMat_precomp;
		T = glm::mat3(
			T_ptr[idx * 3 + 0], 
			T_ptr[idx * 3 + 1],
			T_ptr[idx * 3 + 2]
		);
		normal = make_float3(0.0, 0.0, 1.0);
	}

#if DUAL_VISIABLE
	// 处理双面可见性：确保法线朝向相机
	float cos = -sumf3(p_view * normal);
	if (cos == 0) return;
	float multiplier = cos > 0 ? 1: -1;
	normal = multiplier * normal;
#endif

#if TIGHTBBOX
	// 根据不透明度动态调整包围盒大小，加速渲染
	float cutoff = sqrtf(max(9.f + 2.f * logf(opacities[idx]), 0.000001));
#else
	float cutoff = 3.0f; // 标准 3-sigma 截断
#endif

	// 3. 计算 AABB：确定该高斯在屏幕上覆盖的矩形区域
	float2 point_image;
	float radius;
	{
		float2 extent;
		bool ok = compute_aabb(T, cutoff, point_image, extent);
		if (!ok) return;
		radius = ceil(max(max(extent.x, extent.y), cutoff * FilterSize));
	}

	// 4. 计算覆盖的 Tile：使用精确剔除算法
	uint2 rect_min, rect_max;
	getRect(point_image, radius, rect_min, rect_max, grid);
	
	// 基本 AABB 检查
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	//======= for tile-based culling =======
	if(use_tile_culling)
	{
	// 使用 Ray-Splat Intersection 进行精确 Tile 剔除
	// 提取 T 矩阵的列向量 (Tu, Tv, Tw)
		float3 Tu = {T[0][0], T[0][1], T[0][2]};
		float3 Tv = {T[1][0], T[1][1], T[1][2]};
		float3 Tw = {T[2][0], T[2][1], T[2][2]};

		int touched = computeTilebasedCullingTileCount2D(
			true, Tu, Tv, Tw, cutoff, rect_min, rect_max
		);

		if (touched == 0)
			return;

			// 5. 计算颜色：如果使用了球谐函数(SH)，则转换为 RGB
		if (colors_precomp == nullptr) {
			glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
			rgb[idx * C + 0] = result.x;
			rgb[idx * C + 1] = result.y;
			rgb[idx * C + 2] = result.z;
		}

		// 6. 存储预处理结果
		depths[idx] = p_view.z;
		radii[idx] = (int)radius;
		points_xy_image[idx] = point_image;
		normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
		tiles_touched[idx] = touched; // for tile-based culling

	}
	else
	{
		// 5. 计算颜色：如果使用了球谐函数(SH)，则转换为 RGB
		if (colors_precomp == nullptr) {
			glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
			rgb[idx * C + 0] = result.x;
			rgb[idx * C + 1] = result.y;
			rgb[idx * C + 2] = result.z;
		}

		// 6. 存储预处理结果
		depths[idx] = p_view.z;
		radii[idx] = (int)radius;
		points_xy_image[idx] = point_image;
		normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
		tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	}
}

/**
 * @brief 核心光栅化渲染内核
 * 
 * 使用 Tile-based 渲染：
 * 1. 同一个 Block (Tile) 的线程协作从全局内存批量读取高斯数据到 Shared Memory。
 * 2. 线程独立计算各自像素的 Alpha 混合与颜色合成。
 * 3. 支持输出深度图、法线图、畸变图等辅助信息。
 */
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,            // 每个 Tile 覆盖的高斯点范围线
	const uint32_t* __restrict__ point_list,      // 排序后的高斯点 ID 列表
	int W, int H,                                // 图像尺寸
	float focal_x, float focal_y,                // 焦距
	const float2* __restrict__ points_xy_image,  // 预处理得到的屏幕中心
	const float* __restrict__ features,           // 颜色数据 (RGB/SH)
	const float* __restrict__ transMats,          // 投影变换矩阵 T (3x3)
	const float* __restrict__ depths,             // 每个高斯点的深度
	const float4* __restrict__ normal_opacity,    // 法向与不透明度
	float* __restrict__ final_T,                  // [输出] 最终透射率图 (用于反向传播)
	uint32_t* __restrict__ n_contrib,             // [输出] 每个像素贡献的高斯点数量
	const float* __restrict__ bg_color,           // 背景颜色
	float* __restrict__ out_color,                // [输出] 渲染生成的 RGB 图像
	float* __restrict__ out_others)               // [输出] 辅助输出 (depth, normal, distortion, etc.)
{
	// 1. 确定当前像素坐标和 Tile
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y};

	// 检查像素是否在合法范围内
	bool inside = pix.x < W&& pix.y < H;
	bool done = !inside;

	// 获取当前 Tile 在排序列表中的起始和终止点
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// 使用共享内存加速数据访问
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];

	// 初始化混合参数
	float T = 1.0f; // 透射率（初始为1，代表全透明）
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 }; // 累积颜色

#if RENDER_AXUTILITY
	// 初始化辅助渲染输出
	float N[3] = {0}; // 累积法线
	float D = { 0 }; // 深度
	float M1 = {0}; // 畸变计算中间量
	float M2 = {0};
	float distortion = {0}; // 畸变图
	float median_depth = {0}; // 中值深度
	float median_contributor = {-1};
#endif

	// 2. 遍历高斯点：分块加载并计算
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// 协作加载数据到共享内存
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE) break;

		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
		}
		block.sync();

		// 处理当前共享内存批次中的每个高斯
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor++;

			// 3. 计算 2DGS 的几何贡献 (Ray-Splat Intersection)
			// 利用齐次坐标系下的单应矩阵计算相交。对应 2DGS Eq. (8)-(10)。
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
			
			// 获取局部 u-v 坐标
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 
			
			// 4. 应用低通滤波器防止走样
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 
			float rho = min(rho3d, rho2d); // 2DGS 平滑项

			// 5. 计算混合后的 Alpha
			float depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z; // 相交点深度
			if (depth < near_n) continue;

			float4 nor_o = collected_normal_opacity[j];
			float opa = nor_o.w;

			float power = -0.5f * rho;
			if (power > 0.0f) continue;

			float alpha = min(0.99f, opa * exp(power));
			if (alpha < 1.0f / 255.0f) continue;
			
			// 检查透射率是否已饱和（早停优化）
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// 6. 累积颜色和其他属性 (Alpha Blending)
			float w = alpha * T;
#if RENDER_AXUTILITY
			// 按照 2DGS 论文计算畸变、深度、法向输出
			float A = 1-T;
			float m = far_n / (far_n - near_n) * (1 - near_n / depth);
			distortion += (m * m * A + M2 - 2 * m * M1) * w;
			D  += depth * w;
			M1 += m * w;
			M2 += m * m * w;

			if (T > 0.5) {
				median_depth = depth;
				median_contributor = contributor;
			}
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * w;
#endif

			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
			T = test_T;

			last_contributor = contributor;
		}
	}

	// 7. 写入最终像素颜色
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

#if RENDER_AXUTILITY
		// 写入 2DGS 特有的辅助 Map
		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = M1;
		final_T[pix_id + 2 * H * W] = M2;
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
#endif
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* means2D,
	const float* colors,
	const float* transMats,
	const float* depths,
	const float4* normal_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_others)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		means2D,
		colors,
		transMats,
		depths,
		normal_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_others);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
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
	const int W, const int H,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool use_tile_culling)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		transMats,
		rgb,
		normal_opacity,
		grid,
		tiles_touched,
		prefiltered,
		use_tile_culling
		);
}
