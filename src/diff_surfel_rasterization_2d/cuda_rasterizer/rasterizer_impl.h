/**
 * @file rasterizer_impl.h
 * @brief 2D高斯光栅化器底层实现的数据结构定义
 */

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	/**
	 * @brief 从连续内存块中按需获取指定类型的内存指针
	 * 
	 * @tparam T 指针指向的数据类型
	 * @param chunk 当前内存块的起始指针（会被更新到分配后的位置）
	 * @param ptr 接收分配后的内存指针
	 * @param count 分配元素的数量
	 * @param alignment 对齐字节数
	 */
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	/**
	 * @brief 几何预处理阶段需要的中间张量和状态
	 */
	struct GeometryState
	{
		size_t scan_size;         // 扫描过程需要的空间大小
		float* depths;            // 高斯点的深度值 (P)
		char* scanning_space;     // 显存前缀和扫描空间
		bool* clamped;            // 是否被裁剪标志
		int* internal_radii;      // 内部计算的投影半径 (P)
		float2* means2D;          // 投影后的 2D 中心位置 (P)
		float* transMat;          // 2D 变换矩阵 (P, 9) - 2DGS 特有
		float4* normal_opacity;   // 法线和不透明度 (P, 4)
		float* rgb;               // 转换后的 RGB 颜色 (P, 3)
		uint32_t* point_offsets;  // 每个点在分块排序列表中的偏移
		uint32_t* tiles_touched;  // 每个点覆盖的 Tile 数量

		/**
		 * @brief 从大块显存空间初始化几何状态指针
		 */
		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	/**
	 * @brief 渲染像素需要的状态和计数器
	 */
	struct ImageState
	{
		uint2* ranges;            // 每个 Tile 在排序列表中的起始和终止索引 (Tiles)
		uint32_t* n_contrib;      // 每个像素贡献的高斯点数量 (W*H)
		float* accum_alpha;       // 每个像素累积的透明度 (W*H)

		/**
		 * @brief 从大块显存空间初始化图像状态指针
		 */
		static ImageState fromChunk(char*& chunk, size_t N);
	};

	/**
	 * @brief 分块与排序阶段需要的状态
	 */
	struct BinningState
	{
		size_t sorting_size;               // 排序过程需要的临时空间大小
		uint64_t* point_list_keys_unsorted; // 排序前的 Key: [tile_id | depth]
		uint64_t* point_list_keys;          // 排序后的 Key
		uint32_t* point_list_unsorted;      // 排序前的高斯 ID
		uint32_t* point_list;               // 排序后的高斯 ID
		char* list_sorting_space;           // 显存排序临时空间

		/**
		 * @brief 从大块显存空间初始化分块状态指针
		 */
		static BinningState fromChunk(char*& chunk, size_t P);
	};

	/**
	 * @brief 计算给定点数下，该状态结构体所需的总显存大小
	 */
	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};