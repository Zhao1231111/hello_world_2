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

#ifndef SIMPLEKNN_H_INCLUDED
#define SIMPLEKNN_H_INCLUDED

// 本头文件的公开接口直接使用 CUDA 的 float3，必须显式包含其类型定义。
// 不能依赖调用方偶然先包含 CUDACachingAllocator 等无关头文件提供传递依赖。
#include <vector_types.h>

class SimpleKNN
{
public:
	static void knn(int P, float3* points, float* meanDists);
};

#endif
