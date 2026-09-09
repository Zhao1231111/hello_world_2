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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16

// 前向/反向像素内核原本就使用该阈值跳过不可见贡献；预处理阶段复用同一常量，
// 确保被提前剔除的高斯在任何像素上本来也不会进入 alpha blending。
#define ALPHA_THRESHOLD (1.0f / 255.0f)

#endif
