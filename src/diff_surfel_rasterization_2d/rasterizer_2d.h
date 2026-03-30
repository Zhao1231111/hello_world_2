/*
 * 2D Gaussian Splatting Rasterizer - Autograd Layer
 * 
 * 本文件仿照 Gaussian-LIC 的 rasterizer.h 编写，
 * 为 2DGS 提供与 3DGS 完全一致的封装层次。
 */

#pragma once

#include <tuple>
#include <torch/torch.h>

#include "rasterize_points.h"

/**
 * @brief 2D高斯光栅化设置参数结构体
 * 
 * 封装渲染所需的所有相机参数和配置选项。
 * 与3DGS相比，2DGS不需要 limx/limy（视锥边界），因为使用不同的投影方式。
 */
struct GaussianRasterization2DSettings
{
    GaussianRasterization2DSettings(
        int image_height,
        int image_width,
        float tanfovx,
        float tanfovy,
        torch::Tensor& bg,
        float scale_modifier,
        torch::Tensor& viewmatrix,
        torch::Tensor& projmatrix,
        int sh_degree,
        torch::Tensor& campos,
        bool prefiltered,
        bool use_tile_culling,
        bool debug)
        : image_height_(image_height), image_width_(image_width), 
          tanfovx_(tanfovx), tanfovy_(tanfovy),
          bg_(bg), scale_modifier_(scale_modifier), 
          viewmatrix_(viewmatrix), projmatrix_(projmatrix),
          sh_degree_(sh_degree), campos_(campos), 
          prefiltered_(prefiltered), use_tile_culling_(use_tile_culling), debug_(debug)
    {}

    int image_height_;
    int image_width_;
    float tanfovx_;
    float tanfovy_;
    torch::Tensor bg_;
    float scale_modifier_;
    torch::Tensor viewmatrix_;
    torch::Tensor projmatrix_;
    int sh_degree_;
    torch::Tensor campos_;
    bool prefiltered_;
    bool use_tile_culling_;
    bool debug_;
};

/**
 * @brief 2D高斯光栅化自定义Autograd函数
 * 
 * 继承 torch::autograd::Function，实现自定义的前向和反向传播。
 * 这是连接 PyTorch 自动求导系统与 CUDA 底层实现的核心桥梁。
 */
class GaussianRasterizer2DFunction : public torch::autograd::Function<GaussianRasterizer2DFunction>
{
public:
    /**
     * @brief 前向传播
     * 
     * @param ctx 自动求导上下文
     * @param means3D 3D中心位置 (N, 3)
     * @param means2D 2D中心位置 (N, 2) - 用于梯度计算
     * @param sh 完整SH系数 (N, K, 3)
     * @param colors_precomp 预计算颜色（可选）
     * @param opacities 不透明度 (N, 1)
     * @param scales 2D缩放参数 (N, 2) - 与3DGS不同！
     * @param rotations 旋转参数 (N, 4)
     * @param transMat_precomp 预计算变换矩阵（可选）
     * @param raster_settings 渲染配置
     * 
     * @return tensor_list 包含: color(3,H,W), out_others(7,H,W), radii(N,)
     */
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor means3D,
        torch::Tensor means2D,
        torch::Tensor sh,
        torch::Tensor colors_precomp,
        torch::Tensor opacities,
        torch::Tensor scales,
        torch::Tensor rotations,
        torch::Tensor transMat_precomp,
        GaussianRasterization2DSettings raster_settings);

    /**
     * @brief 反向传播
     * 
     * @param ctx 前向传播保存的上下文
     * @param grad_outputs 从下层传播回来的梯度
     * 
     * @return tensor_list 所有输入参数的梯度
     */
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs);
};

/**
 * @brief 2D高斯光栅化器模块
 * 
 * 封装 GaussianRasterizer2DFunction，提供简洁的调用接口。
 * 与3DGS的 GaussianRasterizer 保持一致的用法。
 */
class GaussianRasterizer2D : public torch::nn::Module
{
public:
    GaussianRasterizer2D(GaussianRasterization2DSettings& raster_settings)
        : raster_settings_(raster_settings) {}

    /**
     * @brief 执行2D高斯光栅化渲染
     * 
     * @return tuple 包含: 
     *         0: color (3, H, W) - 渲染图像
     *         1: out_others (7, H, W) - 深度/法线/畸变等辅助输出
     *         2: radii (N,) - 投影半径
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor means3D,
        torch::Tensor means2D,
        torch::Tensor opacities,
        torch::Tensor sh,
        torch::Tensor colors_precomp,
        torch::Tensor scales,
        torch::Tensor rotations,
        torch::Tensor transMat_precomp);

public:
    GaussianRasterization2DSettings raster_settings_;
};
