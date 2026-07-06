/*
 * 2D Gaussian Splatting Rasterizer - Autograd Layer
 */

#pragma once

#include <tuple>
#include <torch/torch.h>

#include "rasterize_points.h"


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


class GaussianRasterizer2DFunction : public torch::autograd::Function<GaussianRasterizer2DFunction>
{
public:

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


    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs);
};


class GaussianRasterizer2D : public torch::nn::Module
{
public:
    GaussianRasterizer2D(GaussianRasterization2DSettings& raster_settings)
        : raster_settings_(raster_settings) {}


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
