/*
 * Copyright (C) 2025
 * 2D Gaussian Splatting Renderer for Gaussian-LIC
 *
 * This file provides the high-level rendering interface for 2D Gaussian Splatting,
 * mirroring the structure of the 3DGS renderer in rasterizer/renderer.cpp.
 */

#pragma once

#include <tuple>
#include <memory>
#include <torch/torch.h>

class Camera;
class GaussianModel;


struct RenderResult2D
{
    torch::Tensor rendered_image;
    torch::Tensor visibility_mask;
    torch::Tensor rendered_depth;
    torch::Tensor rendered_alpha;
    torch::Tensor rendered_normal;
    torch::Tensor rendered_distortion;
    torch::Tensor rendered_median_depth;
    torch::Tensor surf_normal;
    torch::Tensor radii;
    torch::Tensor screenspace_points;  // Carries gradients for densification statistics.
    torch::Tensor screenspace_points_mask;
    torch::Tensor screenspace_points_local;
    torch::Tensor geom_buffer;
    torch::Tensor binning_buffer;
    torch::Tensor img_buffer;
};


struct PoseLinearizationSettings2D
{
    double rot_epsilon = 1e-4;   // Radians.
    double trans_epsilon = 1e-4; // Meters.
    double alpha_threshold = 0.2;
    double huber_delta = 0.05;
    double gradient_epsilon = 1e-5;
    bool use_central_difference = true;
    int alpha_erode_radius = 1;  // Pixels.
};


struct PoseLinearization2D
{
    torch::Tensor jtj;
    torch::Tensor jtr;
    torch::Tensor valid_mask;
    torch::Tensor residual_rgb;
    double rmse = 0.0;
    int valid_pixels = 0;
    int alpha_pixels = 0;
    double alpha_coverage_ratio = 0.0;
};


RenderResult2D render_2d(
    const std::shared_ptr<Camera>& viewpoint_camera,
    const std::shared_ptr<GaussianModel>& pc,
    const torch::Tensor& bg_color,
    float scaling_modifier = 1.0f,
    bool use_tile_culling = true,
    bool debug_mode = false,
    const torch::Tensor& render_mask = torch::Tensor(),
    bool compute_extras = true
);


PoseLinearization2D linearize_pose_2d(
    const std::shared_ptr<Camera>& viewpoint_camera,
    const std::shared_ptr<GaussianModel>& pc,
    const torch::Tensor& gt_image,
    const torch::Tensor& bg_color,
    const RenderResult2D& base_render,
    const PoseLinearizationSettings2D& settings
);
