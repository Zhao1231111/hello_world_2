#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>
#include <iostream>

class FeatureExtractor {
public:
    FeatureExtractor(const std::string& model_path, int device_id = 0);
    
    // Returns feature map: (1, C, H, W) or specific output shape
    torch::Tensor extract(const torch::Tensor& image);

    bool is_loaded() const { return loaded_; }

private:
    torch::jit::script::Module module_;
    torch::Device device_;
    bool loaded_ = false;
};

struct RegressedGaussianParams {
    torch::Tensor scale;      // (N, 2)
    torch::Tensor rotation;   // (N, 4) - Final World Space Quaternion
    torch::Tensor opacity;    // (N, 1)
    torch::Tensor color_dc;   // (N, 1, 3)
    torch::Tensor color_rest; // (N, 15, 3)
    torch::Tensor delta_pixel;     // (N, 2)
    torch::Tensor delta_inv_depth; // (N, 1)
};

class GaussianRegressor {
public:
    GaussianRegressor(const std::string& model_path, int device_id = 0);

    // Batch inference
    RegressedGaussianParams regress(
        const torch::Tensor& f_curr,
        const torch::Tensor& render_f_curr,
        const torch::Tensor& hist_f,
        const torch::Tensor& hist_render_f,
        const torch::Tensor& curr_dir,
        const torch::Tensor& hist_dir,
        const torch::Tensor& hist_mask,
        const torch::Tensor& inv_depth,
        const torch::Tensor& n_cam,
        const torch::Tensor& mask,
        const torch::Tensor& dis,
        const torch::Tensor& R_wc
    );

    bool is_loaded() const { return loaded_; }

private:
    torch::jit::script::Module module_;
    torch::Device device_;
    bool loaded_ = false;
};
