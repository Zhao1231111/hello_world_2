/**
 * @file spnet_wrapper.cpp
 * @brief SPNet 深度补全网络的 C++ 推理实现
 */

#include "spnet_wrapper.h"
#include <iostream>

SPNetWrapper::SPNetWrapper(const std::string& model_path, float max_depth)
    : max_depth_(max_depth), is_loaded_(false)
{
    try {
        // 加载 TorchScript 模型
        model_ = torch::jit::load(model_path);
        model_.to(torch::kCUDA);
        model_.eval();
        is_loaded_ = true;
        std::cout << "[SPNetWrapper] Model loaded from: " << model_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "[SPNetWrapper] Failed to load model: " << e.what() << std::endl;
        is_loaded_ = false;
    }
}

torch::Tensor SPNetWrapper::complete_depth(
    const torch::Tensor& rgb,
    const torch::Tensor& sparse_depth)
{
    if (!is_loaded_) {
        std::cerr << "[SPNetWrapper] Model not loaded!" << std::endl;
        return sparse_depth.clone();  // 返回原始稀疏深度
    }
    
    // 确保输入在 GPU 上
    auto rgb_cuda = rgb.to(torch::kCUDA);
    auto depth_cuda = sparse_depth.to(torch::kCUDA);
    
    // === 深度归一化 ===
    // SPNet 训练时使用归一化深度 [0, 1]（源自 16-bit PNG 格式）
    auto normalized_depth = depth_cuda / max_depth_;
    normalized_depth = normalized_depth.clamp(0.0f, 1.0f);
    
    // === 生成 hole mask ===
    // hole_raw: 有有效深度的位置为 1，无深度的位置为 0
    auto hole_mask = (sparse_depth > 0).to(torch::kFloat32).to(torch::kCUDA);
    
    // === 添加 batch 维度 ===
    // rgb: (3, H, W) -> (1, 3, H, W)
    // depth: (H, W) -> (1, 1, H, W)
    // mask: (H, W) -> (1, 1, H, W)
    auto rgb_batch = rgb_cuda.unsqueeze(0);
    auto depth_batch = normalized_depth.unsqueeze(0).unsqueeze(0);
    auto mask_batch = hole_mask.unsqueeze(0).unsqueeze(0);
    
    // === 模型推理 ===
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(rgb_batch);
    inputs.push_back(depth_batch);
    inputs.push_back(mask_batch);
    
    torch::Tensor output;
    {
        torch::NoGradGuard no_grad;
        output = model_.forward(inputs).toTensor();  // (1, 1, H, W)
    }
    
    // === 移除 batch 维度并反归一化 ===
    auto dense_depth = output.squeeze(0).squeeze(0);  // (H, W)
    dense_depth = dense_depth * max_depth_;
    
    // 确保深度非负
    dense_depth = dense_depth.clamp_min(0.0f);
    
    return dense_depth;
}
