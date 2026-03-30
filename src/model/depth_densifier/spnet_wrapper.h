/**
 * @file spnet_wrapper.h
 * @brief SPNet 深度补全网络的 C++ 推理封装
 * 
 * 该类封装了 SPNet TorchScript 模型的加载和推理接口，
 * 用于将稀疏 LiDAR 深度图补全为稠密深度图。
 */

#pragma once

#include <string>
#include <torch/script.h>
#include <torch/torch.h>

/**
 * @brief SPNet 深度补全网络封装类
 * 
 * 封装 TorchScript 模型的加载和推理，处理输入输出的归一化。
 * 
 * 输入格式:
 *   - rgb: (3, H, W) float32 [0, 1]
 *   - sparse_depth: (H, W) float32 米制深度
 * 
 * 输出格式:
 *   - dense_depth: (H, W) float32 米制深度
 */
class SPNetWrapper
{
public:
    /**
     * @brief 构造函数，加载 TorchScript 模型
     * @param model_path TorchScript 模型路径 (.pt 文件)
     * @param max_depth 最大深度值，用于归一化 (室外=100m, 室内=20m)
     */
    SPNetWrapper(const std::string& model_path, float max_depth = 100.0f);
    
    /**
     * @brief 执行深度补全推理
     * 
     * @param rgb RGB 图像张量 (3, H, W) [0, 1]
     * @param sparse_depth 稀疏深度图 (H, W) 米制
     * @return torch::Tensor 稠密深度图 (H, W) 米制
     * 
     * @note 内部自动处理深度归一化和反归一化
     */
    torch::Tensor complete_depth(
        const torch::Tensor& rgb,
        const torch::Tensor& sparse_depth
    );
    
    /**
     * @brief 检查模型是否已加载
     */
    bool is_loaded() const { return is_loaded_; }
    
    /**
     * @brief 获取最大深度值
     */
    float get_max_depth() const { return max_depth_; }
    
    /**
     * @brief 设置最大深度值
     */
    void set_max_depth(float max_depth) { max_depth_ = max_depth; }

private:
    torch::jit::script::Module model_;  ///< TorchScript 模型
    float max_depth_;                    ///< 最大深度值 (用于归一化)
    bool is_loaded_;                     ///< 模型是否加载成功
};
