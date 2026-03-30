/**
 * @file da3_wrapper.h
 * @brief DA3 TorchScript 推理封装（关键帧 RGB + 位姿 + 内参 -> 深度/置信度）
 */

#pragma once

#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <Eigen/Eigen>

/**
 * @brief DA3 推理输出
 */
struct DA3Output
{
    torch::Tensor depth;   ///< (H, W) float32，预测深度
    torch::Tensor conf;    ///< (H, W) float32，预测置信度
    bool success = false;  ///< 推理是否成功
};

/**
 * @brief DA3 模型封装类
 */
class DA3Wrapper
{
public:
    explicit DA3Wrapper(const std::string& model_path);

    /**
     * @brief 执行 DA3 推理
     *
     * @param rgb (3, H, W) RGB 图像，值域 [0,1]
     * @param R_cw 世界到相机旋转
     * @param t_cw 世界到相机平移
     * @param fx,fy,cx,cy 相机内参
     * @return DA3Output 深度与置信度输出
     */
    DA3Output predict_depth(
        const torch::Tensor& rgb,
        const Eigen::Matrix3d& R_cw,
        const Eigen::Vector3d& t_cw,
        float fx,
        float fy,
        float cx,
        float cy
    );

    bool is_loaded() const { return is_loaded_; }

private:
    torch::jit::script::Module model_;
    bool is_loaded_ = false;
};

