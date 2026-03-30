/**
 * @file da3_wrapper.cpp
 * @brief DA3 TorchScript 推理实现
 */

#include "da3_wrapper.h"

#include <iostream>
#include <vector>

DA3Wrapper::DA3Wrapper(const std::string& model_path)
{
    try {
        model_ = torch::jit::load(model_path);
        model_.to(torch::kCUDA);
        model_.eval();
        is_loaded_ = true;
        std::cout << "[DA3Wrapper] Model loaded from: " << model_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "[DA3Wrapper] Failed to load model: " << e.what() << std::endl;
        is_loaded_ = false;
    }
}

DA3Output DA3Wrapper::predict_depth(
    const torch::Tensor& rgb,
    const Eigen::Matrix3d& R_cw,
    const Eigen::Vector3d& t_cw,
    float fx,
    float fy,
    float cx,
    float cy)
{
    DA3Output out;
    if (!is_loaded_) {
        std::cerr << "[DA3Wrapper] Model not loaded." << std::endl;
        return out;
    }

    try {
        const int H = static_cast<int>(rgb.size(1));
        const int W = static_cast<int>(rgb.size(2));

        auto rgb_cuda = rgb.to(torch::kCUDA).unsqueeze(0).contiguous(); // (1,3,H,W)

        torch::Tensor intr = torch::eye(
            3,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).unsqueeze(0);
        intr.index_put_({0, 0, 0}, fx);
        intr.index_put_({0, 1, 1}, fy);
        intr.index_put_({0, 0, 2}, cx);
        intr.index_put_({0, 1, 2}, cy);

        torch::Tensor ext = torch::eye(
            4,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).unsqueeze(0);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                ext.index_put_({0, r, c}, static_cast<float>(R_cw(r, c)));
            }
            ext.index_put_({0, r, 3}, static_cast<float>(t_cw(r)));
        }

        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(rgb_cuda);
        inputs.emplace_back(intr);
        inputs.emplace_back(ext);

        torch::IValue iv;
        {
            torch::NoGradGuard no_grad;
            iv = model_.forward(inputs);
        }

        if (!iv.isTuple()) {
            std::cerr << "[DA3Wrapper] Unexpected output type: not tuple." << std::endl;
            return out;
        }
        const auto& elems = iv.toTuple()->elements();
        if (elems.size() < 2 || !elems[0].isTensor() || !elems[1].isTensor()) {
            std::cerr << "[DA3Wrapper] Unexpected tuple layout." << std::endl;
            return out;
        }

        auto depth = elems[0].toTensor(); // (1,H,W)
        auto conf = elems[1].toTensor();  // (1,H,W)
        if (depth.dim() != 3 || conf.dim() != 3 ||
            depth.size(0) != 1 || conf.size(0) != 1 ||
            depth.size(1) != H || depth.size(2) != W) {
            std::cerr << "[DA3Wrapper] Unexpected tensor shape from model." << std::endl;
            return out;
        }

        out.depth = depth.squeeze(0).contiguous();
        out.conf = conf.squeeze(0).contiguous();
        out.success = true;
        return out;
    } catch (const c10::Error& e) {
        std::cerr << "[DA3Wrapper] Inference failed: " << e.what() << std::endl;
        return out;
    }
}

