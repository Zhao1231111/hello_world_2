#include "inference_wrapper.h"

// ====================== FeatureExtractor ======================

FeatureExtractor::FeatureExtractor(const std::string& model_path, int device_id)
    : device_(torch::kCUDA, device_id)
{
    try {
        module_ = torch::jit::load(model_path, device_);
        module_.eval();
        loaded_ = true;
    } catch (const c10::Error& e) {
        std::cerr << "[FeatureExtractor] Failed to load from: " << model_path << "\n" << e.what() << std::endl;
        loaded_ = false;
    }
}

torch::Tensor FeatureExtractor::extract(const torch::Tensor& image) {
    torch::NoGradGuard no_grad;
    if (!loaded_) {
        std::cerr << "[FeatureExtractor] Not loaded!" << std::endl;
        return torch::Tensor();
    }
    
    // image: (B, C, H, W)
    // module expects tensor or dict depending on export
    // Assuming wrapper export that takes tensor
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(image.to(device_));
    
    try {
        auto output = module_.forward(inputs).toTensor();
        return output;
    } catch (const c10::Error& e) {
        std::cerr << "[FeatureExtractor] Forward failed: " << e.what() << std::endl;
        return torch::Tensor();
    }
}

// ====================== GaussianRegressor ======================

GaussianRegressor::GaussianRegressor(const std::string& model_path, int device_id)
    : device_(torch::kCUDA, device_id)
{
    try {
        module_ = torch::jit::load(model_path, device_);
        module_.eval();
        loaded_ = true;
    } catch (const c10::Error& e) {
        std::cerr << "[GaussianRegressor] Failed to load from: " << model_path << "\n" << e.what() << std::endl;
        loaded_ = false;
    }
}

RegressedGaussianParams GaussianRegressor::regress(
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
) {
    torch::NoGradGuard no_grad;
    RegressedGaussianParams params;
    if (!loaded_) {
        std::cerr << "[GaussianRegressor] Not loaded!" << std::endl;
        return params;
    }
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(f_curr.to(device_));
    inputs.push_back(render_f_curr.to(device_));
    inputs.push_back(hist_f.to(device_));
    inputs.push_back(hist_render_f.to(device_));
    inputs.push_back(curr_dir.to(device_));
    inputs.push_back(hist_dir.to(device_));
    inputs.push_back(hist_mask.to(device_));
    inputs.push_back(inv_depth.to(device_));
    inputs.push_back(n_cam.to(device_));
    inputs.push_back(mask.to(device_));
    inputs.push_back(dis.to(device_));
    inputs.push_back(R_wc.to(device_));
    
    try {
        auto output = module_.forward(inputs);
        
        // Output format: (scale, rotation, opacity, color_dc, color_rest, delta_pixel, delta_inv_depth)
        // Check if output is tuple
        if (output.isTuple()) {
            auto elements = output.toTuple()->elements();
            if (elements.size() >= 5) {
                params.scale = elements[0].toTensor();
                params.rotation = elements[1].toTensor();
                params.opacity = elements[2].toTensor();
                params.color_dc = elements[3].toTensor();
                params.color_rest = elements[4].toTensor();
                
                if (elements.size() >= 7) {
                    params.delta_pixel = elements[5].toTensor();
                    params.delta_inv_depth = elements[6].toTensor();
                }
            } else {
                std::cerr << "[GaussianRegressor] Output tuple size mismatch: " << elements.size() << std::endl;
            }
        } else {
             std::cerr << "[GaussianRegressor] Output is not a tuple!" << std::endl;
        }
    } catch (const c10::Error& e) {
        std::cerr << "[GaussianRegressor] Forward failed: " << e.what() << std::endl;
    }
    
    return params;
}
