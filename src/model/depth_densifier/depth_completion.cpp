/**
 * @file depth_completion.cpp
 * @brief 深度补全流水线实现 (Gaussian-LIC2 Algorithm 1)
 */

#include "depth_completion.h"
#include "camera.h"
#include "loss_utils.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

bool detect_completion_failure(
    const torch::Tensor& sparse_depth,
    const torch::Tensor& dense_depth,
    float epsilon_1)
{
    // 获取有效 LiDAR 像素 mask
    auto valid_mask = sparse_depth > 0;
    
    if (valid_mask.sum().item<int64_t>() == 0) {
        // 没有有效 LiDAR 点，无法判断
        return false;
    }
    
    // 计算补全前后的平均深度
    float mean_before = sparse_depth.masked_select(valid_mask).mean().item<float>();
    float mean_after = dense_depth.masked_select(valid_mask).mean().item<float>();
    
    // 计算相对变化
    float change = std::abs(mean_after - mean_before) / mean_before;
    
    if (change > epsilon_1) {
        std::cout << "[DepthCompletion] Completion failed: mean change = " 
                  << change * 100 << "% > " << epsilon_1 * 100 << "%" << std::endl;
        return true;
    }
    
    return false;
}

torch::Tensor filter_depth_edges(
    const torch::Tensor& depth,
    float gradient_threshold)
{
    int H = depth.size(0);
    int W = depth.size(1);
    
    // 计算 x 方向梯度 (中心差分)
    auto grad_x = torch::zeros_like(depth);
    grad_x.index({torch::indexing::Slice(), torch::indexing::Slice(1, W-1)}) = 
        (depth.index({torch::indexing::Slice(), torch::indexing::Slice(2, W)}) -
         depth.index({torch::indexing::Slice(), torch::indexing::Slice(0, W-2)})) / 2.0f;
    
    // 计算 y 方向梯度 (中心差分)
    auto grad_y = torch::zeros_like(depth);
    grad_y.index({torch::indexing::Slice(1, H-1), torch::indexing::Slice()}) = 
        (depth.index({torch::indexing::Slice(2, H), torch::indexing::Slice()}) -
         depth.index({torch::indexing::Slice(0, H-2), torch::indexing::Slice()})) / 2.0f;
    
    // 计算梯度幅值
    auto grad_mag = torch::sqrt(grad_x.pow(2) + grad_y.pow(2));
    
    // 过滤高梯度区域
    auto valid_mask = grad_mag < gradient_threshold;
    
    return valid_mask;
}

torch::Tensor dilate_mask(
    const torch::Tensor& lidar_mask,
    int dilate_radius)
{
    // dilate_radius = 0 时不膨胀，直接返回原始 mask
    if (dilate_radius <= 0) {
        return lidar_mask;
    }
    
    int kernel_size = 2 * dilate_radius + 1;
    
    // 使用 max pooling 实现形态学膨胀
    auto mask_float = lidar_mask.to(torch::kFloat32).unsqueeze(0).unsqueeze(0);
    
    auto dilated = torch::nn::functional::max_pool2d(
        mask_float,
        torch::nn::functional::MaxPool2dFuncOptions({kernel_size, kernel_size})
            .stride({1, 1})
            .padding({dilate_radius, dilate_radius})
    );
    
    return dilated.squeeze(0).squeeze(0) > 0;
}

DepthCompletionResult sample_from_patches(
    const torch::Tensor& sparse_depth,
    const torch::Tensor& filtered_dense_depth,
    const torch::Tensor& dilated_lidar_mask,
    const torch::Tensor& rgb,
    const DepthCompletionParams& params)
{
    DepthCompletionResult result;
    result.success = true;
    
    int H = sparse_depth.size(0);
    int W = sparse_depth.size(1);
    int ps = params.patch_size;
    
    // 访问 CPU 进行采样
    auto sparse_cpu = sparse_depth.to(torch::kCPU);
    auto dense_cpu = filtered_dense_depth.to(torch::kCPU);
    auto mask_cpu = dilated_lidar_mask.to(torch::kCPU);
    auto rgb_cpu = rgb.to(torch::kCPU);
    
    auto sparse_acc = sparse_cpu.accessor<float, 2>();
    auto dense_acc = dense_cpu.accessor<float, 2>();
    auto mask_acc = mask_cpu.accessor<bool, 2>();
    auto rgb_acc = rgb_cpu.accessor<float, 3>();
    
    // 遍历所有 patch
    for (int py = 0; py < H; py += ps) {
        for (int px = 0; px < W; px += ps) {
            // 计算 patch 边界
            int py_end = std::min(py + ps, H);
            int px_end = std::min(px + ps, W);
            
            // 检查 patch 内是否有 LiDAR 点
            bool has_lidar = false;
            for (int y = py; y < py_end && !has_lidar; ++y) {
                for (int x = px; x < px_end && !has_lidar; ++x) {
                    if (sparse_acc[y][x] > 0) {
                        has_lidar = true;
                    }
                }
            }
            
            // 如果 patch 有 LiDAR 点，跳过
            if (has_lidar) {
                continue;
            }
            
            // 在 patch 内找最小深度的有效像素
            float min_depth = std::numeric_limits<float>::max();
            int min_u = -1, min_v = -1;
            
            for (int y = py; y < py_end; ++y) {
                for (int x = px; x < px_end; ++x) {
                    float d = dense_acc[y][x];
                    
                    // 检查条件:
                    // 1. 深度有效 (> 0)
                    // 2. 深度小于阈值 epsilon_2
                    // 3. 不在膨胀的 LiDAR mask 内
                    if (d > 0 && d < params.epsilon_2 && !mask_acc[y][x]) {
                        if (d < min_depth) {
                            min_depth = d;
                            min_u = x;
                            min_v = y;
                        }
                    }
                }
            }
            
            // 如果找到有效像素，加入补充列表
            if (min_u >= 0 && min_v >= 0) {
                result.supplement_pixels.push_back({min_u, min_v});
                result.supplement_depths.push_back(min_depth);
                result.supplement_colors.push_back({
                    rgb_acc[0][min_v][min_u],  // R
                    rgb_acc[1][min_v][min_u],  // G
                    rgb_acc[2][min_v][min_u]   // B
                });
            }
        }
    }
    
    return result;
}

DepthCompletionResult complete_depth_for_keyframe(
    SPNetWrapper& spnet,
    const torch::Tensor& rgb,
    const torch::Tensor& sparse_depth,
    const DepthCompletionParams& params)
{
    DepthCompletionResult result;
    result.success = false;
    
    // === Step 1: 调用 SPNet 进行深度补全 ===
    auto dense_depth = spnet.complete_depth(rgb, sparse_depth);
    result.dense_depth = dense_depth;
    
    // === Step 2: 失败检测 ===
    if (detect_completion_failure(sparse_depth, dense_depth, params.epsilon_1)) {
        std::cout << "[DepthCompletion] Completion failed, skipping supplementation" << std::endl;
        return result;
    }
    
    // === Step 3: 边缘过滤 ===
    auto edge_mask = filter_depth_edges(dense_depth, params.depth_grad_threshold);
    
    // 过滤后的稠密深度图
    auto filtered_dense_depth = dense_depth * edge_mask.to(torch::kFloat32);
    
    // 同时过滤超出最大深度的像素
    auto depth_valid_mask = (dense_depth > 0) & (dense_depth < params.max_depth);
    filtered_dense_depth = filtered_dense_depth * depth_valid_mask.to(torch::kFloat32);
    
    // 显式返回过滤深度图，供外部（gaussian初始化/扩展）直接计算像素法线
    result.filtered_dense_depth = filtered_dense_depth;
    result.filtered_mask = edge_mask & depth_valid_mask;
    
    // === Step 4: LiDAR Mask 膨胀 (可选) ===
    auto lidar_mask = sparse_depth > 0;
    auto dilated_lidar_mask = dilate_mask(lidar_mask, params.dilate_radius);
    
    // === Step 5: Patch 采样 ===
    auto sample_result = sample_from_patches(
        sparse_depth, filtered_dense_depth, dilated_lidar_mask, rgb, params
    );
    
    // 合并结果
    result.success = true;
    result.supplement_pixels = std::move(sample_result.supplement_pixels);
    result.supplement_depths = std::move(sample_result.supplement_depths);
    result.supplement_colors = std::move(sample_result.supplement_colors);
    
    std::cout << "[DepthCompletion] Sampled " << result.supplement_pixels.size() 
              << " supplement points from patches" << std::endl;
    
    return result;
}

// =========================================================================
// 新增 unified SPNet logic
// =========================================================================

// my main function
SPNetPointsResult get_spnet_points(
    SPNetWrapper& spnet,
    const torch::Tensor& rgb,
    const torch::Tensor& sparse_depth,
    const DepthCompletionParams& params,
    const std::shared_ptr<Camera>& cam)
{
    SPNetPointsResult res;
    
    // 1. 深度补全
    auto result = complete_depth_for_keyframe(spnet, rgb, sparse_depth, params);
    
    if (!result.success) {
        return res; 
    }
    
    // 2. 强制融合 LiDAR 点
    auto lidar_mask = sparse_depth > 0;
    // Use torch::where for safer replacement
    if (result.filtered_dense_depth.sizes() != sparse_depth.sizes()) {
        std::cerr << "[DepthCompletion] Shape mismatch: " << result.filtered_dense_depth.sizes() << " vs " << sparse_depth.sizes() << std::endl;
        return res;
    }
    torch::Tensor final_depth_map = torch::where(lidar_mask, sparse_depth, result.filtered_dense_depth);
    
    std::cout << "[DepthCompletion] Fused " << lidar_mask.sum().item<long>() << " LiDAR points." << std::endl;
    
    // 3. Determine valid pixels
    // Instead of using the full dense completion, we only want:
    // a) Existing LiDAR points
    // b) The specifically sampled supplement points
    
    auto supplement_mask = torch::zeros_like(lidar_mask);
    
    if (!result.supplement_pixels.empty()) {
        long num_supp = result.supplement_pixels.size();
        // Create tensor from vector coordinates
        auto opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
        torch::Tensor indices = torch::empty({num_supp, 2}, opts);
        auto acc = indices.accessor<long, 2>();
        for(size_t i=0; i<num_supp; ++i) {
            acc[i][0] = result.supplement_pixels[i].second; // y (row)
            acc[i][1] = result.supplement_pixels[i].first;  // x (col)
        }
        
        indices = indices.to(lidar_mask.device());
        supplement_mask.index_put_({indices.select(1, 0), indices.select(1, 1)}, true);
    }
    
    auto valid_pixel_mask = lidar_mask | supplement_mask;
    
    // Ensure we also respect the filter mask (edges, max depth) for safety, 
    // although supplement points are already filtered.
    valid_pixel_mask = valid_pixel_mask & (result.filtered_mask > 0);
    valid_pixel_mask = valid_pixel_mask & (final_depth_map > 0);
    
    auto flat_mask = valid_pixel_mask.view({-1}); // (N_pixels)
    
    // 4. Extract Data
    res.fused_depths = final_depth_map.view({-1, 1}).index({flat_mask});
    
    auto rgb_hwc = rgb.permute({1, 2, 0});
    res.fused_colors = rgb_hwc.view({-1, 3}).index({flat_mask});
    
    // 5. Back-project Points (Camera -> World)
    int H = rgb.size(1);
    int W = rgb.size(2);
    
    auto y_coords = torch::arange(H, torch::TensorOptions().device(rgb.device())).unsqueeze(1).repeat({1, W});
    auto x_coords = torch::arange(W, torch::TensorOptions().device(rgb.device())).unsqueeze(0).repeat({H, 1});
    
    auto valid_x = x_coords.view({-1}).index({flat_mask}).to(torch::kFloat32);
    auto valid_y = y_coords.view({-1}).index({flat_mask}).to(torch::kFloat32);
    auto valid_depth = res.fused_depths.flatten();
    
    // Store Fused Pixels (u, v)
    res.fused_pixels = torch::stack({valid_x, valid_y}, 1);

    auto x_cam = (valid_x - cam->cx_) * valid_depth / cam->fx_;
    auto y_cam = (valid_y - cam->cy_) * valid_depth / cam->fy_;
    auto z_cam = valid_depth;
    
    auto pts_cam = torch::stack({x_cam, y_cam, z_cam}, 1); // (N, 3)
    
    // Transform to World Space
    Eigen::Matrix3d R_wc_eigen = cam->R_cw_.transpose();
    Eigen::Vector3d t_wc_eigen = -R_wc_eigen * cam->t_cw_;
    
    torch::Tensor R_wc_tensor = torch::zeros({3, 3}, torch::TensorOptions().device(rgb.device()));
    torch::Tensor t_wc_tensor = torch::zeros({3}, torch::TensorOptions().device(rgb.device()));
    
    for(int i=0; i<3; ++i) {
        t_wc_tensor[i] = (float)t_wc_eigen(i);
        for(int j=0; j<3; ++j) {
            R_wc_tensor[i][j] = (float)R_wc_eigen(i, j);
        }
    }
    
    // P_w = P_c @ R_wc.T + t_wc
    res.fused_points = torch::matmul(pts_cam, R_wc_tensor.t()) + t_wc_tensor.unsqueeze(0);
    
    return res;
}

namespace {

inline std::string frameTagFromCamera(const std::shared_ptr<Camera>& cam)
{
    if (!cam) return "unknown";
    if (cam->image_name_.empty()) return "unknown";
    return fs::path(cam->image_name_).stem().string();
}

/**
 * @brief 保存深度可视化图（线性归一化 + colormap）
 */
void saveDepthVisualization(const torch::Tensor& depth, const std::string& out_path)
{
    if (!depth.defined() || depth.dim() != 2) return;

    auto depth_cpu = depth.to(torch::kCPU).contiguous();
    const int H = static_cast<int>(depth_cpu.size(0));
    const int W = static_cast<int>(depth_cpu.size(1));
    auto dep = depth_cpu.accessor<float, 2>();

    float min_valid = std::numeric_limits<float>::max();
    float max_valid = std::numeric_limits<float>::lowest();
    bool has_valid = false;
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            const float d = dep[v][u];
            if (std::isfinite(d) && d > 0.0f) {
                has_valid = true;
                min_valid = std::min(min_valid, d);
                max_valid = std::max(max_valid, d);
            }
        }
    }
    if (!has_valid) return;

    const float range = std::max(max_valid - min_valid, 1e-6f);
    cv::Mat gray(H, W, CV_8UC1, cv::Scalar(0));
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            const float d = dep[v][u];
            if (!std::isfinite(d) || d <= 0.0f) continue;
            const float n = std::clamp((d - min_valid) / range, 0.0f, 1.0f);
            gray.at<uint8_t>(v, u) = static_cast<uint8_t>(n * 255.0f);
        }
    }

    cv::Mat color;
    cv::applyColorMap(gray, color, cv::COLORMAP_TURBO);
    fs::create_directories(fs::path(out_path).parent_path());
    cv::imwrite(out_path, color);
}

/**
 * @brief DA3 深度与稀疏深度最小二乘对齐
 *
 * 模型：
 * 1) 仿射模型 y = a*x + b（优先）
 * 2) 纯尺度 y = a*x（当仿射出现明显负偏置/非正塌缩时自动回退）
 */
bool leastSquaresAlignDepthAffine(
    const torch::Tensor& pred_depth,      // (H, W)
    const torch::Tensor& sparse_depth,    // (H, W)
    const torch::Tensor& conf,            // (H, W), 可为空
    float conf_threshold,
    int min_points,
    float& a,
    float& b,
    torch::Tensor& aligned_depth,
    int& valid_count,
    std::string& align_mode)
{
    align_mode = "invalid";
    if (!pred_depth.defined() || !sparse_depth.defined() ||
        pred_depth.dim() != 2 || sparse_depth.dim() != 2 ||
        pred_depth.sizes() != sparse_depth.sizes()) {
        return false;
    }

    auto pred_cpu = pred_depth.to(torch::kCPU).contiguous();
    auto sparse_cpu = sparse_depth.to(torch::kCPU).contiguous();
    auto pred_a = pred_cpu.accessor<float, 2>();
    auto sparse_a = sparse_cpu.accessor<float, 2>();

    bool use_conf = conf.defined() && conf.dim() == 2 && conf.sizes() == pred_depth.sizes();
    torch::Tensor conf_cpu;
    const float* conf_ptr = nullptr;
    if (use_conf) {
        conf_cpu = conf.to(torch::kCPU).contiguous();
        conf_ptr = conf_cpu.data_ptr<float>();
    }

    // 收集匹配点并累计统计量
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    std::vector<float> y_samples;
    y_samples.reserve(16384);

    valid_count = 0;
    const int H = static_cast<int>(pred_cpu.size(0));
    const int W = static_cast<int>(pred_cpu.size(1));
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            const float x = pred_a[v][u];
            const float y = sparse_a[v][u];
            if (!std::isfinite(x) || !std::isfinite(y) || x <= 0.0f || y <= 0.0f) continue;
            if (use_conf && conf_threshold > 0.0f &&
                conf_ptr[static_cast<size_t>(v) * W + u] < conf_threshold) continue;
            sum_x += x;
            sum_y += y;
            sum_xx += static_cast<double>(x) * x;
            sum_xy += static_cast<double>(x) * y;
            y_samples.push_back(y);
            ++valid_count;
        }
    }
    if (valid_count < min_points) return false;

    // 求解纯尺度模型 y = a*x
    bool scale_ok = false;
    float a_scale = 1.0f;
    if (std::abs(sum_xx) >= 1e-12) {
        a_scale = static_cast<float>(sum_xy / sum_xx);
        scale_ok = std::isfinite(a_scale);
    }

    // 求解仿射模型 y = a*x + b
    bool affine_ok = false;
    float a_aff = 1.0f, b_aff = 0.0f;
    const double n = static_cast<double>(valid_count);
    const double denom = n * sum_xx - sum_x * sum_x;
    if (std::abs(denom) >= 1e-12) {
        a_aff = static_cast<float>((n * sum_xy - sum_x * sum_y) / denom);
        b_aff = static_cast<float>((sum_y - static_cast<double>(a_aff) * sum_x) / n);
        affine_ok = std::isfinite(a_aff) && std::isfinite(b_aff);
    }
    if (!affine_ok && !scale_ok) return false;

    // 根据仿射结果的稳定性判断是否回退到纯尺度
    float median_y = 1.0f;
    if (!y_samples.empty()) {
        auto mid_it = y_samples.begin() + y_samples.size() / 2;
        std::nth_element(y_samples.begin(), mid_it, y_samples.end());
        median_y = *mid_it;
    }

    float affine_nonpos_ratio = 0.0f;
    if (affine_ok) {
        int pred_pos_count = 0;
        int pred_nonpos_after_affine = 0;
        for (int v = 0; v < H; ++v) {
            for (int u = 0; u < W; ++u) {
                const float x = pred_a[v][u];
                if (!std::isfinite(x) || x <= 0.0f) continue;
                ++pred_pos_count;
                if (a_aff * x + b_aff <= 0.0f) ++pred_nonpos_after_affine;
            }
        }
        if (pred_pos_count > 0) {
            affine_nonpos_ratio =
                static_cast<float>(pred_nonpos_after_affine) / static_cast<float>(pred_pos_count);
        }
    }

    const bool bad_negative_bias =
        affine_ok && std::isfinite(median_y) && (median_y > 1e-6f) && (b_aff < -0.25f * median_y);
    const bool bad_nonpos_collapse = affine_ok && (affine_nonpos_ratio > 0.10f);
    const bool use_scale_fallback = scale_ok && (!affine_ok || bad_negative_bias || bad_nonpos_collapse);

    if (use_scale_fallback) {
        a = a_scale;
        b = 0.0f;
        align_mode = affine_ok ? "scale_only_fallback" : "scale_only";
    } else {
        a = a_aff;
        b = b_aff;
        align_mode = "affine";
    }

    aligned_depth = (pred_depth * a + b).clamp_min(0.0f);
    return true;
}

/**
 * @brief 由“对齐后深度 + 有效深度掩码”构建 DA3 法线图与有效法线掩码
 */
std::pair<torch::Tensor, torch::Tensor> buildDA3NormalMap(
    const torch::Tensor& aligned_depth,      // (H, W)
    const torch::Tensor& depth_valid_mask,   // (H, W) bool
    const std::shared_ptr<Camera>& cam,
    bool enforce_normal_face_camera)
{
    // depth_to_normal 输出 (3, H, W)，这里转成 (H, W, 3) 方便逐像素处理/采样。
    auto normal_chw = loss_utils::depth_to_normal(
        aligned_depth, cam->fx_, cam->fy_, cam->cx_, cam->cy_);
    auto normal_hwc = normal_chw.permute({1, 2, 0}).contiguous();  // (H, W, 3)

    // 法线自身有效性（防止零向量/异常值）
    auto normal_norm = torch::norm(normal_hwc, 2, 2, true);        // (H, W, 1)
    auto normal_self_valid = normal_norm > 1e-6f;                  // (H, W, 1)

    // 最终有效掩码：深度有效 + 法线自身有效
    auto valid_mask = depth_valid_mask.unsqueeze(2) & normal_self_valid;  // (H, W, 1)

    // 可选法线朝向约束：强制法线朝向相机（dot(n_cam, p_cam) < 0）。
    if (enforce_normal_face_camera) {
        const int H = static_cast<int>(aligned_depth.size(0));
        const int W = static_cast<int>(aligned_depth.size(1));
        auto opts = aligned_depth.options().dtype(torch::kFloat32);

        auto y_coords = torch::arange(0, H, opts).view({H, 1}).expand({H, W});
        auto x_coords = torch::arange(0, W, opts).view({1, W}).expand({H, W});
        auto d = aligned_depth.to(torch::kFloat32);

        auto x_cam = (x_coords - cam->cx_) * d / cam->fx_;
        auto y_cam = (y_coords - cam->cy_) * d / cam->fy_;
        auto p_cam = torch::stack({x_cam, y_cam, d}, 2);  // (H, W, 3)

        auto dot_np = (normal_hwc * p_cam).sum(2, true);  // (H, W, 1)
        auto need_flip = valid_mask & (dot_np > 0.0f);
        normal_hwc = torch::where(need_flip, -normal_hwc, normal_hwc);
    }

    // 归一化并清空无效像素法线（无效区域后续统一走视线法线回退）
    normal_hwc = torch::nn::functional::normalize(
        normal_hwc,
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(2));
    normal_hwc = normal_hwc * valid_mask.to(torch::kFloat32);

    return {normal_hwc, valid_mask.to(torch::kFloat32)};
}

}  // namespace

DA3GuidanceResult build_da3_guidance(
    DA3Wrapper& da3,
    const torch::Tensor& rgb,
    const torch::Tensor& sparse_depth,
    const std::shared_ptr<Camera>& cam,
    float conf_threshold,
    float min_depth,
    float max_depth,
    int align_min_points,
    bool enforce_normal_face_camera,
    bool save_depth_vis,
    const std::string& depth_vis_dir)
{
    DA3GuidanceResult out;
    if (!cam) return out;

    auto da3_out = da3.predict_depth(
        rgb, cam->R_cw_, cam->t_cw_, cam->fx_, cam->fy_, cam->cx_, cam->cy_);
    if (!da3_out.success || !da3_out.depth.defined()) {
        return out;
    }

    float a = 1.0f, b = 0.0f;
    int valid_points = 0;
    torch::Tensor aligned_depth;
    std::string align_mode;
    if (!leastSquaresAlignDepthAffine(
            da3_out.depth, sparse_depth, da3_out.conf, conf_threshold, align_min_points,
            a, b, aligned_depth, valid_points, align_mode)) {
        return out;
    }

    // 有效深度过滤条件（按你的要求）：min_depth < d < max_depth 且 conf >= threshold
    auto depth_valid_mask = (aligned_depth > min_depth) & (aligned_depth < max_depth);
    if (da3_out.conf.defined() && conf_threshold > 0.0f) {
        depth_valid_mask = depth_valid_mask & (da3_out.conf >= conf_threshold);
    }

    auto [normal_map, normal_valid_mask] = buildDA3NormalMap(
        aligned_depth, depth_valid_mask, cam, enforce_normal_face_camera);

    if (save_depth_vis && !depth_vis_dir.empty()) {
        const std::string frame_tag = frameTagFromCamera(cam);
        saveDepthVisualization(aligned_depth, depth_vis_dir + "/" + frame_tag + "_da3_aligned_ls.png");
    }

    const int valid_normals = normal_valid_mask.sum().item<int>();
    std::cout << "\033[1;34m [DA3] LS align a=" << std::fixed << std::setprecision(4) << a
              << ", b=" << b
              << ", mode=" << align_mode
              << ", valid=" << valid_points
              << ", normal_valid=" << valid_normals
              << "\033[0m" << std::endl;

    out.success = true;
    out.aligned_depth = aligned_depth;
    out.normal_map = normal_map;
    out.normal_valid_mask = normal_valid_mask;
    return out;
}

std::pair<torch::Tensor, torch::Tensor> sample_da3_guidance_by_pixels(
    const DA3GuidanceResult& guidance,
    const torch::Tensor& pixels)
{
    const int64_t N = pixels.defined() ? pixels.size(0) : 0;
    torch::Device device(torch::kCUDA);
    if (pixels.defined()) device = pixels.device();
    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    // 任何输入无效时，返回全无效结果（后续统一走视线法线回退）
    if (N <= 0 || !guidance.success ||
        !guidance.normal_map.defined() || !guidance.normal_valid_mask.defined()) {
        return {
            torch::zeros({std::max<int64_t>(N, 0), 3}, float_opts),
            torch::zeros({std::max<int64_t>(N, 0), 1}, float_opts)
        };
    }

    if (guidance.normal_map.dim() != 3 || guidance.normal_valid_mask.dim() != 3) {
        return {
            torch::zeros({N, 3}, float_opts),
            torch::zeros({N, 1}, float_opts)
        };
    }

    const int64_t H = guidance.normal_map.size(0);
    const int64_t W = guidance.normal_map.size(1);
    if (H <= 0 || W <= 0) {
        return {
            torch::zeros({N, 3}, float_opts),
            torch::zeros({N, 1}, float_opts)
        };
    }

    auto pix_long = pixels.to(torch::kLong);
    auto u = pix_long.index({torch::indexing::Slice(), 0}).clamp(0, W - 1);
    auto v = pix_long.index({torch::indexing::Slice(), 1}).clamp(0, H - 1);
    auto linear_idx = (v * W + u).to(torch::kLong);  // (N)

    auto normal_flat = guidance.normal_map.view({H * W, 3});
    auto valid_flat = guidance.normal_valid_mask.view({H * W, 1});

    auto sampled_normals = normal_flat.index_select(0, linear_idx).to(device).to(torch::kFloat32);
    auto sampled_valid_mask = valid_flat.index_select(0, linear_idx).to(device).to(torch::kFloat32);

    return {sampled_normals, sampled_valid_mask};
}
