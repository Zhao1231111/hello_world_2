#pragma once

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "camera.h"
#include "tensor_utils.h"
#include "gaussian.h"
#include "model/depth_densifier/depth_completion.h"
#include "diff_surfel_rasterization_2d/renderer_2d.h"


// 判断像素是否在图像范围内
inline bool inPixelBounds(int u, int v, int W, int H)
{
    return u >= 0 && u < W && v >= 0 && v < H;
}

// 根据世界坐标系法线构造旋转四元数：
// 将局部 +Z 轴旋转到 normal_w 方向，返回 (w, x, y, z)
inline Eigen::Vector4d quaternionFromNormalWorld(const Eigen::Vector3d& normal_w)
{
    if (normal_w.norm() < 1e-8) {
        return Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
    }
    Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(
        Eigen::Vector3d::UnitZ(), normal_w.normalized()
    );
    q.normalize();
    return Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
}

/**
 * @brief 将 LiDAR 点云投影到图像平面生成稀疏深度图
 * 
 * 对应 Phase 1: 处理 LiDAR 点云
 * 
 * @param dataset 数据集
 * @param cam 目标相机
 * @return torch::Tensor 稀疏深度图 (H, W)
 */
inline torch::Tensor process_lidar_projection(
    const std::shared_ptr<Dataset>& dataset,
    const std::shared_ptr<Camera>& cam)
{
    int H = cam->image_height_;
    int W = cam->image_width_;
    
    // 构建稀疏深度图 (处理多点投影到同一像素的情况)
    // Key: "u_v", Value: (point_index, depth) 保留最近的
    std::unordered_map<std::string, std::pair<int, float>> pixel_depth_map;
    
    // 遍历点云
    for (size_t i = 0; i < dataset->pointcloud_.size(); ++i) {
        const auto& pt = dataset->pointcloud_[i];
        Eigen::Vector3d pt_c = cam->R_cw_ * pt + cam->t_cw_;
        if (pt_c.z() <= 0) continue;  // 相机后方跳过
        
        int u = static_cast<int>(pt_c.x() * cam->fx_ / pt_c.z() + cam->cx_);
        int v = static_cast<int>(pt_c.y() * cam->fy_ / pt_c.z() + cam->cy_);
        
        if (u >= 0 && u < W && v >= 0 && v < H) {
            std::string key = std::to_string(u) + "_" + std::to_string(v);
            float depth = static_cast<float>(pt_c.z());
            // 保留最近的深度
            if (!pixel_depth_map.count(key) || depth < pixel_depth_map[key].second) {
                // Keep original index if needed, but here we just need depth map.
                pixel_depth_map[key] = {static_cast<int>(i), depth};
            }
        }
    }
    
    // 将 map 转换为稀疏深度图 Tensor
    torch::Tensor sparse_depth = torch::zeros({H, W}, torch::kFloat32).cuda();
    
    for (const auto& [key, val] : pixel_depth_map) {
        size_t pos = key.find('_');
        int u = std::stoi(key.substr(0, pos));
        int v = std::stoi(key.substr(pos + 1));
        sparse_depth.index_put_({v, u}, val.second);
    }
    
    return sparse_depth;
}

/**
 * @brief 根据法线计算旋转四元数 (Geometric Initialization)
 * 
 * 1. 若法线有效，使用法线方向进行旋转初始化；
 * 2. 若法线无效，统一回退为“视线法线”：
 *    n_cam = -normalize(p_cam)，其中 p_cam 为该像素对应的相机系射线方向。
 * 
 * 这样可保证所有回退路径语义一致，不再出现“随机四元数”或“单位四元数”
 * 两种互相冲突的行为，便于后续优化稳定收敛。
 * 
 * @param normals_c 相机坐标系下的法线 (N, 3)
 * @param valid_mask 法线有效性掩码 (N, 1) [0/1]
 * @param pixels 像素坐标 (N, 2)，顺序为 (u, v)
 * @param depths 深度值 (N) 或 (N, 1)，用于回退时构造视线方向
 * @param cam 相机对象 (用于获取 R_cw)
 * @return torch::Tensor 旋转四元数 (N, 4) [w, x, y, z] 世界坐标系
 */
inline torch::Tensor compute_rotation_from_normals(
    const torch::Tensor& normals_c,
    const torch::Tensor& valid_mask,
    const torch::Tensor& pixels,
    const torch::Tensor& depths,
    const std::shared_ptr<Camera>& cam)
{
    int num = normals_c.size(0);
    
    // Move to CPU for geometric computation (Eigen)
    // (Can be done in CUDA/Tensor, but Quat conversion is tricky without custom kernel)
    auto n_c_cpu = normals_c.cpu().contiguous();
    auto mask_cpu = valid_mask.cpu().contiguous();
    auto pixels_cpu = pixels.cpu().contiguous().to(torch::kFloat32);

    torch::Tensor depth_flat = depths;
    if (depth_flat.dim() == 2 && depth_flat.size(1) == 1) {
        depth_flat = depth_flat.squeeze(1);
    }
    depth_flat = depth_flat.cpu().contiguous().to(torch::kFloat32);
    
    auto n_c_acc = n_c_cpu.accessor<float, 2>();
    auto mask_acc = mask_cpu.accessor<float, 2>();
    auto pixel_acc = pixels_cpu.accessor<float, 2>();
    auto depth_acc = depth_flat.accessor<float, 1>();
    
    // Output tensor
    torch::Tensor rots = torch::zeros({num, 4}, torch::kFloat32); // CPU first
    auto rots_acc = rots.accessor<float, 2>();
    
    // Camera to World Rotation
    Eigen::Matrix3d R_wc = cam->R_cw_.transpose();
    
    for (int i = 0; i < num; ++i) {
        // ---------- 1) 先尝试使用输入法线 ----------
        bool normal_valid = (mask_acc[i][0] > 0.5f);
        Eigen::Vector3d n_c(
            static_cast<double>(n_c_acc[i][0]),
            static_cast<double>(n_c_acc[i][1]),
            static_cast<double>(n_c_acc[i][2]));
        if (!std::isfinite(n_c.x()) || !std::isfinite(n_c.y()) || !std::isfinite(n_c.z()) || n_c.norm() < 1e-8) {
            normal_valid = false;
        }

        // ---------- 2) 无效时统一回退到“视线法线” ----------
        if (!normal_valid) {
            const double u = static_cast<double>(pixel_acc[i][0]);
            const double v = static_cast<double>(pixel_acc[i][1]);
            double d = static_cast<double>(depth_acc[i]);
            if (!std::isfinite(d) || d <= 1e-6) d = 1.0;

            Eigen::Vector3d p_cam(
                (u - cam->cx_) * d / cam->fx_,
                (v - cam->cy_) * d / cam->fy_,
                d);

            // 极端情况下（例如深度异常）保证仍有稳定回退方向
            if (!std::isfinite(p_cam.x()) || !std::isfinite(p_cam.y()) || !std::isfinite(p_cam.z()) || p_cam.norm() < 1e-8) {
                p_cam = Eigen::Vector3d(0.0, 0.0, 1.0);
            }
            n_c = -p_cam.normalized();
        } else {
            n_c.normalize();
        }

        // ---------- 3) 转到世界系并构造四元数 ----------
        Eigen::Vector3d n_w = R_wc * n_c;
        Eigen::Vector4d q = quaternionFromNormalWorld(n_w);
        rots_acc[i][0] = static_cast<float>(q(0));
        rots_acc[i][1] = static_cast<float>(q(1));
        rots_acc[i][2] = static_cast<float>(q(2));
        rots_acc[i][3] = static_cast<float>(q(3));
    }
    
    return rots.cuda();
}

/**
 * @brief 提取当前帧特征 (如果尚未提取)
 */
inline void extract_features(
    GaussianModel* pc,
    const std::shared_ptr<Camera>& cam)
{
    if (pc->feature_extractor_ && pc->feature_extractor_->is_loaded() && !cam->feature_map_.defined()) {
        torch::NoGradGuard no_grad; // Prevent computation graph tracking
        auto img_tensor = cam->original_image_.unsqueeze(0).cuda(); // (1, 3, H, W)
        cam->feature_map_ = pc->feature_extractor_->extract(img_tensor).squeeze(0).detach(); // (C, H, W)
    }
}

/**
 * @brief 计算上下文特征 (weighted_orig_feat, weighted_render_feat)
 * 
 * 将点云投影到前 slide_window_size_ 帧，并利用目标视角的 Z 轴面片法向量加权。
 * 同时在这一个循环内执行当前模型状态的原图采样和渲染图采样。
 */
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> compute_context_features(
    const std::shared_ptr<Dataset>& dataset,
    const std::shared_ptr<Camera>& current_cam,
    const torch::Tensor& fused_points,    // (N, 3) World Space
    const torch::Tensor& fused_normals_w, // (N, 3) World Space Normals
    const torch::Tensor& fused_pixels,    // (N, 2) Current Cam Pixels
    int window_size,
    const std::shared_ptr<GaussianModel>& pc,
    const torch::Tensor& bg,
    const torch::Tensor& render_mask)
{
    torch::NoGradGuard no_grad; 
    
    int num_points = fused_points.size(0);
    int Orig_C_dim = 0;
    int Render_C_dim = 0;
    
    if (current_cam->feature_map_.defined()) {
        Orig_C_dim = current_cam->feature_map_.size(0);
        Render_C_dim = Orig_C_dim; 
    } else {
        for (auto it = dataset->train_cameras_.rbegin(); it != dataset->train_cameras_.rend(); ++it) {
            if ((*it)->feature_map_.defined()) {
                Orig_C_dim = (*it)->feature_map_.size(0);
                Render_C_dim = Orig_C_dim;
                break;
            }
        }
    }
    
    if (Orig_C_dim == 0) {
        return std::make_tuple(
            torch::zeros({num_points, 128}, torch::kCUDA), 
            torch::zeros({num_points, 128}, torch::kCUDA), 
            torch::zeros({num_points, 3}, torch::kCUDA), 
            torch::zeros({num_points, window_size, 128}, torch::kCUDA), 
            torch::zeros({num_points, window_size, 128}, torch::kCUDA), 
            torch::zeros({num_points, window_size, 3}, torch::kCUDA), 
            torch::zeros({num_points, window_size, 1}, torch::kCUDA)
        );
    }
    
    // --- Compute current frame features & curr_dir ---
    torch::Tensor f_curr = torch::zeros({num_points, Orig_C_dim}, torch::kFloat32).cuda();
    torch::Tensor render_f_curr = torch::zeros({num_points, Render_C_dim}, torch::kFloat32).cuda();
    
    Eigen::Matrix3f curr_R_cw = current_cam->R_cw_.cast<float>();
    Eigen::Vector3f curr_t_cw = current_cam->t_cw_.cast<float>();
    auto curr_R_tensor = tensor_utils::EigenMatrix2TorchTensor(curr_R_cw, torch::kCUDA);
    auto curr_t_tensor = tensor_utils::EigenMatrix2TorchTensor(curr_t_cw, torch::kCUDA);
    auto curr_pts_cam = torch::matmul(fused_points, curr_R_tensor.t()) + curr_t_tensor.view({1, 3});
    auto curr_dir = torch::nn::functional::normalize(curr_pts_cam, torch::nn::functional::NormalizeFuncOptions().p(2.0).dim(-1));

    if (current_cam->feature_map_.defined()) {
        int W = current_cam->image_width_;
        int H = current_cam->image_height_;
        
        auto x_p = fused_pixels.index({torch::indexing::Slice(), 0}).to(torch::kFloat32);
        auto y_p = fused_pixels.index({torch::indexing::Slice(), 1}).to(torch::kFloat32);
        auto grid_x = 2.0f * x_p / (W - 1) - 1.0f;
        auto grid_y = 2.0f * y_p / (H - 1) - 1.0f;
        auto grid = torch::stack({grid_x, grid_y}, -1).view({1, 1, num_points, 2}).cuda();

        auto f_curr_sample = torch::nn::functional::grid_sample(
             current_cam->feature_map_.unsqueeze(0),
             grid,
             torch::nn::functional::GridSampleFuncOptions().align_corners(true).padding_mode(torch::kZeros)
        ); 
        f_curr = f_curr_sample.view({Orig_C_dim, num_points}).t();

        if (pc && pc->feature_extractor_ && bg.defined()) {
            auto render_pkg = render_2d(current_cam, pc, bg, 1.0f, true, false, render_mask, false);
            auto render_tensor = render_pkg.rendered_image.unsqueeze(0);
            auto render_feat_map_curr = pc->feature_extractor_->extract(render_tensor).detach();
            
            auto render_f_curr_sample = torch::nn::functional::grid_sample(
                 render_feat_map_curr,
                 grid,
                 torch::nn::functional::GridSampleFuncOptions().align_corners(true).padding_mode(torch::kZeros)
            ); 
            render_f_curr = render_f_curr_sample.view({Render_C_dim, num_points}).t();
        }
    }
    
    // --- Allocate Historical Buffers ---
    auto hist_f = torch::zeros({num_points, window_size, Orig_C_dim}, torch::kFloat32).cuda();
    auto hist_render_f = torch::zeros({num_points, window_size, Render_C_dim}, torch::kFloat32).cuda();
    auto hist_dir = torch::zeros({num_points, window_size, 3}, torch::kFloat32).cuda();
    auto hist_mask = torch::zeros({num_points, window_size, 1}, torch::kFloat32).cuda();
    
    int train_num = dataset->train_cameras_.size();
    
    // Find current camera index
    int current_idx = train_num - 1;
    for (int i = 0; i < train_num; ++i) {
        if (dataset->train_cameras_[i] == current_cam) {
            current_idx = i;
            break;
        }
    }
    
    int count_processed = 0;
    
    for (int i = current_idx - 1; i >= 0 && count_processed < window_size; --i) {
        auto cam = dataset->train_cameras_[i];
        if (!cam->feature_map_.defined()) continue;
        
        int w_idx = count_processed;
        count_processed++;
        
        Eigen::Matrix3f R_cw = cam->R_cw_.cast<float>();
        Eigen::Vector3f t_cw = cam->t_cw_.cast<float>();
        auto R_tensor = tensor_utils::EigenMatrix2TorchTensor(R_cw, torch::kCUDA);
        auto t_tensor = tensor_utils::EigenMatrix2TorchTensor(t_cw, torch::kCUDA);
        
        auto pts_cam = torch::matmul(fused_points, R_tensor.t()) + t_tensor.view({1, 3});
        auto z_cam = pts_cam.index({torch::indexing::Slice(), 2}); // (N)
        
        auto cam_center_w_tensor = cam->camera_center_.view({1, 3});
        auto dir_w = fused_points - cam_center_w_tensor; // (N, 3)
        auto dir_curr_cam = torch::matmul(dir_w, curr_R_tensor.t()); // (N, 3)
        auto dir_curr_cam_norm = torch::nn::functional::normalize(dir_curr_cam, torch::nn::functional::NormalizeFuncOptions().p(2.0).dim(-1));
        
        hist_dir.index_put_({torch::indexing::Slice(), w_idx, torch::indexing::Slice()}, dir_curr_cam_norm);

        auto valid_mask = (z_cam > 0.1f);
        // Safely clamp z_cam to avoid division by zero or negative projection resulting in Inf/NaN coords
        auto z_cam_safe = torch::where(valid_mask, z_cam, torch::ones_like(z_cam) * 0.1f);
        
        float fx = cam->fx_;
        float fy = cam->fy_;
        float cx = cam->cx_; 
        float cy = cam->cy_;
        int W = cam->image_width_;
        int H = cam->image_height_;
        
        auto x_proj = (pts_cam.index({torch::indexing::Slice(), 0}) * fx) / z_cam_safe + cx;
        auto y_proj = (pts_cam.index({torch::indexing::Slice(), 1}) * fy) / z_cam_safe + cy;
        
        valid_mask = valid_mask & (x_proj >= 0) & (x_proj < W) & (y_proj >= 0) & (y_proj < H);
        
        hist_mask.index_put_({torch::indexing::Slice(), w_idx, 0}, valid_mask.to(torch::kFloat32));
        
        if (valid_mask.sum().item<int>() == 0) continue;
        
        auto grid_x = 2.0f * x_proj / (W - 1) - 1.0f;
        auto grid_y = 2.0f * y_proj / (H - 1) - 1.0f;
        auto grid = torch::stack({grid_x, grid_y}, -1).view({1, 1, num_points, 2});
        
        auto sampled_orig = torch::nn::functional::grid_sample(
            cam->feature_map_.unsqueeze(0),
            grid,
            torch::nn::functional::GridSampleFuncOptions().align_corners(true).padding_mode(torch::kZeros)
        );
        auto orig_feat = sampled_orig.view({Orig_C_dim, num_points}).t();
        
        auto mask_f = valid_mask.to(torch::kFloat32).unsqueeze(1); // (N, 1)
        hist_f.index_put_({torch::indexing::Slice(), w_idx, torch::indexing::Slice()}, orig_feat * mask_f);
        
        if (pc && pc->feature_extractor_ && bg.defined()) {
            auto render_pkg = render_2d(cam, pc, bg, 1.0f, true, false, render_mask, false);
            auto render_tensor = render_pkg.rendered_image.unsqueeze(0);
            auto render_feat_map = pc->feature_extractor_->extract(render_tensor).detach();
            
            auto sampled_render = torch::nn::functional::grid_sample(
                render_feat_map,
                grid,
                torch::nn::functional::GridSampleFuncOptions().align_corners(true).padding_mode(torch::kZeros)
            );
            auto render_feat = sampled_render.view({Render_C_dim, num_points}).t();
            hist_render_f.index_put_({torch::indexing::Slice(), w_idx, torch::indexing::Slice()}, render_feat * mask_f);
        }
    }
    
    return std::make_tuple(f_curr, render_f_curr, curr_dir, hist_f, hist_render_f, hist_dir, hist_mask);
}

// Helper function to compute KNN mean distance for 2D pixels using SimpleKNN
inline torch::Tensor compute_knn_distance(const torch::Tensor& fused_pixels) {
    int total_num = fused_pixels.size(0);
    if (total_num == 0) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        return torch::empty({0, 1}, options);
    }

    // Pad fused_pixels (N, 2) to (N, 3) with Z = 0
    auto zeros = torch::zeros({total_num, 1}, fused_pixels.options());
    auto pixels_3d = torch::cat({fused_pixels, zeros}, 1).contiguous(); // (N, 3), float32, cuda

    // Allocate memory for meanDists on GPU
    float* d_meanDists;
    cudaMalloc((void**)&d_meanDists, total_num * sizeof(float));

    // Call SimpleKNN
    float3* d_points = (float3*)pixels_3d.data_ptr<float>();
    SimpleKNN::knn(total_num, d_points, d_meanDists);

    // Create a tensor from the result
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor dis_tensor = torch::empty({total_num, 1}, options);
    cudaMemcpy(dis_tensor.data_ptr<float>(), d_meanDists, total_num * sizeof(float), cudaMemcpyDeviceToDevice);
    
    dis_tensor = dis_tensor.clamp(/*min=*/0.0, /*max=*/200.0);
    // Free memory
    cudaFree(d_meanDists);

    return dis_tensor;
}

