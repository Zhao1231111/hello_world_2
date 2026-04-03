/*
 * Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion
 * Copyright (C) 2025 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <torch/torch.h>
#include <Eigen/Eigen>

#include "tensor_utils.h"

class Camera
{
public:
    Camera(){}

    void setIntrinsic(double w, double h, 
                      double fx, double fy,
                      double cx, double cy)
    {
        image_width_ = w;
        image_height_ = h;
        fx_ = fx;
        fy_ = fy;
        cx_ = cx;
        cy_ = cy;
        FoVx_ = 2.0 * std::atan(w / (2.0 * fx));
        FoVy_ = 2.0 * std::atan(h / (2.0 * fy));
    }

    /**
     * @brief 通过世界到相机的逆变换设置相机位姿
     *
     * 这里对外仍保持 GaussianLIC 既有的接口形式：输入世界系下的相机位姿
     * \f$\mathbf{T}_{wc}\f$，内部统一转换并保存为优化阶段使用的
     * \f$\mathbf{T}_{cw}\f$。
     */
    void setPose(const Eigen::Matrix3d& R_wc, 
                 const Eigen::Vector3d& t_wc)
    {
        setPoseFromCw(R_wc.transpose(), - R_wc.transpose() * t_wc);
    }

    /**
     * @brief 直接以 \f$\mathbf{T}_{cw}\f$ 形式设置相机位姿
     *
     * 该接口专门给位姿滤波使用，避免先转成 \f$\mathbf{T}_{wc}\f$ 再转回
     * \f$\mathbf{T}_{cw}\f$ 带来的重复计算和符号混乱。
     */
    void setPoseFromCw(const Eigen::Matrix3d& R_cw,
                       const Eigen::Vector3d& t_cw)
    {
        R_cw_ = R_cw;
        t_cw_ = t_cw;
        refreshPoseTensors();
    }

    /**
     * @brief 依据当前 Eigen 位姿重建渲染所需的所有张量
     *
     * 记固定投影矩阵为 \f$\mathbf{P}_{fixed}\f$，视图矩阵为 \f$\mathbf{V}\f$，
     * 则 2DGS 渲染阶段实际使用的完整投影矩阵为：
     * \f[
     * \mathbf{F} = \mathbf{V}\mathbf{P}_{fixed}
     * \f]
     * 为了与现有张量排布保持一致，这里仍按代码中的转置存储约定生成。
     */
    void refreshPoseTensors()
    {
        setWorldViewTransform();
        setProjectionMatrix();
        full_proj_transform_ = (world_view_transform_.unsqueeze(0).bmm(projection_matrix_.unsqueeze(0))).squeeze(0);

        limx_neg_ = - 0.15 * image_width_ / fx_ - cx_ / fx_;
        limx_pos_ = 1.15 * image_width_ / fx_ - cx_ / fx_;
        limy_neg_ = - 0.15 * image_height_ / fy_ - cy_ / fy_;
        limy_pos_ = 1.15 * image_height_ / fy_ - cy_ / fy_;
    }

    void setWorldViewTransform()
    {
        Eigen::Matrix4f Rt;
        Rt.setZero();
        Eigen::Matrix3f R = R_cw_.cast<float>();
        Rt.topLeftCorner<3, 3>() = R;
        Eigen::Vector3f t = t_cw_.cast<float>();
        Rt.topRightCorner<3, 1>() = t;
        Rt(3, 3) = 1.0f;

        Eigen::Matrix4f C2W = Rt.inverse();
        Eigen::Vector3f cam_center = C2W.block<3, 1>(0, 3);
        cam_center += trans_;
        cam_center *= scale_;
        C2W.block<3, 1>(0, 3) = cam_center;
        Rt = C2W.inverse();  // Tcw

        world_view_transform_ = tensor_utils::EigenMatrix2TorchTensor(Rt, torch::kCUDA).transpose(0, 1);
        camera_center_ = torch::tensor(
            {cam_center.x(), cam_center.y(), cam_center.z()},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    }

    void setProjectionMatrix()
    {
        torch::Tensor P = torch::zeros({4, 4}, torch::kCUDA);

        float W = image_width_;
        float H = image_height_;
        float cx = cx_;
        float cy = cy_;
        float fx = fx_;
        float fy = fy_;
        float znear = znear_;
        float zfar = zfar_;
        P.index({0, 0}) = 1.0 / std::tan(FoVx_ / 2);
        P.index({1, 1}) = 1.0 / std::tan(FoVy_ / 2);
        P.index({0, 2}) = (2 * cx - W) / W;
        P.index({1, 2}) = (2 * cy - H) / H;
        P.index({3, 2}) = 1.0f;
        P.index({2, 2}) = zfar / (zfar - znear);
        P.index({2, 3}) = -(zfar * znear) / (zfar - znear);

        projection_matrix_ = P.transpose(0, 1);
    }

public:
    std::string image_name_;
    double image_stamp_sec_ = -1.0;  // 当前图像 ROS 时间戳，单位秒，用于后续位姿评估日志对齐

    int image_width_;              
    int image_height_;
    torch::Tensor original_image_;
          
    float fx_;
    float fy_;
    float cx_; 
    float cy_;   
    float FoVx_; 
    float FoVy_;

    float zfar_ = 100.0f;
    float znear_ = 0.01f;

    Eigen::Vector3f trans_ = Eigen::Vector3f::Zero();
    float scale_ = 1.0;

    //相机坐标系下世界坐标系的旋转矩阵
    Eigen::Matrix3d R_cw_;
    Eigen::Vector3d t_cw_;

    torch::Tensor world_view_transform_; 
    torch::Tensor projection_matrix_;    
    torch::Tensor full_proj_transform_;   
    torch::Tensor camera_center_;      

    float limx_neg_;
    float limx_pos_;
    float limy_neg_;
    float limy_pos_;

    // For Dataset Generation - Store Regression MLPs Input
    torch::Tensor regress_f_curr_;         // (N, C)
    torch::Tensor regress_render_f_curr_;  // (N, C)
    torch::Tensor regress_hist_f_;         // (N, W, C)
    torch::Tensor regress_hist_render_f_;  // (N, W, C)
    torch::Tensor regress_curr_dir_;       // (N, 3)
    torch::Tensor regress_hist_dir_;       // (N, W, 3)
    torch::Tensor regress_hist_mask_;      // (N, W, 1)
    
    torch::Tensor regress_inv_depth_;  // (N, 1)
    torch::Tensor regress_n_cam_;      // (N, 3) 
    torch::Tensor regress_mask_tensor_;// (N, 1)
    torch::Tensor regress_base_pixel_; // (N, 2)
    torch::Tensor base_sh_;            // (N, 3)
    torch::Tensor regress_dis_;        // (N, 1)
    std::vector<int64_t> added_ids_;   // (N)

    // === 当前帧 IEKF 位姿优化状态 ===
    Eigen::Matrix3d R_cw_pred_ = Eigen::Matrix3d::Identity();  // 当前帧进入滤波前的预测位姿
    Eigen::Vector3d t_cw_pred_ = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 6, 6> pose_cov_ = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;  // 位姿协方差
    bool pose_refined_ = false;  // 当前帧是否已经完成过一次在线位姿滤波

    // Offline dataset generation cache
    torch::Tensor raw_points_;         // (N, 3)
    torch::Tensor raw_depths_;         // (N)
    torch::Tensor raw_normals_;        // (N, 3)
    torch::Tensor raw_pixels_;         // (N, 2)
    torch::Tensor raw_valid_mask_;     // (N, 1)
    torch::Tensor raw_colors_;         // (N, 3)
    float raw_focal_ = 0.0f;

    // Feature Map for Regression (C, H, W)
    torch::Tensor feature_map_;
    
    // Feature Map for Regression of Rendered existing Gaussians (C, H, W).
    // This is temporary and will be cleared right after regress_render_f_curr_ is extracted
    // to save VRAM.

};
