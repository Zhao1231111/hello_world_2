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

#include "yaml_utils.h"

#include <chrono>
#include <deque>
#include <queue>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <algorithm>

#include <geometry_msgs/PoseStamped.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <eigen_conversions/eigen_msg.h>
#include <Eigen/Eigen>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

class Params
{
public:
    Params(const YAML::Node &node)
    {
        height = node["height"].as<int>();
        width = node["width"].as<int>();
        fx = node["fx"].as<double>();
        fy = node["fy"].as<double>();
        cx = node["cx"].as<double>();
        cy = node["cy"].as<double>();

        select_every_k_frame = node["select_every_k_frame"].as<int>();

        sh_degree = node["sh_degree"].as<int>();
        white_background = node["white_background"].as<bool>();
        random_background = node["random_background"].as<bool>();
        convert_SHs_python = node["convert_SHs_python"].as<bool>();
        compute_cov3D_python = node["compute_cov3D_python"].as<bool>();
        lambda_erank = node["lambda_erank"].as<double>();
        scaling_scale = node["scaling_scale"].as<int>();

        position_lr = node["position_lr"].as<double>();
        feature_lr = node["feature_lr"].as<double>();
        opacity_lr = node["opacity_lr"].as<double>();
        scaling_lr = node["scaling_lr"].as<double>();
        rotation_lr = node["rotation_lr"].as<double>();
        lambda_dssim = node["lambda_dssim"].as<double>();

        apply_exposure = node["apply_exposure"].as<bool>();
        exposure_lr = node["exposure_lr"].as<double>();
        skybox_points_num = node["skybox_points_num"].as<int>();
        skybox_radius = node["skybox_radius"].as<int>();

        // === 新增 ===
        lambda_dist = node["lambda_dist"].as<double>();
        lambda_normal = node["lambda_normal"].as<double>();
        alpha_threshold = node["alpha_threshold"].as<double>();
        slide_window_size = node["slide_window_size"].as<int>();
        hiloss_threshold = node["hiloss_threshold"].as<double>();
        hiColorLoss_threshold = node["hiColorLoss_threshold"].as<double>();
        train_times_threshold = node["train_times_threshold"].as<int>();
        scale_ratio_threshold = node["scale_ratio_threshold"].as<double>();
        opacity_threshold = node["opacity_threshold"].as<double>();
        enable_normal_cull = node["enable_normal_cull"] ? node["enable_normal_cull"].as<bool>() : true;
        normal_z_cutoff = node["normal_z_cutoff"] ? node["normal_z_cutoff"].as<double>() : -0.1;
        use_Gaussian_regress = node["use_Gaussian_regress"].as<bool>();
        apply_regressed_rotation =
            node["apply_regressed_rotation"] ? node["apply_regressed_rotation"].as<bool>() : true;
        apply_regressed_position =
            node["apply_regressed_position"] ? node["apply_regressed_position"].as<bool>() : true;
        apply_regressed_sh_dc =
            node["apply_regressed_sh_dc"] ? node["apply_regressed_sh_dc"].as<bool>() : true;
        apply_regressed_sh_rest =
            node["apply_regressed_sh_rest"] ? node["apply_regressed_sh_rest"].as<bool>() : true;
        if_prune = node["if_prune"].as<bool>();
        dark_color_threshold = node["dark_color_threshold"] ? node["dark_color_threshold"].as<double>() : 0.0;
        color_gradient_threshold = node["color_gradient_threshold"] ? node["color_gradient_threshold"].as<double>() : 0.0;
        voxel_size = node["voxel_size"] ? node["voxel_size"].as<double>() : 0.1;
        enable_regress_high_grad_traditional_init =
            node["enable_regress_high_grad_traditional_init"] ? node["enable_regress_high_grad_traditional_init"].as<bool>() : false;
        regress_high_grad_keep_ratio =
            node["regress_high_grad_keep_ratio"] ? node["regress_high_grad_keep_ratio"].as<double>() : 0.2;
        clamp_scale = node["clamp_scale"] ? node["clamp_scale"].as<double>() : 1.7;
        far_depth_threshold = node["far_depth_threshold"] ? node["far_depth_threshold"].as<double>() : 10.0;

        // SPNet深度补全与旋转初始化参数（可选；缺失则使用默认值）
        // 这些参数由 gaussian.cpp 的 initialize()/extend() 使用
        spnet_epsilon_1 = node["spnet_epsilon_1"] ? node["spnet_epsilon_1"].as<float>() : 0.3f;
        spnet_epsilon_2 = node["spnet_epsilon_2"] ? node["spnet_epsilon_2"].as<float>() : 80.0f;
        spnet_patch_size = node["spnet_patch_size"] ? node["spnet_patch_size"].as<int>() : 30;
        spnet_dilate_radius = node["spnet_dilate_radius"] ? node["spnet_dilate_radius"].as<int>() : 0;
        spnet_depth_grad_threshold = node["spnet_depth_grad_threshold"] ? node["spnet_depth_grad_threshold"].as<float>() : 0.5f;
        spnet_use_depth_normal_rotation_init = node["spnet_use_depth_normal_rotation_init"] ? node["spnet_use_depth_normal_rotation_init"].as<bool>() : true;
        spnet_enforce_normal_face_camera = node["spnet_enforce_normal_face_camera"] ? node["spnet_enforce_normal_face_camera"].as<bool>() : true;
        spnet_fallback_view_ray_when_invalid = node["spnet_fallback_view_ray_when_invalid"] ? node["spnet_fallback_view_ray_when_invalid"].as<bool>() : true;

        // DA3 稠密深度与法线初始化参数
        da3_min_depth = node["da3_min_depth"] ? node["da3_min_depth"].as<float>() : 0.1f;
        da3_max_depth = node["da3_max_depth"] ? node["da3_max_depth"].as<float>() : 100.0f;
        da3_conf_threshold = node["da3_conf_threshold"] ? node["da3_conf_threshold"].as<float>() : 0.0f;
        da3_align_min_points = node["da3_align_min_points"] ? node["da3_align_min_points"].as<int>() : 30;
        da3_save_depth_vis = node["da3_save_depth_vis"] ? node["da3_save_depth_vis"].as<bool>() : true;

        backbone_path = node["backbone_path"].as<std::string>();
        mlp_path = node["mlp_path"].as<std::string>();
        opacity_modifier = node["opacity_modifier"].as<double>();
        scale_modifier = node["scale_modifier"].as<double>();
        opacity_modifier_up = node["opacity_modifier_up"].as<double>();
        extend_debug = node["extend_debug"].as<bool>();
        if_tileCull_in_extend =
            node["if_tileCull_in_extend"] ? node["if_tileCull_in_extend"].as<bool>() : false;
        if_full_regress =
            node["if_full_regress"] ? node["if_full_regress"].as<bool>() : false;

        // Backward Pose Optimization (IEKF) — 全部可选，默认不启用
        enable_pose_refinement = node["enable_pose_refinement"] ? node["enable_pose_refinement"].as<bool>() : false;
        pose_max_iterations = node["pose_max_iterations"] ? node["pose_max_iterations"].as<int>() : 5;
        pose_img_point_cov = node["pose_img_point_cov"] ? node["pose_img_point_cov"].as<double>() : 100.0;
        pose_patch_size = node["pose_patch_size"] ? node["pose_patch_size"].as<int>() : 3;
        pose_init_cov = node["pose_init_cov"] ? node["pose_init_cov"].as<double>() : 0.01;
        pose_conv_thresh_rot = node["pose_conv_thresh_rot"] ? node["pose_conv_thresh_rot"].as<double>() : 0.001;
        pose_conv_thresh_pos = node["pose_conv_thresh_pos"] ? node["pose_conv_thresh_pos"].as<double>() : 0.001;
        pose_refine_start_frame = node["pose_refine_start_frame"] ? node["pose_refine_start_frame"].as<int>() : 2;
        pose_alpha_threshold = node["pose_alpha_threshold"] ? node["pose_alpha_threshold"].as<double>() : 0.2;
        pose_huber_delta = node["pose_huber_delta"] ? node["pose_huber_delta"].as<double>() : 0.05;
        pose_min_valid_pixels = node["pose_min_valid_pixels"] ? node["pose_min_valid_pixels"].as<int>() : 256;
        pose_rerender_each_iter = node["pose_rerender_each_iter"] ? node["pose_rerender_each_iter"].as<bool>() : true;
        pose_max_step_rot_deg = node["pose_max_step_rot_deg"] ? node["pose_max_step_rot_deg"].as<double>() : 0.3;
        pose_max_step_trans_cm = node["pose_max_step_trans_cm"] ? node["pose_max_step_trans_cm"].as<double>() : 3.0;
        pose_rmse_increase_tolerance_ratio =
            node["pose_rmse_increase_tolerance_ratio"] ? node["pose_rmse_increase_tolerance_ratio"].as<double>() : 0.01;
        pose_min_alpha_coverage_ratio =
            node["pose_min_alpha_coverage_ratio"] ? node["pose_min_alpha_coverage_ratio"].as<double>() : 0.005;
        pose_alpha_erode_radius = node["pose_alpha_erode_radius"] ? node["pose_alpha_erode_radius"].as<int>() : 1;

        // === 致密化参数（可选，缺失则使用默认值）===
        densify_grad_threshold    = node["densify_grad_threshold"]    ? node["densify_grad_threshold"].as<double>()  : 0.0002;
        percent_dense             = node["percent_dense"]             ? node["percent_dense"].as<double>()            : 0.01;
        densify_from_train_times  = node["densify_from_train_times"]  ? node["densify_from_train_times"].as<int>()    : 5;
        densification_interval    = node["densification_interval"]    ? node["densification_interval"].as<int>()      : 3;
        densify_index_gap         = node["densify_index_gap"]         ? node["densify_index_gap"].as<int>()           : slide_window_size;
        densify_max_per_round     = node["densify_max_per_round"]     ? node["densify_max_per_round"].as<int>()       : 1;
        densify_covis_window      = node["densify_covis_window"]      ? node["densify_covis_window"].as<int>()        : slide_window_size;
        densify_min_train_after_covis = node["densify_min_train_after_covis"] ? node["densify_min_train_after_covis"].as<int>() : densification_interval;
        densify_train_gate_alpha  = node["densify_train_gate_alpha"]  ? node["densify_train_gate_alpha"].as<double>() : 0.5;
        post_densify_window_radius= node["post_densify_window_radius"]? node["post_densify_window_radius"].as<int>()  : slide_window_size;
        post_densify_boost_rounds = node["post_densify_boost_rounds"] ? node["post_densify_boost_rounds"].as<int>()   : 2;
        post_densify_boost_budget = node["post_densify_boost_budget"] ? node["post_densify_boost_budget"].as<int>()   : std::min(100 / 3, 2 * slide_window_size + 1);
        opacity_cull_threshold    = node["opacity_cull_threshold"]    ? node["opacity_cull_threshold"].as<double>()   : 0.005;
        scene_extent              = node["scene_extent"]              ? node["scene_extent"].as<double>()              : 50.0;
        densify_alpha             = node["densify_alpha"]             ? node["densify_alpha"].as<double>()             : 1.5;
        densify_new_opacity_scale = node["densify_new_opacity_scale"] ? node["densify_new_opacity_scale"].as<double>() : 0.5;
        densify_new_opacity_min   = node["densify_new_opacity_min"]   ? node["densify_new_opacity_min"].as<double>()   : 0.02;
        densify_newborn_boost_steps = node["densify_newborn_boost_steps"] ? node["densify_newborn_boost_steps"].as<int>() : 20;
        densify_newborn_pos_lr_scale = node["densify_newborn_pos_lr_scale"] ? node["densify_newborn_pos_lr_scale"].as<double>() : 2.0;

        // === 单帧训练过程可视化评估参数 ===
        enable_train_visual_eval =
            node["enable_train_visual_eval"] ? node["enable_train_visual_eval"].as<bool>() : false;
        train_visual_eval_every_k_train_times =
            node["train_visual_eval_every_k_train_times"] ? node["train_visual_eval_every_k_train_times"].as<int>() : 1;
        train_visual_eval_output_dir =
            node["train_visual_eval_output_dir"] ? node["train_visual_eval_output_dir"].as<std::string>() : "";
        if (node["train_visual_eval_frame_ids"]) {
            train_visual_eval_frame_ids = node["train_visual_eval_frame_ids"].as<std::vector<int>>();
        }
    }

    /// dataset
    int height;
    int width;
    double fx;
    double fy;
    double cx;
    double cy;

    int select_every_k_frame;

    /// gaussian
    int sh_degree;
    bool white_background;
    bool random_background;
    bool convert_SHs_python;
    bool compute_cov3D_python;
    float lambda_erank;
    double scaling_scale;

    double position_lr;
    double feature_lr;
    double opacity_lr;
    double scaling_lr;
    double rotation_lr;
    double lambda_dssim;
    double lambda_dist;
    double lambda_normal;
    double alpha_threshold;
    int slide_window_size;
    double hiloss_threshold;
    double hiColorLoss_threshold;
    int train_times_threshold;
    double scale_ratio_threshold;
    double opacity_threshold;
    bool enable_normal_cull;
    double normal_z_cutoff;
    float spnet_epsilon_1;                     // 补全失败检测阈值
    float spnet_epsilon_2;                     // 采样深度上限
    int spnet_patch_size;                      // patch采样大小
    int spnet_dilate_radius;                   // LiDAR mask膨胀半径
    float spnet_depth_grad_threshold;          // 深度梯度过滤阈值
    bool spnet_use_depth_normal_rotation_init; // 是否启用“深度法线驱动旋转初始化”
    bool spnet_enforce_normal_face_camera;     // 法线是否强制朝向相机
    bool spnet_fallback_view_ray_when_invalid; // 像素法线无效时是否回退视线法线

    float da3_min_depth;                       // DA3 深度最小有效值
    float da3_max_depth;                       // DA3 深度最大有效值
    float da3_conf_threshold;                  // DA3 置信度阈值（0 表示不启用）
    int da3_align_min_points;                  // DA3 对齐最小有效点数量
    bool da3_save_depth_vis;                   // 是否保存 DA3 对齐深度可视化

    bool use_Gaussian_regress;
    bool apply_regressed_rotation;
    bool apply_regressed_position;
    bool apply_regressed_sh_dc;
    bool apply_regressed_sh_rest;
    bool if_prune;
    double dark_color_threshold;
    double color_gradient_threshold;
    double voxel_size;
    bool enable_regress_high_grad_traditional_init;
    double regress_high_grad_keep_ratio;
    double clamp_scale;
    double far_depth_threshold;
    
    std::string backbone_path;
    std::string mlp_path;
    
    bool apply_exposure;
    double exposure_lr;
    int skybox_points_num;
    int skybox_radius;
    
    std::string dataset_path_;
    bool generate_dataset_;
    int dataset_target_train_times;
    double opacity_modifier;
    double scale_modifier;
    double opacity_modifier_up;

    bool extend_debug;
    bool if_tileCull_in_extend;
    bool if_full_regress;

    // === Backward Pose Optimization (IEKF) ===
    bool enable_pose_refinement = false;    // 是否启用位姿修正
    int pose_max_iterations = 5;            // IEKF 最大迭代次数
    double pose_img_point_cov = 100.0;      // 像素噪声方差 σ²
    int pose_patch_size = 3;                // 旧版灰度 patch 参数，当前 RGB 线性化实现仅为兼容保留
    double pose_init_cov = 0.01;            // 先验协方差初始值
    double pose_conv_thresh_rot = 0.001;    // 旋转收敛阈值
    double pose_conv_thresh_pos = 0.001;    // 平移收敛阈值
    int pose_refine_start_frame = 2;        // 从第几帧开始启用位姿修正
    double pose_alpha_threshold = 0.2;      // 位姿滤波时的最小 alpha 阈值
    double pose_huber_delta = 0.05;         // 位姿滤波使用的 Huber 阈值
    int pose_min_valid_pixels = 256;        // 允许执行位姿滤波的最小有效像素数
    bool pose_rerender_each_iter = true;    // 每次 IEKF 迭代后是否重新渲染再线性化
    double pose_max_step_rot_deg = 0.3;     // 单轮位姿更新允许的最大旋转步长，单位度
    double pose_max_step_trans_cm = 3.0;    // 单轮位姿更新允许的最大平移步长，单位厘米
    double pose_rmse_increase_tolerance_ratio = 0.01;  // 候选位姿 RMSE 允许相对恶化比例
    double pose_min_alpha_coverage_ratio = 0.005;      // 执行位姿优化要求的最小 alpha 覆盖率
    int pose_alpha_erode_radius = 1;        // 对 alpha 掩码做腐蚀时的半径，去掉边界过渡带

    // === 致密化参数 ===
    double densify_grad_threshold;   // 梯度阈值，控制致密化触发
    double percent_dense;            // scale 判断百分比阈值
    int    densify_from_train_times; // 帧训练多少次后才开始致密化
    int    densification_interval;   // 每隔多少次训练执行一次致密化
    int    densify_index_gap;        // 同一轮被致密化帧的最小索引间隔
    int    densify_max_per_round;    // 每轮最多致密化帧数
    int    densify_covis_window;     // 共视邻域半径（索引近似）
    int    densify_min_train_after_covis; // 邻域上次致密化后最小训练增量
    double densify_train_gate_alpha; // 训练充分性软加权系数
    int    post_densify_window_radius; // 致密化后窗口半径
    int    post_densify_boost_rounds;  // 致密化后窗口优先保留轮数
    int    post_densify_boost_budget;  // 每轮窗口优先预算
    double opacity_cull_threshold;   // 不透明度裁剪阈值
    double scene_extent;             // 场景范围（用于 scale 判断）
    double densify_alpha;            // 致密化位移步长倍数 (α = densify_alpha * max_scale)
    double densify_new_opacity_scale; // 致密化新增点透明度缩放系数
    double densify_new_opacity_min;   // 致密化新增点最小透明度（避免立刻被裁）
    int    densify_newborn_boost_steps; // 新增点位置梯度放大的持续可见训练次数
    double densify_newborn_pos_lr_scale; // 新生期位置梯度放大倍数（等效局部更高 position_lr）

    // === 单帧训练过程可视化评估参数 ===
    bool enable_train_visual_eval = false;                 // 是否启用单帧训练过程评估
    int train_visual_eval_every_k_train_times = 1;         // 每训练多少次保存一次
    std::vector<int> train_visual_eval_frame_ids;          // 需要跟踪的训练帧原始 frame id
    std::string train_visual_eval_output_dir;              // 评估结果输出目录
};

struct Frame 
{
    sensor_msgs::PointCloud2ConstPtr point_msg;
    geometry_msgs::PoseStampedConstPtr pose_msg;
    sensor_msgs::ImageConstPtr image_msg;
};
