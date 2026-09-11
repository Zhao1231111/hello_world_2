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

#include <memory>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <chrono>
#include <utility>

#include <torch/torch.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "mapping.h"
#include "camera.h"
#include "eigen_utils.h"
#include "general_utils.h"
#include "optim_utils.h"
#include "tinyply.h"

#include "simple-knn/spatial.h"
#include "simple-knn/simple_knn.h"
#include "diff_surfel_rasterization_2d/renderer_2d.h"
#include "model/inference_wrapper.h"

// 前向声明
class SPNetWrapper;
class DA3Wrapper;

const double C0 = 0.28209479177387814;
inline double RGB2SH(double color) {return (color - 0.5) / C0;}
inline torch::Tensor RGB2SH(torch::Tensor& rgb) {return (rgb - 0.5f) / C0;}

class Dataset
{
public:
    Dataset(const Params& prm)
      : width_(prm.width), height_(prm.height),
        fx_(prm.fx), fy_(prm.fy), cx_(prm.cx), cy_(prm.cy),
        select_every_k_frame_(prm.select_every_k_frame),
        blind_patch_size_(prm.spnet_patch_size),
        pose_init_cov_(prm.pose_init_cov), has_logged_image_size_(false),
        all_frame_num_(0), is_keyframe_current_(false) {}
        
    void addFrame(Frame& cur_frame);

public:
    int width_;
    int height_;
    double fx_;
    double fy_;
    double cx_;
    double cy_;

    //选择每k帧作为关键帧
    int select_every_k_frame_;
    int blind_patch_size_;
    double pose_init_cov_;
    bool has_logged_image_size_;

    //所有帧数
    int all_frame_num_;
    bool is_keyframe_current_;

    //世界坐标系下的旋转矩阵
    Eigen::aligned_vector<Eigen::Matrix3d> R_wc_;
    Eigen::aligned_vector<Eigen::Vector3d> t_wc_;

    //点云
    Eigen::aligned_vector<Eigen::Vector3d> pointcloud_;
    Eigen::aligned_vector<Eigen::Vector3d> pointcolor_;
    std::vector<float> pointdepth_;

    //SPNet 深度补全的补充点
    Eigen::aligned_vector<Eigen::Vector3d> supplement_pointcloud_;
    Eigen::aligned_vector<Eigen::Vector3d> supplement_pointcolor_;
    std::vector<float> supplement_pointdepth_;
    Eigen::aligned_vector<Eigen::Vector4d> supplement_rotations_;

    //相机对象列表
    std::vector<std::shared_ptr<Camera>> train_cameras_;
    std::vector<std::shared_ptr<Camera>> test_cameras_;
};


//定义宏，将Tensor转换为vector
#define GAUSSIAN_MODEL_TENSORS_TO_VEC                        \
    this->Tensor_vec_xyz_ = {this->xyz_};                    \
    this->Tensor_vec_feature_dc_ = {this->features_dc_};     \
    this->Tensor_vec_feature_rest_ = {this->features_rest_}; \
    this->Tensor_vec_opacity_ = {this->opacity_};            \
    this->Tensor_vec_scaling_ = {this->scaling_};            \
    this->Tensor_vec_rotation_ = {this->rotation_};          \
    this->Tensor_vec_exposure_ = {this->exposure_};

//定义宏，初始化Tensor
#define GAUSSIAN_MODEL_INIT_TENSORS(device_type)                                             \
    this->xyz_ = torch::empty(0, torch::TensorOptions().device(device_type));                \
    this->features_dc_ = torch::empty(0, torch::TensorOptions().device(device_type));        \
    this->features_rest_ = torch::empty(0, torch::TensorOptions().device(device_type));      \
    this->scaling_ = torch::empty(0, torch::TensorOptions().device(device_type));            \
    this->rotation_ = torch::empty(0, torch::TensorOptions().device(device_type));           \
    this->opacity_ = torch::empty(0, torch::TensorOptions().device(device_type));            \
    this->exposure_ = torch::empty(0, torch::TensorOptions().device(device_type));           \
    GAUSSIAN_MODEL_TENSORS_TO_VEC

class GaussianModel
{
public:
    GaussianModel(const Params& prm);

    torch::Tensor getScaling();
    torch::Tensor getRotation();
    torch::Tensor getXYZ();
    torch::Tensor getFeaturesDc();
    torch::Tensor getFeaturesRest();
    torch::Tensor getFeatures();  // 合并 DC 和 REST 为完整 SH 张量 (用于 2DGS)
    torch::Tensor getOpacity();
    // torch::Tensor getCovariance(int scaling_modifier); // 注释掉 3D 协方差计算

    torch::Tensor getExposure();

    void initialize(
        const std::shared_ptr<Dataset>& dataset,
        std::shared_ptr<SPNetWrapper> spnet = nullptr,
        std::shared_ptr<DA3Wrapper> da3 = nullptr);
    void setDA3DebugDir(const std::string& dir) { da3_debug_dir_ = dir; }
    void saveMap(const std::string& result_path);
    void saveCameraRegressData(const std::shared_ptr<Camera>& cam, const std::string& result_path, int frame_idx);
    void saveDataset(const std::string& path, const std::shared_ptr<Dataset>& dataset, int iter = -1);

    void trainingSetup();

    void densificationPostfix(
        torch::Tensor& new_xyz,
        torch::Tensor& new_features_dc,
        torch::Tensor& new_features_rest,
        torch::Tensor& new_opacities,
        torch::Tensor& new_scaling,
        torch::Tensor& new_rotation,
        torch::Tensor& new_ids);

    void prune(torch::Tensor& keep_mask);

public:
    int sh_degree_;//SH阶数
    bool white_background_;
    bool random_background_;
    bool convert_SHs_python_;
    bool compute_cov3D_python_;
    double lambda_erank_;//ERank损失函数参数
    double scaling_scale_;

    //学习率
    double position_lr_;
    double feature_lr_;
    double opacity_lr_;
    double scaling_lr_;
    double rotation_lr_;
    double lambda_dssim_;//DSSIM损失函数参数

    // === 新增 ===
    double lambda_dist_;
    double lambda_normal_;
    double alpha_threshold_;
    double scale_ratio_threshold_;
    double opacity_threshold_;
    double scale_threshold_;
    bool enable_normal_cull_;
    double normal_z_cutoff_;
    float spnet_epsilon_1_;                    // 补全失败检测阈值
    float spnet_epsilon_2_;                    // 采样深度上限
    int spnet_patch_size_;                     // patch采样大小
    int spnet_dilate_radius_;                  // LiDAR mask膨胀半径
    float spnet_depth_grad_threshold_;         // 深度梯度过滤阈值
    bool enable_spnet_ = true;
    bool enable_ablation_logging_ = false;
    int experiment_seed_ = -1;
    bool spnet_use_depth_normal_rotation_init_;// 是否启用深度法线驱动旋转初始化
    bool spnet_enforce_normal_face_camera_;    // 法线是否强制朝向相机
    bool spnet_fallback_view_ray_when_invalid_;// 无效像素时是否回退视线法线
    float da3_min_depth_;                      // DA3 深度最小有效值
    float da3_max_depth_;                      // DA3 深度最大有效值
    float da3_conf_threshold_;                 // DA3 置信度阈值
    int da3_align_min_points_;                 // DA3 最小二乘对齐最小有效点数
    bool da3_save_depth_vis_;                  // 是否保存 DA3 对齐深度可视化

    bool use_Gaussian_regress_;                 //是否使用高斯回归
    bool apply_regressed_rotation_;            // 是否采用回归得到的高斯 rotation
    bool apply_regressed_position_;            // 是否采用回归得到的高斯位置
    bool apply_regressed_sh_dc_;               // 是否采用回归得到的 SH DC
    bool apply_regressed_sh_rest_;             // 是否采用回归得到的 SH REST
    bool if_prune_;
    double opacity_prune_forRegress_;          // 回归高斯的 opacity 裁剪阈值
    
    int slide_window_size_;
    double hiColorLoss_threshold_;
    int train_times_threshold_;
    
    double clamp_scale_;
    double dark_color_threshold_;
    double color_gradient_threshold_;
    double voxel_size_;
    bool enable_regress_high_grad_traditional_init_;
    double regress_high_grad_keep_ratio_;
    double far_depth_threshold_;

    bool apply_exposure_;//是否应用曝光
    double exposure_lr_;//曝光损失函数参数
    int skybox_points_num_;
    int skybox_radius_;


    torch::Tensor xyz_;//点云
    torch::Tensor features_dc_;//SH系数
    torch::Tensor features_rest_;//剩余系数?
    torch::Tensor scaling_;//缩放因子
    torch::Tensor rotation_;//旋转矩阵
    torch::Tensor opacity_;//透明度
    
    torch::Tensor exposure_;
    
    // Global ID Management
    torch::Tensor global_ids_;     // (N, 1) long tensor
    int64_t max_id_ = 0;           // Counter for unique IDs

    std::vector<torch::Tensor> Tensor_vec_xyz_,
                               Tensor_vec_feature_dc_,
                               Tensor_vec_feature_rest_,
                               Tensor_vec_opacity_,
                               Tensor_vec_scaling_ ,
                               Tensor_vec_rotation_,
                               Tensor_vec_exposure_;

    std::vector<int> keyframe_train_times_;

    std::shared_ptr<torch::optim::Adam> optimizer_;
    std::shared_ptr<SparseGaussianAdam> sparse_optimizer_;

    std::shared_ptr<torch::optim::Adam> exposure_optimizer_;

    std::shared_ptr<FeatureExtractor> feature_extractor_;
    std::shared_ptr<GaussianRegressor> gaussian_regressor_;

    bool is_init_;

    torch::Tensor bg_;

    std::chrono::steady_clock::time_point t_start_;
    std::chrono::steady_clock::time_point t_end_;
    double t_forward_;
    double t_backward_;
    double t_step_;
    double t_optlist_;
    double t_tocuda_;
    double spnet_time_ = 0.0;
    int64_t spnet_calls_ = 0;
    std::string da3_debug_dir_;

    std::string dataset_path_;
    std::string result_path_;
    std::string lpips_path_;
    bool generate_dataset_; 
    double opacity_modifier_;
    double scale_modifier_;
    double opacity_modifier_up_;
    bool extend_debug_;
    bool if_full_regress_ = false;

    // === Backward Pose Optimization ===
    bool enable_pose_refinement_ = false;
    int pose_max_iterations_ = 5;
    double pose_img_point_cov_ = 100.0;
    int pose_patch_size_ = 3;          // 旧版灰度 patch 参数，当前 RGB 线性化实现仅为兼容保留
    double pose_init_cov_ = 0.01;
    double pose_conv_thresh_rot_ = 0.001;
    double pose_conv_thresh_pos_ = 0.001;
    int pose_refine_start_frame_ = 2;
    double pose_alpha_threshold_ = 0.2;
    double pose_huber_delta_ = 0.05;
    int pose_min_valid_pixels_ = 256;
    bool pose_rerender_each_iter_ = true;
    double pose_max_step_rot_deg_ = 0.3;
    double pose_max_step_trans_cm_ = 3.0;
    double pose_rmse_increase_tolerance_ratio_ = 0.01;
    double pose_min_alpha_coverage_ratio_ = 0.005;
    int pose_alpha_erode_radius_ = 1;

    // optimize 调用轮次仍用于可复现实验的随机种子和训练过程日志，与已删除的致密化无关。
    int optimize_round_ = 0;

    // === 单帧训练过程可视化评估 ===
    bool enable_train_visual_eval_ = false;                  // 是否启用单帧训练过程评估
    int train_visual_eval_every_k_train_times_ = 1;          // 每训练多少次保存一次
    std::vector<int> train_visual_eval_frame_ids_;           // 需要跟踪的训练帧原始 frame id
    std::string train_visual_eval_output_dir_;               // 单帧训练过程评估输出目录
    std::vector<int> train_visual_eval_last_saved_times_;    // 记录每个训练帧上次保存时的训练次数，避免重复保存
    std::vector<int> train_visual_eval_warned_test_frame_ids_; // 已经提示过“这是 test 帧”的 frame id
};

void extend(
    const std::shared_ptr<Dataset>& dataset,
    std::shared_ptr<GaussianModel>& pc,
    std::shared_ptr<SPNetWrapper> spnet,
    std::shared_ptr<DA3Wrapper> da3 = nullptr);
double optimize(const std::shared_ptr<Dataset>& dataset, std::shared_ptr<GaussianModel>& pc);
void runTrainVisualEvalIfNeeded(const std::shared_ptr<Dataset>& dataset,
                                std::shared_ptr<GaussianModel>& pc,
                                const std::shared_ptr<Camera>& train_camera,
                                int train_camera_idx);
void evaluateVisualQuality(const std::shared_ptr<Dataset>& dataset, 
                           std::shared_ptr<GaussianModel>& pc,
                           const std::string& result_path,
                           const std::string& lpips_path,
                           bool save_image);
