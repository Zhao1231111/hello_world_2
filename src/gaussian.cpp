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

#include "gaussian.h"
#include "tensor_utils.h"
#include "loss_utils.h"
#include "feature_utils.h"

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <Eigen/Geometry>
#include <torch/script.h>
#include <memory>
#include "simple-knn/simple_knn.h"

namespace fs = std::filesystem;

namespace {

int parseFrameIdFromImageName(const std::string& image_name)
{
    const size_t underscore_pos = image_name.find_last_of('_');
    const size_t dot_pos = image_name.find_last_of('.');
    if (underscore_pos == std::string::npos || dot_pos == std::string::npos || underscore_pos + 1 >= dot_pos) {
        return -1;
    }

    try {
        return std::stoi(image_name.substr(underscore_pos + 1, dot_pos - underscore_pos - 1));
    } catch (const std::exception&) {
        return -1;
    }
}

std::string getImageStem(const std::string& image_name)
{
    return fs::path(image_name).stem().string();
}

std::string formatTrainTimesStem(int train_times)
{
    std::ostringstream oss;
    oss << "times_" << std::setw(4) << std::setfill('0') << train_times;
    return oss.str();
}

void saveTensorImageAsBgr(const torch::Tensor& image_chw, const std::string& image_path)
{
    auto image_u8 = image_chw.detach().to(torch::kCPU).permute({1, 2, 0}).contiguous();
    image_u8 = image_u8.mul(255).clamp(0, 255).to(torch::kU8);
    cv::Mat image_mat(
        image_u8.size(0),
        image_u8.size(1),
        CV_8UC3,
        image_u8.data_ptr<uint8_t>());
    cv::Mat bgr_mat;
    cv::cvtColor(image_mat, bgr_mat, cv::COLOR_RGB2BGR);
    cv::imwrite(image_path, bgr_mat);
}

void logDatasetTensorHealth(const std::string& name, const torch::Tensor& tensor)
{
    if (!tensor.defined()) {
        std::cout << "[Dataset][Check] " << name << " 未定义。" << std::endl;
        return;
    }

    auto flat = tensor.reshape({-1});
    const int64_t total = flat.numel();
    const int64_t invalid = (~torch::isfinite(flat)).sum().item<int64_t>();

    std::cout << "[Dataset][Check] " << name
              << " total=" << total
              << ", invalid=" << invalid;

    if (total > 0 && invalid < total) {
        auto valid_flat = flat.index({torch::isfinite(flat)});
        std::cout << ", min=" << valid_flat.min().item<float>()
                  << ", max=" << valid_flat.max().item<float>();
    }
    std::cout << std::endl;
}

std::shared_ptr<torch::jit::script::Module> loadLpipsModuleCached(const std::string& lpips_path)
{
    static std::mutex cache_mutex;
    static std::string cached_path;
    static std::shared_ptr<torch::jit::script::Module> cached_module;

    std::lock_guard<std::mutex> lock(cache_mutex);
    if (lpips_path.empty()) {
        return nullptr;
    }
    if (cached_path == lpips_path) {
        return cached_module;
    }

    try
    {
        auto module = std::make_shared<torch::jit::script::Module>(torch::jit::load(lpips_path + "/lpips_alex.pt"));
        module->to(torch::kCUDA);
        module->eval();
        cached_path = lpips_path;
        cached_module = module;
        return cached_module;
    }
    catch (const c10::Error& e)
    {
        cached_path = lpips_path;
        cached_module.reset();
        std::cerr << "lpips model loading failed: " << e.what() << std::endl;
        return nullptr;
    }
}

bool isRequestedTrainVisualEvalFrame(const std::vector<int>& frame_ids, int frame_id)
{
    for (int id : frame_ids) {
        if (id == frame_id) {
            return true;
        }
    }
    return false;
}

bool hasWarnedTestFrame(const std::vector<int>& warned_frame_ids, int frame_id)
{
    for (int warned_id : warned_frame_ids) {
        if (warned_id == frame_id) {
            return true;
        }
    }
    return false;
}

void warnRequestedTestFrames(const std::shared_ptr<Dataset>& dataset,
                             std::shared_ptr<GaussianModel>& pc)
{


    for (const auto& test_camera : dataset->test_cameras_) {
        const int frame_id = parseFrameIdFromImageName(test_camera->image_name_);
        if (!isRequestedTrainVisualEvalFrame(pc->train_visual_eval_frame_ids_, frame_id)) {
            continue;
        }
        if (hasWarnedTestFrame(pc->train_visual_eval_warned_test_frame_ids_, frame_id)) {
            continue;
        }

        std::cout << "[TrainVisualEval] 警告: frame id " << frame_id
                  << " 对应的是 test 帧 " << test_camera->image_name_
                  << "，不是训练帧，已跳过该 id。" << std::endl;
        pc->train_visual_eval_warned_test_frame_ids_.push_back(frame_id);
    }
}

fs::path resolveTrainVisualEvalOutputDir(const std::string& result_path, const std::string& configured_dir)
{
    if (configured_dir.empty()) {
        return fs::path(result_path) / "train_visual_eval";
    }

    fs::path output_path(configured_dir);
    if (output_path.is_absolute()) {
        return output_path;
    }
    return fs::path(result_path) / output_path;
}

void appendTrainVisualEvalHistory(const fs::path& train_dir,
                                  int train_times,
                                  int optimize_round,
                                  double psnr,
                                  double ssim,
                                  double lpips)
{

    const fs::path csv_path = train_dir / "metrics_history.csv";
    const bool write_header = !fs::exists(csv_path);

    std::ofstream ofs(csv_path, std::ios::app);
    if (!ofs.is_open()) return;

    if (write_header) {
        ofs << "train_times,optimize_round,psnr,ssim,lpips\n";
    }

    ofs << train_times << ","
        << optimize_round << ","
        << psnr << ","
        << ssim << ","
        << lpips << "\n";
}

struct FrameMetricRecord
{
    std::string image_name;
    int frame_id = -1;
    int subset_index = -1;
    double psnr = std::numeric_limits<double>::quiet_NaN();
    double ssim = std::numeric_limits<double>::quiet_NaN();
    double lpips = std::numeric_limits<double>::quiet_NaN();
};

struct VisualQualityEvalResult
{
    std::vector<FrameMetricRecord> records;
    double mean_psnr = std::numeric_limits<double>::quiet_NaN();
    double mean_ssim = std::numeric_limits<double>::quiet_NaN();
    double mean_lpips = std::numeric_limits<double>::quiet_NaN();
};

double computeMeanOrNaN(double sum, size_t count)
{
    if (count == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return sum / static_cast<double>(count);
}

void savePerFrameMetricsCsv(const fs::path& csv_path,
                            const std::vector<FrameMetricRecord>& records)
{
    fs::create_directories(csv_path.parent_path());

    std::ofstream ofs(csv_path);
    if (!ofs.is_open()) {
        std::cerr << "[Eval] 无法写入逐帧指标文件: " << csv_path << std::endl;
        return;
    }

    ofs << "image_name,frame_id,subset_index,psnr,ssim,lpips\n";
    ofs << std::fixed << std::setprecision(8);
    for (const auto& record : records) {
        ofs << record.image_name << ","
            << record.frame_id << ","
            << record.subset_index << ","
            << record.psnr << ","
            << record.ssim << ","
            << record.lpips << "\n";
    }
}

VisualQualityEvalResult evaluateCameraSplit(
    const std::vector<std::shared_ptr<Camera>>& cameras,
    const std::string& split_name,
    std::shared_ptr<GaussianModel>& pc,
    const torch::Tensor& bg,
    const std::shared_ptr<torch::jit::script::Module>& lpips_module,
    bool save_image,
    const std::string& render_dir_path,
    const std::string& gt_dir_path)
{
    VisualQualityEvalResult result;
    double psnr_sum = 0.0;
    double ssim_sum = 0.0;
    double lpips_sum = 0.0;
    size_t valid_lpips_count = 0;

    for (size_t camera_idx = 0; camera_idx < cameras.size(); ++camera_idx) {
        const auto& camera = cameras[camera_idx];

        auto render_pkg = render_2d(camera, pc, bg, 1.0f, false);
        auto rendered_image = render_pkg.rendered_image.clamp(0, 1);
        auto gt_image = camera->original_image_.to(torch::kCUDA).clamp(0, 1);

        const double psnr = loss_utils::psnr(rendered_image, gt_image).mean().item<double>();
        const double ssim = loss_utils::ssim(rendered_image, gt_image).item<double>();

        double lpips = std::numeric_limits<double>::quiet_NaN();
        if (lpips_module != nullptr) {
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(rendered_image.unsqueeze(0));
            inputs.push_back(gt_image.unsqueeze(0));
            lpips = lpips_module->forward(inputs).toTensor().item<double>();
            lpips_sum += lpips;
            ++valid_lpips_count;
        }

        int frame_id = parseFrameIdFromImageName(camera->image_name_);
        if (frame_id < 0) {
            frame_id = static_cast<int>(camera_idx);
        }

        result.records.push_back(
            FrameMetricRecord{
                camera->image_name_,
                frame_id,
                static_cast<int>(camera_idx),
                psnr,
                ssim,
                lpips,
            });

        psnr_sum += psnr;
        ssim_sum += ssim;

        if (save_image) {

            torch::cuda::synchronize();
            saveTensorImageAsBgr(rendered_image, render_dir_path + "/" + camera->image_name_);
            saveTensorImageAsBgr(gt_image, gt_dir_path + "/" + camera->image_name_);
        }
    }

    result.mean_psnr = computeMeanOrNaN(psnr_sum, result.records.size());
    result.mean_ssim = computeMeanOrNaN(ssim_sum, result.records.size());
    result.mean_lpips = computeMeanOrNaN(lpips_sum, valid_lpips_count);
    return result;
}

}

void Dataset::addFrame(Frame& cur_frame)
{

    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(cur_frame.image_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat image_bgr = cv_ptr->image;
    if (!has_logged_image_size_)
    {
        has_logged_image_size_ = true;
        std::cout << "[Dataset] first image msg size = "
                  << image_bgr.cols << "x" << image_bgr.rows
                  << ", config size = " << width_ << "x" << height_
                  << std::endl;
    }
    if (width_ > 0 && height_ > 0 &&
        (image_bgr.cols != width_ || image_bgr.rows != height_))
    {
        std::cout << "\033[1;33m[Dataset] resizing incoming image from "
                  << image_bgr.cols << "x" << image_bgr.rows
                  << " to config size " << width_ << "x" << height_
                  << "\033[0m" << std::endl;
        cv::resize(image_bgr, image_bgr, cv::Size(width_, height_), 0, 0, cv::INTER_LINEAR);
    }
    cv::Mat image_rgb;
    cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);
    image_rgb.convertTo(image_rgb, CV_32FC3, 1.0f / 255.0f);


    Eigen::Quaterniond q_wc;
    Eigen::Vector3d t_wc;
    tf::quaternionMsgToEigen(cur_frame.pose_msg->pose.orientation, q_wc);
    tf::pointMsgToEigen(cur_frame.pose_msg->pose.position, t_wc);
    R_wc_.push_back(q_wc.toRotationMatrix());
    t_wc_.push_back(t_wc);


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*cur_frame.point_msg, *cloud);
    for (const auto& pt : cloud->points)
    {
        pointcloud_.emplace_back(Eigen::Vector3d(pt.x, pt.y, pt.z));
        pointcolor_.emplace_back(Eigen::Vector3d(pt.r, pt.g, pt.b) / 255.0);
        Eigen::Matrix3d R_cw = q_wc.toRotationMatrix().transpose();
        Eigen::Vector3d t_cw = - R_cw * t_wc;
        Eigen::Vector3d pt_c = R_cw * pointcloud_.back() + t_cw;
        assert(pt_c(2) > 0);
        pointdepth_.push_back(static_cast<float>(pt_c(2)));
    }


    int width = image_rgb.cols, height = image_rgb.rows;
    if ((all_frame_num_ + 1) % select_every_k_frame_ == 0)
    {
        is_keyframe_current_ = true;
        std::shared_ptr<Camera> cam = std::make_shared<Camera>();

        cam->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(image_rgb, torch::kCPU, true);


        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << all_frame_num_;
        std::string formatted_str = ss.str();
        cam->image_name_ = "train_" + formatted_str + ".jpg";
        cam->image_stamp_sec_ = cur_frame.image_msg->header.stamp.toSec();

        cam->setIntrinsic(width, height, fx_, fy_, cx_, cy_);
        cam->setPose(q_wc.toRotationMatrix(), t_wc);


        cam->R_cw_pred_ = cam->R_cw_;
        cam->t_cw_pred_ = cam->t_cw_;
        cam->pose_cov_ = Eigen::Matrix<double, 6, 6>::Identity() * pose_init_cov_;
        cam->pose_refined_ = false;

        train_cameras_.emplace_back(cam);
    }
    else
    {
        is_keyframe_current_ = false;
        std::shared_ptr<Camera> cam = std::make_shared<Camera>();

        cam->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(image_rgb, torch::kCPU);


        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << all_frame_num_;
        std::string formatted_str = ss.str();
        cam->image_name_ = "test_" + formatted_str + ".jpg";
        cam->image_stamp_sec_ = cur_frame.image_msg->header.stamp.toSec();

        cam->setIntrinsic(width, height, fx_, fy_, cx_, cy_);
        cam->setPose(q_wc.toRotationMatrix(), t_wc);
        cam->R_cw_pred_ = cam->R_cw_;
        cam->t_cw_pred_ = cam->t_cw_;
        cam->pose_cov_ = Eigen::Matrix<double, 6, 6>::Identity() * pose_init_cov_;
        cam->pose_refined_ = false;

        test_cameras_.emplace_back(cam);
    }

    all_frame_num_ += 1;
}

GaussianModel::GaussianModel(const Params& prm)
{
    sh_degree_ = prm.sh_degree;
    white_background_ = prm.white_background;
    random_background_ = prm.random_background;
    convert_SHs_python_ = prm.convert_SHs_python;
    compute_cov3D_python_ = prm.compute_cov3D_python;
    lambda_erank_ = prm.lambda_erank;
    scaling_scale_ = prm.scaling_scale;

    position_lr_ = prm.position_lr;
    feature_lr_ = prm.feature_lr;
    opacity_lr_ = prm.opacity_lr;
    scaling_lr_ = prm.scaling_lr;
    rotation_lr_ = prm.rotation_lr;
    lambda_dssim_ = prm.lambda_dssim;

    apply_exposure_ = prm.apply_exposure;
    exposure_lr_ = prm.exposure_lr;
    skybox_points_num_ = prm.skybox_points_num;
    skybox_radius_ = prm.skybox_radius;

    lambda_dist_ = prm.lambda_dist;
    lambda_normal_ = prm.lambda_normal;
    alpha_threshold_ = prm.alpha_threshold;
    slide_window_size_ = prm.slide_window_size;
    hiloss_threshold_ = prm.hiloss_threshold;
    hiColorLoss_threshold_ = prm.hiColorLoss_threshold;
    train_times_threshold_ = prm.train_times_threshold;
    scale_ratio_threshold_ = prm.scale_ratio_threshold;
    opacity_threshold_ = prm.opacity_threshold;
    enable_normal_cull_ = prm.enable_normal_cull;
    normal_z_cutoff_ = prm.normal_z_cutoff;
    spnet_epsilon_1_ = prm.spnet_epsilon_1;
    spnet_epsilon_2_ = prm.spnet_epsilon_2;
    spnet_patch_size_ = prm.spnet_patch_size;
    spnet_dilate_radius_ = prm.spnet_dilate_radius;
    spnet_depth_grad_threshold_ = prm.spnet_depth_grad_threshold;

    spnet_enforce_normal_face_camera_ = prm.spnet_enforce_normal_face_camera;

    da3_min_depth_ = prm.da3_min_depth;
    da3_max_depth_ = prm.da3_max_depth;
    da3_conf_threshold_ = prm.da3_conf_threshold;
    da3_align_min_points_ = prm.da3_align_min_points;
    da3_save_depth_vis_ = prm.da3_save_depth_vis;

    use_Gaussian_regress_ = prm.use_Gaussian_regress;
    apply_regressed_rotation_ = prm.apply_regressed_rotation;
    apply_regressed_position_ = prm.apply_regressed_position;
    apply_regressed_sh_dc_ = prm.apply_regressed_sh_dc;
    apply_regressed_sh_rest_ = prm.apply_regressed_sh_rest;
    if_prune_ = prm.if_prune;
    opacity_prune_forRegress_ = prm.opacity_prune_forRegress;

    dataset_path_ = prm.dataset_path_;
    generate_dataset_ = prm.generate_dataset_;
    opacity_modifier_ = prm.opacity_modifier;
    scale_modifier_ = prm.scale_modifier;
    opacity_modifier_up_ = prm.opacity_modifier_up;
    extend_debug_ = prm.extend_debug;
    if_tileCull_in_extend_ = prm.if_tileCull_in_extend;
    if_full_regress_ = prm.if_full_regress;


    enable_pose_refinement_ = prm.enable_pose_refinement;
    pose_max_iterations_ = prm.pose_max_iterations;
    pose_img_point_cov_ = prm.pose_img_point_cov;
    pose_patch_size_ = prm.pose_patch_size;
    pose_init_cov_ = prm.pose_init_cov;
    pose_conv_thresh_rot_ = prm.pose_conv_thresh_rot;
    pose_conv_thresh_pos_ = prm.pose_conv_thresh_pos;
    pose_refine_start_frame_ = prm.pose_refine_start_frame;
    pose_alpha_threshold_ = prm.pose_alpha_threshold;
    pose_huber_delta_ = prm.pose_huber_delta;
    pose_min_valid_pixels_ = prm.pose_min_valid_pixels;
    pose_rerender_each_iter_ = prm.pose_rerender_each_iter;
    pose_max_step_rot_deg_ = prm.pose_max_step_rot_deg;
    pose_max_step_trans_cm_ = prm.pose_max_step_trans_cm;
    pose_rmse_increase_tolerance_ratio_ = prm.pose_rmse_increase_tolerance_ratio;
    pose_min_alpha_coverage_ratio_ = prm.pose_min_alpha_coverage_ratio;
    pose_alpha_erode_radius_ = prm.pose_alpha_erode_radius;

    clamp_scale_ = prm.clamp_scale;
    dark_color_threshold_ = prm.dark_color_threshold;
    color_gradient_threshold_ = prm.color_gradient_threshold;
    voxel_size_ = prm.voxel_size;
    enable_regress_high_grad_traditional_init_ = prm.enable_regress_high_grad_traditional_init;
    regress_high_grad_keep_ratio_ = prm.regress_high_grad_keep_ratio;
    far_depth_threshold_ = prm.far_depth_threshold;


    densify_grad_threshold_   = prm.densify_grad_threshold;
    percent_dense_            = prm.percent_dense;
    densify_from_train_times_ = prm.densify_from_train_times;
    densification_interval_   = prm.densification_interval;
    densify_index_gap_        = prm.densify_index_gap;
    densify_max_per_round_    = prm.densify_max_per_round;
    densify_covis_window_     = prm.densify_covis_window;
    densify_min_train_after_covis_ = prm.densify_min_train_after_covis;
    densify_train_gate_alpha_ = prm.densify_train_gate_alpha;
    post_densify_window_radius_ = prm.post_densify_window_radius;
    post_densify_boost_rounds_  = prm.post_densify_boost_rounds;
    post_densify_boost_budget_  = prm.post_densify_boost_budget;
    opacity_cull_threshold_   = prm.opacity_cull_threshold;
    scene_extent_             = prm.scene_extent;
    densify_alpha_            = prm.densify_alpha;
    densify_new_opacity_scale_ = prm.densify_new_opacity_scale;
    densify_new_opacity_min_   = prm.densify_new_opacity_min;
    densify_newborn_boost_steps_ = prm.densify_newborn_boost_steps;
    densify_newborn_pos_lr_scale_ = prm.densify_newborn_pos_lr_scale;

    enable_train_visual_eval_ = prm.enable_train_visual_eval;
    train_visual_eval_every_k_train_times_ = prm.train_visual_eval_every_k_train_times;
    train_visual_eval_frame_ids_ = prm.train_visual_eval_frame_ids;
    train_visual_eval_output_dir_ = prm.train_visual_eval_output_dir;

    auto device_type = torch::kCUDA;
    GAUSSIAN_MODEL_INIT_TENSORS(device_type)

    is_init_ = false;

    t_forward_ = 0;
    t_backward_ = 0;
    t_step_ = 0;
    t_optlist_ = 0;
    t_tocuda_ = 0;

    std::string backbone_path = prm.backbone_path;
    if(std::filesystem::exists(backbone_path)){
        std::cout << "[GaussianModel] Loading backbone model..." << std::endl;
        feature_extractor_ = std::make_shared<FeatureExtractor>(backbone_path);
        if(feature_extractor_->is_loaded()){
            std::cout << "[GaussianModel] Backbone model loaded successfully." << std::endl;
        }
        else{
            std::cout << "[GaussianModel] Failed to load backbone model." << std::endl;
        }
    }
    else{
        std::cout << "[GaussianModel] Backbone model not found: " << backbone_path << std::endl;
    }

    if(use_Gaussian_regress_){
        std::cout << "[GaussianModel] Loading regression models..." << std::endl;
        std::string mlp_path = prm.mlp_path;


        if (std::filesystem::exists(mlp_path)) {
            gaussian_regressor_ = std::make_shared<GaussianRegressor>(mlp_path);
            if (gaussian_regressor_->is_loaded()) {
                std::cout << "[GaussianModel] Regression models loaded successfully." << std::endl;
            } else {
                std::cout << "[GaussianModel] Failed to load regression models." << std::endl;
            }
        } else {
            std::cout << "[GaussianModel] Model files not found: " << mlp_path << std::endl;
        }
    }
}

 torch::Tensor GaussianModel::getScaling()
 {
     return torch::exp(scaling_);
 }

 torch::Tensor GaussianModel::getRotation()
 {
     return torch::nn::functional::normalize(rotation_);
 }

 torch::Tensor GaussianModel::getXYZ()
 {
     return xyz_;
 }

 torch::Tensor GaussianModel::getFeaturesDc()
 {
     return features_dc_;
 }

 torch::Tensor GaussianModel::getFeaturesRest()
 {
     return features_rest_;
 }


 torch::Tensor GaussianModel::getFeatures()
 {


     if (features_rest_.size(0) > 0 && features_rest_.size(1) > 0) {
         return torch::cat({features_dc_, features_rest_}, 1);
     } else {
         return features_dc_;
     }
 }

torch::Tensor GaussianModel::getOpacity()
{
    return torch::sigmoid(opacity_);
}

torch::Tensor GaussianModel::getExposure()
{
    return exposure_;
}

struct RegressionInput{
    bool success;


    torch::Tensor regress_f_curr_;
    torch::Tensor regress_render_f_curr_;
    torch::Tensor regress_hist_f_;
    torch::Tensor regress_hist_render_f_;
    torch::Tensor regress_curr_dir_;
    torch::Tensor regress_hist_dir_;
    torch::Tensor regress_hist_mask_;

    torch::Tensor regress_inv_depth_;
    torch::Tensor regress_n_cam_;
    torch::Tensor regress_mask_tensor_;
    torch::Tensor regress_base_pixel_;
    torch::Tensor base_sh_;
    torch::Tensor regress_dis_;
};

struct RegressionResult {
    bool success;
    torch::Tensor scales;
    torch::Tensor rots;
    torch::Tensor opacities;
    torch::Tensor features_dc;
    torch::Tensor features_rest;
    torch::Tensor positions;
    torch::Tensor keep_mask;
    int64_t kept_count = 0;
    int64_t pruned_count = 0;
};

static torch::Tensor filter_tensor_by_keep_mask(
    const torch::Tensor& tensor,
    const torch::Tensor& keep_mask)
{
    if (!tensor.defined() || !keep_mask.defined()) {
        return tensor;
    }
    if (tensor.dim() == 0 || tensor.size(0) != keep_mask.size(0)) {
        return tensor;
    }
    return tensor.index({keep_mask});
}

static void log_tensor_validity_stats(
    const std::string& name,
    const torch::Tensor& tensor)
{
    if (!tensor.defined()) {
        std::cout << "[MLP][Check] " << name << " 未定义" << std::endl;
        return;
    }

    auto tensor_detached = tensor.detach();
    int64_t total_count = tensor_detached.numel();
    if (total_count == 0) {
        std::cout << "[MLP][Check] " << name << " 为空张量" << std::endl;
        return;
    }

    auto finite_mask = torch::isfinite(tensor_detached);
    int64_t finite_count = finite_mask.sum().item<int64_t>();
    int64_t invalid_count = total_count - finite_count;

    std::cout << "[MLP][Check] " << name
              << " total=" << total_count
              << ", invalid=" << invalid_count;

    if (finite_count > 0) {
        auto finite_values = tensor_detached.index({finite_mask});
        double min_val = finite_values.min().item<double>();
        double max_val = finite_values.max().item<double>();
        std::cout << ", min=" << min_val
                  << ", max=" << max_val;
    }

    std::cout << std::endl;
}

static void log_regression_health(
    const RegressionInput& rg_input,
    const RegressedGaussianParams& reg_result,
    const RegressionResult& res)
{
    bool need_detail_log = false;

    if (res.scales.defined() && !torch::isfinite(res.scales).all().item<bool>()) need_detail_log = true;
    if (res.rots.defined() && !torch::isfinite(res.rots).all().item<bool>()) need_detail_log = true;
    if (res.opacities.defined() && !torch::isfinite(res.opacities).all().item<bool>()) need_detail_log = true;
    if (res.positions.defined() && !torch::isfinite(res.positions).all().item<bool>()) need_detail_log = true;

    int64_t non_positive_inv_depth = 0;
    int64_t invalid_pred_inv_depth = 0;
    if (reg_result.delta_inv_depth.defined()) {
        auto pred_inv_depth = (rg_input.regress_inv_depth_ + reg_result.delta_inv_depth).detach();
        invalid_pred_inv_depth = (~torch::isfinite(pred_inv_depth)).sum().item<int64_t>();
        non_positive_inv_depth = (pred_inv_depth <= 0).sum().item<int64_t>();
        if (invalid_pred_inv_depth > 0 || non_positive_inv_depth > 0) {
            need_detail_log = true;
        }
    }

    bool rot_norm_bad = false;
    double rot_norm_min = 0.0;
    double rot_norm_max = 0.0;
    if (res.rots.defined() && res.rots.numel() > 0) {
        auto rot_norm = torch::norm(res.rots.detach(), 2, 1);
        rot_norm_min = rot_norm.min().item<double>();
        rot_norm_max = rot_norm.max().item<double>();
        rot_norm_bad = !torch::isfinite(rot_norm).all().item<bool>() || rot_norm_min < 0.5 || rot_norm_max > 1.5;
        if (rot_norm_bad) {
            need_detail_log = true;
        }
    }

    std::cout << "[MLP][Check] summary"
              << " scale_finite=" << (res.scales.defined() ? torch::isfinite(res.scales).all().item<bool>() : 0)
              << ", rot_finite=" << (res.rots.defined() ? torch::isfinite(res.rots).all().item<bool>() : 0)
              << ", opacity_finite=" << (res.opacities.defined() ? torch::isfinite(res.opacities).all().item<bool>() : 0)
              << ", position_finite=" << (res.positions.defined() ? torch::isfinite(res.positions).all().item<bool>() : 0)
              << ", kept=" << res.kept_count
              << ", pruned=" << res.pruned_count
              << ", pred_inv_depth_invalid=" << invalid_pred_inv_depth
              << ", pred_inv_depth_non_positive=" << non_positive_inv_depth;
    if (res.rots.defined() && res.rots.numel() > 0) {
        std::cout << ", rot_norm_min=" << rot_norm_min
                  << ", rot_norm_max=" << rot_norm_max;
    }
    if (res.opacities.defined() && res.opacities.numel() > 0){
        auto opacity_min = res.opacities.min().item<double>();
        auto opacity_max = res.opacities.max().item<double>();

        auto real_opacity_min = torch::sigmoid(torch::tensor(opacity_min)).item<double>();
        auto real_opacity_max = torch::sigmoid(torch::tensor(opacity_max)).item<double>();
        std::cout << ", opac_real_min=" << real_opacity_min
                  << ", opac_real_max=" << real_opacity_max;
    }
    std::cout << std::endl;

    if (!need_detail_log) {
        return;
    }

    std::cout << "[MLP][Check] 检测到回归结果中存在可疑值，下面输出详细统计。" << std::endl;
    log_tensor_validity_stats("reg_result.scale", reg_result.scale);
    log_tensor_validity_stats("reg_result.rotation", reg_result.rotation);
    log_tensor_validity_stats("reg_result.opacity", reg_result.opacity);
    log_tensor_validity_stats("reg_result.delta_pixel", reg_result.delta_pixel);
    log_tensor_validity_stats("reg_result.delta_inv_depth", reg_result.delta_inv_depth);
    log_tensor_validity_stats("res.scales", res.scales);
    log_tensor_validity_stats("res.rots", res.rots);
    log_tensor_validity_stats("res.opacities", res.opacities);
    log_tensor_validity_stats("res.positions", res.positions);

    if (reg_result.delta_inv_depth.defined()) {
        auto pred_inv_depth = (rg_input.regress_inv_depth_ + reg_result.delta_inv_depth).detach();
        log_tensor_validity_stats("pred_inv_depth", pred_inv_depth);
    }
}

RegressionInput prepare_rg_input(
    GaussianModel* pc,
    const std::shared_ptr<Dataset>& dataset,
    const std::shared_ptr<Camera>& cam,
    const torch::Tensor& fused_points,
    const torch::Tensor& fused_depths,
    const torch::Tensor& fused_valid_mask,
    const torch::Tensor& fused_pixels,
    const torch::Tensor& fused_normals,
    const torch::Tensor& fused_colors,
    const torch::Tensor& render_mask)
{
    int total_num = fused_points.size(0);
    int H = cam->image_height_;
    int W = cam->image_width_;


    extract_features(pc, cam);

    if (!cam->feature_map_.defined()) {
        std::cerr << "[MLP] Feature extraction failed." << std::endl;
        return {false, {}, {}, {}, {}, {}, {}};
    }


    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();

    auto mask_tensor = fused_valid_mask.to(torch::kFloat32);
    if (mask_tensor.dim() == 1) {
        mask_tensor = mask_tensor.unsqueeze(1);
    }

    std::shared_ptr<GaussianModel> pc_ptr(pc, [](GaussianModel*){});
    auto ctx_feats = compute_context_features(
        dataset, cam, fused_points, fused_normals, fused_pixels, pc->slide_window_size_, pc_ptr, bg, render_mask);
    auto f_curr = std::get<0>(ctx_feats);
    auto render_f_curr = std::get<1>(ctx_feats);
    auto curr_dir = std::get<2>(ctx_feats);
    auto hist_f = std::get<3>(ctx_feats);
    auto hist_render_f = std::get<4>(ctx_feats);
    auto hist_dir = std::get<5>(ctx_feats);
    auto hist_mask = std::get<6>(ctx_feats);


    auto inv_depth = 1.0f / (fused_depths + 1e-6);
    if (inv_depth.dim() == 1) {
        inv_depth = inv_depth.unsqueeze(1);
    }
    auto n_cam = fused_normals;

    auto R_wc_t = tensor_utils::EigenMatrix2TorchTensor(cam->R_cw_.transpose().cast<float>(), torch::kCUDA);
    auto R_wc_batch = R_wc_t.unsqueeze(0).expand({total_num, 3, 3});

    auto knn_dis = compute_knn_distance(fused_pixels);

    RegressionInput rg_input = {
        true,
        f_curr,
        render_f_curr,
        hist_f,
        hist_render_f,
        curr_dir,
        hist_dir,
        hist_mask,
        inv_depth,
        n_cam,
        mask_tensor,
        fused_pixels,
        fused_colors,
        knn_dis
    };
    rg_input.success = rg_input.regress_f_curr_.size(0) > 0;
    return rg_input;
}

RegressionResult regress_gaussians(
    GaussianModel* pc,
    const std::shared_ptr<Camera>& cam,
    const RegressionInput& rg_input,
    float focal)
{
    int total_num = rg_input.regress_f_curr_.size(0);

    auto R_wc_t = tensor_utils::EigenMatrix2TorchTensor(cam->R_cw_.transpose().cast<float>(), torch::kCUDA);
    auto R_wc_batch = R_wc_t.unsqueeze(0).expand({total_num, 3, 3});


    auto reg_result = pc->gaussian_regressor_->regress(
        rg_input.regress_f_curr_, rg_input.regress_render_f_curr_,
        rg_input.regress_hist_f_, rg_input.regress_hist_render_f_,
        rg_input.regress_curr_dir_, rg_input.regress_hist_dir_,
        rg_input.regress_hist_mask_, rg_input.regress_inv_depth_,
        rg_input.regress_n_cam_, rg_input.regress_mask_tensor_,
        rg_input.regress_dis_, R_wc_batch);

    RegressionResult res;
    res.success = reg_result.scale.defined();

    if (res.success) {
         res.keep_mask = torch::ones({total_num},
                                     torch::TensorOptions().dtype(torch::kBool).device(reg_result.scale.device()));
         if (pc->opacity_prune_forRegress_ > 0.0 && reg_result.opacity.defined()) {
             auto raw_opacity = reg_result.opacity.reshape({-1});
             res.keep_mask = torch::logical_and(
                 torch::isfinite(raw_opacity),
                 raw_opacity >= pc->opacity_prune_forRegress_);
         }
         res.kept_count = res.keep_mask.sum().item<int64_t>();
         res.pruned_count = total_num - res.kept_count;


         auto s_norm = reg_result.scale;

         auto s_final = s_norm / (rg_input.regress_inv_depth_ * focal + 1e-6);
         res.scales = torch::log(s_final + 1e-8);


         res.rots = reg_result.rotation;


         auto safe_opacity = torch::clamp(reg_result.opacity, 1e-6, 0.85);
         res.opacities = general_utils::inverse_sigmoid(safe_opacity) * pc->opacity_modifier_;


         res.features_dc = (reg_result.color_dc.transpose(1, 2) + rg_input.base_sh_.unsqueeze(2)).contiguous();
         res.features_rest = reg_result.color_rest.transpose(1, 2).contiguous();


         if (reg_result.delta_pixel.defined() && reg_result.delta_inv_depth.defined() && rg_input.regress_base_pixel_.defined()) {
             auto pred_inv_depth = rg_input.regress_inv_depth_ + reg_result.delta_inv_depth;
             auto pred_z = 1.0f / (pred_inv_depth + 1e-6);

             auto pred_u = rg_input.regress_base_pixel_.select(1, 0).unsqueeze(1) + reg_result.delta_pixel.select(1, 0).unsqueeze(1);
             auto pred_v = rg_input.regress_base_pixel_.select(1, 1).unsqueeze(1) + reg_result.delta_pixel.select(1, 1).unsqueeze(1);

             float fx = cam->fx_;
             float fy = cam->fy_;
             float cx = cam->cx_;
             float cy = cam->cy_;

             auto pred_x = (pred_u - cx) * pred_z / fx;
             auto pred_y = (pred_v - cy) * pred_z / fy;

             auto p_cam = torch::cat({pred_x, pred_y, pred_z}, 1).unsqueeze(-1);


             auto t_wc_tensor = cam->camera_center_.to(torch::kCUDA).view({1, 3, 1}).expand({total_num, 3, 1});
             auto p_world = torch::baddbmm(t_wc_tensor, R_wc_batch, p_cam);
             res.positions = p_world.squeeze(-1);
         }

         if (res.pruned_count > 0) {
             res.scales = filter_tensor_by_keep_mask(res.scales, res.keep_mask);
             res.rots = filter_tensor_by_keep_mask(res.rots, res.keep_mask);
             res.opacities = filter_tensor_by_keep_mask(res.opacities, res.keep_mask);
             res.features_dc = filter_tensor_by_keep_mask(res.features_dc, res.keep_mask);
             res.features_rest = filter_tensor_by_keep_mask(res.features_rest, res.keep_mask);
             res.positions = filter_tensor_by_keep_mask(res.positions, res.keep_mask);
             std::cout << "[MLP] Pruned " << res.pruned_count
                       << " regressed Gaussians by opacity threshold "
                       << pc->opacity_prune_forRegress_
                       << ", kept " << res.kept_count << std::endl;
         }

         log_regression_health(rg_input, reg_result, res);
    }

    return res;
}

void GaussianModel::initialize(const std::shared_ptr<Dataset>& dataset,
                               std::shared_ptr<SPNetWrapper> spnet,
                               std::shared_ptr<DA3Wrapper> da3)
{

    int num_lidar = static_cast<int>(dataset->pointcloud_.size());
    assert(num_lidar > 0);

    std::shared_ptr<Camera> init_cam = dataset->train_cameras_.empty() ? nullptr : dataset->train_cameras_.back();
    if (init_cam == nullptr) {
        std::cerr << "[GaussianModel] No camera found for initialization!" << std::endl;
        return;
    }

    int deg_2 = (sh_degree_ + 1) * (sh_degree_ + 1);
    float focal = static_cast<float>((init_cam->fx_ + init_cam->fy_) / 2.0);


    torch::Tensor sparse_depth = process_lidar_projection(dataset, init_cam);


    DepthCompletionParams params;
    params.epsilon_1 = spnet_epsilon_1_;
    params.epsilon_2 = spnet_epsilon_2_;
    params.patch_size = spnet_patch_size_;
    params.dilate_radius = spnet_dilate_radius_;
    params.depth_grad_threshold = spnet_depth_grad_threshold_;
    if (spnet) params.max_depth = spnet->get_max_depth();

    SPNetPointsResult sp_result;


    torch::Tensor rgb = init_cam->original_image_.cuda();


    if (spnet != nullptr && spnet->is_loaded()) {

        sp_result = get_spnet_points(*spnet, rgb, sparse_depth, params, init_cam);

        std::cout << "\033[1;32m [SPNet] Generated " << sp_result.fused_points.size(0)
                  << " points (including LiDAR & Supplement)\033[0m" << std::endl;
    } else {
        std::cerr << "[GaussianModel] SPNet not loaded, initialization might be incomplete." << std::endl;
        return;
    }


    int total_num = sp_result.fused_points.size(0);
    if (total_num == 0) return;






    torch::Tensor da3_normals_for_points;
    torch::Tensor da3_valid_mask_for_points;
    if (da3 != nullptr && da3->is_loaded()) {
        auto da3_guidance = build_da3_guidance(
            *da3, rgb, sparse_depth, init_cam,
            da3_conf_threshold_, da3_min_depth_, da3_max_depth_, da3_align_min_points_,
            spnet_enforce_normal_face_camera_, da3_save_depth_vis_, da3_debug_dir_);
        std::tie(da3_normals_for_points, da3_valid_mask_for_points) =
            sample_da3_guidance_by_pixels(da3_guidance, sp_result.fused_pixels);
    } else {
        std::cerr << "[GaussianModel] DA3 not loaded, all normals will fallback to view-ray." << std::endl;
        da3_normals_for_points = torch::zeros({total_num, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(sp_result.fused_points.device()));
        da3_valid_mask_for_points = torch::zeros({total_num, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(sp_result.fused_points.device()));
    }

    torch::Tensor fused_point_cloud = sp_result.fused_points;
    torch::Tensor fused_color = RGB2SH(sp_result.fused_colors);
    torch::Tensor features;
    torch::Tensor scales;
    torch::Tensor opacities;
    torch::Tensor rots;

    if (!this->feature_extractor_->is_loaded()) {
        std::cerr << "[GaussianModel] Feature extractor not loaded!" << std::endl;
        return;
    }

    if (da3_valid_mask_for_points.dim() == 1) {
        da3_valid_mask_for_points = da3_valid_mask_for_points.unsqueeze(1);
    }
    if (sp_result.fused_depths.dim() == 1) {
        sp_result.fused_depths = sp_result.fused_depths.unsqueeze(1);
    }

    if (use_Gaussian_regress_ && this->gaussian_regressor_ && this->gaussian_regressor_->is_loaded()) {
        auto rg_input = prepare_rg_input(this, dataset, init_cam, sp_result.fused_points,
                                sp_result.fused_depths, da3_valid_mask_for_points,
                                sp_result.fused_pixels, da3_normals_for_points, fused_color, torch::Tensor());

        if (rg_input.success) {
            auto res = regress_gaussians(this, init_cam, rg_input, focal);
            if (res.success) {
                sp_result.fused_points = filter_tensor_by_keep_mask(sp_result.fused_points, res.keep_mask);
                sp_result.fused_colors = filter_tensor_by_keep_mask(sp_result.fused_colors, res.keep_mask);
                fused_point_cloud = filter_tensor_by_keep_mask(fused_point_cloud, res.keep_mask);
                fused_color = filter_tensor_by_keep_mask(fused_color, res.keep_mask);
                da3_normals_for_points = filter_tensor_by_keep_mask(da3_normals_for_points, res.keep_mask);
                da3_valid_mask_for_points = filter_tensor_by_keep_mask(da3_valid_mask_for_points, res.keep_mask);
                sp_result.fused_pixels = filter_tensor_by_keep_mask(sp_result.fused_pixels, res.keep_mask);
                sp_result.fused_depths = filter_tensor_by_keep_mask(sp_result.fused_depths, res.keep_mask);

                const int64_t regressed_num = res.kept_count;
                scales = res.scales;
                opacities = res.opacities;
                auto features_dc = apply_regressed_sh_dc_
                    ? res.features_dc
                    : fused_color.unsqueeze(2).contiguous();
                auto features_rest = apply_regressed_sh_rest_
                    ? res.features_rest
                    : torch::zeros({regressed_num, 3, deg_2 - 1}, fused_color.options());
                features = torch::cat({features_dc, features_rest}, 2).contiguous();

                if (apply_regressed_rotation_) {
                    rots = res.rots;
                } else {
                    rots = compute_rotation_from_normals(
                        da3_normals_for_points,
                        da3_valid_mask_for_points,
                        sp_result.fused_pixels,
                        sp_result.fused_depths,
                        init_cam);
                }

                if (apply_regressed_position_ && res.positions.defined()) {
                    fused_point_cloud = res.positions;
                }
                std::cout << "[MLP] Initialized " << regressed_num << " Gaussians with Regression." << std::endl;
            }
            else {
                std::cerr << "[GaussianModel] Regression failed!" << std::endl;
                return;
            }
        }
    }
    else {
        features = torch::zeros({total_num, 3, deg_2}, torch::kFloat32).cuda();
        features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}, fused_color);

        scales = torch::log(scaling_scale_ * sp_result.fused_depths / focal).repeat({1, 2});
        opacities = general_utils::inverse_sigmoid(0.1f * torch::ones({total_num, 1}, torch::kFloat32).cuda());
        rots = compute_rotation_from_normals(
            da3_normals_for_points,
            da3_valid_mask_for_points,
            sp_result.fused_pixels,
            sp_result.fused_depths,
            init_cam);
    }
    if (generate_dataset_) {


        init_cam->raw_points_ = sp_result.fused_points.cpu();
        init_cam->raw_depths_ = sp_result.fused_depths.cpu();
        init_cam->raw_normals_ = da3_normals_for_points.cpu();
        init_cam->raw_pixels_ = sp_result.fused_pixels.cpu();
        init_cam->raw_valid_mask_ = da3_valid_mask_for_points.cpu();
        init_cam->raw_colors_ = fused_color.cpu();
        init_cam->raw_focal_ = focal;
    }


    if (skybox_points_num_ > 0)
    {
        int num = skybox_points_num_;
        double radius = skybox_radius_;
        torch::Tensor pi = torch::acos(torch::tensor(-1.0, torch::kFloat32).cuda());
        torch::Tensor theta = 2.0 * pi * torch::rand({num}, torch::kFloat32).cuda();
        torch::Tensor phi = torch::acos(1.0 - 1.4 * torch::rand({num}, torch::kFloat32).cuda());
        torch::Tensor sky_fused_point_cloud = torch::zeros({num, 3}, torch::kFloat32).cuda();
        sky_fused_point_cloud.index({torch::indexing::Slice(), 0}) = radius * 10 * torch::cos(theta) * torch::sin(phi);
        sky_fused_point_cloud.index({torch::indexing::Slice(), 1}) = radius * 10 * torch::sin(theta) * torch::sin(phi);
        sky_fused_point_cloud.index({torch::indexing::Slice(), 2}) = radius * 10 * torch::cos(phi);

        torch::Tensor sky_features = torch::zeros({num, 3, deg_2}, torch::kFloat32).cuda();
        sky_features.index({torch::indexing::Slice(), 0, 0}) = 0.7;
        sky_features.index({torch::indexing::Slice(), 1, 0}) = 0.8;
        sky_features.index({torch::indexing::Slice(), 2, 0}) = 0.95;

        torch::Tensor point_cloud_copy = sky_fused_point_cloud.clone();
        torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
        torch::Tensor sky_scales = torch::log(torch::sqrt(dist2));
        sky_scales = sky_scales.unsqueeze(1).repeat({1, 2});
        torch::Tensor sky_rots = torch::zeros({num, 4}, torch::kFloat32).cuda();
        sky_rots.index({torch::indexing::Slice(), 0}) = 1;
        torch::Tensor sky_opacities = general_utils::inverse_sigmoid(0.7f * torch::ones({num, 1}, torch::kFloat32).cuda());

        fused_point_cloud = torch::cat({sky_fused_point_cloud, fused_point_cloud}, 0);
        features = torch::cat({sky_features, features}, 0);
        scales = torch::cat({sky_scales, scales}, 0);
        rots = torch::cat({sky_rots, rots}, 0);
        opacities = torch::cat({sky_opacities, opacities}, 0);
    }

    this->xyz_ = fused_point_cloud.requires_grad_();

    this->features_dc_ = features.index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous().requires_grad_();
    this->features_rest_ = features.index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(1, features.size(2))}).transpose(1, 2).contiguous().requires_grad_();
    this->scaling_ = scales.requires_grad_();
    this->rotation_ = rots.requires_grad_();
    this->opacity_ = opacities.requires_grad_();

    if (apply_exposure_)
    {
        torch::Tensor exposure = torch::eye(3, torch::kFloat32).cuda();
        exposure = torch::cat({exposure, torch::zeros({3, 1}, torch::kFloat32).cuda()}, 1);
        this->exposure_ = exposure.requires_grad_();
    }

    GAUSSIAN_MODEL_TENSORS_TO_VEC


    int num_total = fused_point_cloud.size(0);
    this->global_ids_ = torch::arange(num_total, torch::kInt64).view({num_total, 1}).cuda();
    this->max_id_ = num_total;


    torch::Tensor scene_ids;
    if (skybox_points_num_ > 0) {
        scene_ids = this->global_ids_.index({torch::indexing::Slice(skybox_points_num_)}).flatten();
    } else {
        scene_ids = this->global_ids_.flatten();
    }


    scene_ids = scene_ids.cpu().contiguous();
    int64_t* scene_ids_ptr = scene_ids.data_ptr<int64_t>();
    init_cam->added_ids_.assign(scene_ids_ptr, scene_ids_ptr + scene_ids.size(0));

    int num_supplement = std::max<int>(0, scene_ids.size(0) - num_lidar);

    std::cout << std::fixed << std::setprecision(2)
            << "\033[1;37m Init Map with "
            << double(fused_point_cloud.size(0)) / 10000 << "w GS"
            << " (LiDAR: " << num_lidar << ", SPNet: " << num_supplement << ")"
            << ",\033[0m";

    dataset->pointcloud_.clear();
    dataset->pointcolor_.clear();
    dataset->pointdepth_.clear();
}

void GaussianModel::saveCameraRegressData(const std::shared_ptr<Camera>& cam, const std::string& result_path, int frame_idx) {
    if (!cam) {
        return;
    }


    if (!cam->regress_f_curr_.defined() || cam->regress_f_curr_.size(0) == 0) {

        return;
    }

    std::string base_path = result_path;
    std::string frames_dir = base_path + "/frames";
    if (!fs::exists(frames_dir)) fs::create_directories(frames_dir);

    std::string seed_filename_base = "frame_" + std::to_string(frame_idx) + "_seeds";

    torch::Tensor added_ids_tensor = torch::from_blob(const_cast<int64_t*>(cam->added_ids_.data()), {static_cast<long>(cam->added_ids_.size())}, torch::kInt64).clone();


    Eigen::Matrix3d R_wc_eigen = cam->R_cw_.transpose();
    torch::Tensor R_wc_tensor = torch::tensor({
        {R_wc_eigen(0,0), R_wc_eigen(0,1), R_wc_eigen(0,2)},
        {R_wc_eigen(1,0), R_wc_eigen(1,1), R_wc_eigen(1,2)},
        {R_wc_eigen(2,0), R_wc_eigen(2,1), R_wc_eigen(2,2)}
    }, torch::kFloat32);


    int64_t num_pts = cam->regress_f_curr_.size(0);
    R_wc_tensor = R_wc_tensor.unsqueeze(0).expand({num_pts, 3, 3}).clone();


    std::vector<torch::Tensor> tensors_to_save = {
        cam->regress_f_curr_.cpu(),
        cam->regress_render_f_curr_.cpu(),
        cam->regress_hist_f_.cpu(),
        cam->regress_hist_render_f_.cpu(),
        cam->regress_curr_dir_.cpu(),
        cam->regress_hist_dir_.cpu(),
        cam->regress_hist_mask_.cpu(),
        cam->regress_inv_depth_.cpu(),
        cam->regress_n_cam_.cpu(),
        cam->regress_mask_tensor_.cpu(),
        added_ids_tensor,
        R_wc_tensor,
        cam->base_sh_.cpu(),
        cam->regress_base_pixel_.cpu(),
        cam->regress_dis_.cpu()
    };
    torch::save(tensors_to_save, frames_dir + "/" + seed_filename_base + ".pt");
    std::cout << "\033[1;32m [Dataset] Saved camera regression data of frame " << frame_idx << " \033[0m" << std::endl;


    cam->regress_f_curr_ = torch::Tensor();
    cam->regress_render_f_curr_ = torch::Tensor();
    cam->regress_hist_f_ = torch::Tensor();
    cam->regress_hist_render_f_ = torch::Tensor();
    cam->regress_curr_dir_ = torch::Tensor();
    cam->regress_hist_dir_ = torch::Tensor();
    cam->regress_hist_mask_ = torch::Tensor();
    cam->regress_inv_depth_ = torch::Tensor();
    cam->regress_n_cam_ = torch::Tensor();
    cam->regress_mask_tensor_ = torch::Tensor();
    cam->regress_base_pixel_ = torch::Tensor();
    cam->base_sh_ = torch::Tensor();
    cam->regress_dis_ = torch::Tensor();

}

void GaussianModel::saveDataset(const std::string& path, const std::shared_ptr<Dataset>& dataset, int iter)
{
    (void)iter;
    std::string base_path = path;
    if (!fs::exists(base_path)) fs::create_directories(base_path);


    std::string train_meta_path = base_path + "/train_cameras.json";
    std::ofstream train_meta(train_meta_path);
    train_meta << "[\n";

    std::string images_dir = base_path + "/images";
    if (!fs::exists(images_dir)) fs::create_directories(images_dir);



    if (generate_dataset_ && feature_extractor_ && feature_extractor_->is_loaded()) {
        std::cout << "\033[1;36m [Dataset] Starting Retrospective Feature Extraction for ALL frames \033[0m" << std::endl;


        torch::Tensor bg;
        if (white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
        else bg = torch::zeros({3}, torch::kFloat32).cuda();

        for (size_t i = 0; i < dataset->train_cameras_.size(); ++i) {
            auto K_cam = dataset->train_cameras_[i];
            if (!K_cam->raw_points_.defined()) continue;


            int64_t K_min_id = 0;
            if (!K_cam->added_ids_.empty()) {
                K_min_id = K_cam->added_ids_.front();
            }
            auto keep_mask = (this->global_ids_ < K_min_id).flatten();


            auto pts = K_cam->raw_points_.cuda();
            auto depths = K_cam->raw_depths_.cuda();
            auto valid_mask = K_cam->raw_valid_mask_.cuda();
            auto pixels = K_cam->raw_pixels_.cuda();
            auto normals = K_cam->raw_normals_.cuda();
            auto colors_sh = K_cam->raw_colors_.cuda();



            std::cout << "[Dataset] 开始回溯提取 frame " << i
                      << " 的回归输入，points=" << pts.size(0)
                      << ", keep_mask_true=" << keep_mask.sum().item<int64_t>()
                      << std::endl;
            auto rg_input = prepare_rg_input(this, dataset, K_cam,
                                             pts, depths, valid_mask,
                                             pixels, normals, colors_sh, keep_mask);




            torch::cuda::synchronize();

            if (rg_input.success) {
                logDatasetTensorHealth("rg_input.regress_f_curr", rg_input.regress_f_curr_);
                logDatasetTensorHealth("rg_input.regress_render_f_curr", rg_input.regress_render_f_curr_);
                logDatasetTensorHealth("rg_input.regress_hist_f", rg_input.regress_hist_f_);
                logDatasetTensorHealth("rg_input.regress_hist_render_f", rg_input.regress_hist_render_f_);
                logDatasetTensorHealth("rg_input.regress_inv_depth", rg_input.regress_inv_depth_);
                logDatasetTensorHealth("rg_input.regress_base_pixel", rg_input.regress_base_pixel_);

                K_cam->regress_f_curr_ = rg_input.regress_f_curr_.cpu();
                K_cam->regress_render_f_curr_ = rg_input.regress_render_f_curr_.cpu();
                K_cam->regress_hist_f_ = rg_input.regress_hist_f_.cpu();
                K_cam->regress_hist_render_f_ = rg_input.regress_hist_render_f_.cpu();
                K_cam->regress_curr_dir_ = rg_input.regress_curr_dir_.cpu();
                K_cam->regress_hist_dir_ = rg_input.regress_hist_dir_.cpu();
                K_cam->regress_hist_mask_ = rg_input.regress_hist_mask_.cpu();

                K_cam->regress_inv_depth_ = rg_input.regress_inv_depth_.cpu();
                K_cam->regress_n_cam_ = rg_input.regress_n_cam_.cpu();
                K_cam->regress_mask_tensor_ = rg_input.regress_mask_tensor_.cpu();
                K_cam->base_sh_ = rg_input.base_sh_.cpu();
                K_cam->regress_base_pixel_ = rg_input.regress_base_pixel_.cpu();
                K_cam->regress_dis_ = rg_input.regress_dis_.cpu();
            } else {
                std::cerr << "[Dataset] frame " << i
                          << " 的回归输入提取失败，rg_input.success=false。" << std::endl;
            }


            K_cam->raw_points_ = torch::Tensor();
            K_cam->raw_depths_ = torch::Tensor();
            K_cam->raw_valid_mask_ = torch::Tensor();
            K_cam->raw_pixels_ = torch::Tensor();
            K_cam->raw_normals_ = torch::Tensor();
            K_cam->raw_colors_ = torch::Tensor();
            K_cam->raw_focal_ = 0.0f;


            saveCameraRegressData(K_cam, path, i);


            int window_size = slide_window_size_;
            int limit_idx = (int)i - window_size;
            if (limit_idx >= 0 && limit_idx < dataset->train_cameras_.size()) {
                auto& cam_to_clear = dataset->train_cameras_[limit_idx];
                if (cam_to_clear->feature_map_.defined()) {
                   cam_to_clear->feature_map_ = torch::Tensor();
                }
            }
        }
    }

    for (size_t i = 0; i < dataset->train_cameras_.size(); ++i) {
        auto cam = dataset->train_cameras_[i];


        std::string img_filename = "train_" + std::to_string(i) + ".png";
        cv::Mat mat = tensor_utils::torchTensor2CvMat_Float32(cam->original_image_);
        cv::Mat mat_8u;
        mat.convertTo(mat_8u, CV_8UC3, 255.0);
        cv::cvtColor(mat_8u, mat_8u, cv::COLOR_RGB2BGR);
        cv::imwrite(images_dir + "/" + img_filename, mat_8u);


        std::string seed_filename_base = "frame_" + std::to_string(i) + "_seeds.pt";

        Eigen::Quaterniond q(cam->R_cw_.transpose());
        Eigen::Vector3d t = - cam->R_cw_.transpose() * cam->t_cw_;

        train_meta << "  {\n";
        train_meta << "    \"id\": " << i << ",\n";
        train_meta << "    \"img_name\": \"" << img_filename << "\",\n";
        train_meta << "    \"width\": " << cam->image_width_ << ",\n";
        train_meta << "    \"height\": " << cam->image_height_ << ",\n";
        train_meta << "    \"fx\": " << cam->fx_ << ",\n";
        train_meta << "    \"fy\": " << cam->fy_ << ",\n";
        train_meta << "    \"cx\": " << cam->cx_ << ",\n";
        train_meta << "    \"cy\": " << cam->cy_ << ",\n";
        train_meta << "    \"position\": [" << t.x() << ", " << t.y() << ", " << t.z() << "],\n";
        train_meta << "    \"rotation\": [" << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z() << "],\n";
        train_meta << "    \"seed_file\": \"" << seed_filename_base << "\"\n";

        if (i < dataset->train_cameras_.size() - 1) train_meta << "  },\n";
        else train_meta << "  }\n";
    }
    train_meta << "]\n";
    train_meta.close();


    std::string test_meta_path = base_path + "/test_cameras.json";
    std::ofstream test_meta(test_meta_path);
    test_meta << "[\n";
    for (size_t i = 0; i < dataset->test_cameras_.size(); ++i) {
        auto cam = dataset->test_cameras_[i];

        std::string img_filename = "test_" + std::to_string(i) + ".png";
        cv::Mat mat = tensor_utils::torchTensor2CvMat_Float32(cam->original_image_);
        cv::Mat mat_8u;
        mat.convertTo(mat_8u, CV_8UC3, 255.0);
        cv::cvtColor(mat_8u, mat_8u, cv::COLOR_RGB2BGR);
        cv::imwrite(images_dir + "/" + img_filename, mat_8u);

        Eigen::Quaterniond q(cam->R_cw_.transpose());
        Eigen::Vector3d t = - cam->R_cw_.transpose() * cam->t_cw_;

        test_meta << "  {\n";
        test_meta << "    \"id\": " << i << ",\n";
        test_meta << "    \"img_name\": \"" << img_filename << "\",\n";
        test_meta << "    \"width\": " << cam->image_width_ << ",\n";
        test_meta << "    \"height\": " << cam->image_height_ << ",\n";
        test_meta << "    \"fx\": " << cam->fx_ << ",\n";
        test_meta << "    \"fy\": " << cam->fy_ << ",\n";
        test_meta << "    \"cx\": " << cam->cx_ << ",\n";
        test_meta << "    \"cy\": " << cam->cy_ << ",\n";
        test_meta << "    \"position\": [" << t.x() << ", " << t.y() << ", " << t.z() << "],\n";
        test_meta << "    \"rotation\": [" << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z() << "]\n";

        if (i < dataset->test_cameras_.size() - 1) test_meta << "  },\n";
        else test_meta << "  }\n";
    }
    test_meta << "]\n";
    test_meta.close();

    std::cout << "\033[1;32m [Dataset] Saved dataset to " << base_path << " \033[0m" << std::endl;
    this->saveMap(base_path);
    std::cout << "\033[1;32m [Dataset] Saved map to " << base_path << " \033[0m" << std::endl;
}



void GaussianModel::saveMap(const std::string& result_path)
{
    std::string pc_path = result_path + "/point_cloud.ply";

    torch::Tensor xyz = this->xyz_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();

    torch::Tensor f_dc = this->features_dc_.index({torch::indexing::Slice(skybox_points_num_)}).detach().transpose(1, 2).flatten(1).contiguous().cpu();
    torch::Tensor f_rest = this->features_rest_.index({torch::indexing::Slice(skybox_points_num_)}).detach().transpose(1, 2).flatten(1).contiguous().cpu();
    torch::Tensor opacities = this->opacity_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();
    torch::Tensor scale = this->scaling_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();
    torch::Tensor rotation = this->rotation_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();
    torch::Tensor ids = this->global_ids_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();

    std::filebuf fb_binary;
    fb_binary.open(pc_path, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);

    tinyply::PlyFile result_file;


    result_file.add_properties_to_element(
        "vertex", {"x", "y", "z"},
        tinyply::Type::FLOAT32, xyz.size(0),
        reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()),
        tinyply::Type::INVALID, 0);









    std::size_t n_f_dc = this->features_dc_.size(1) * this->features_dc_.size(2);
    std::vector<std::string> property_names_f_dc(n_f_dc);
    for (int i = 0; i < n_f_dc; ++i)
        property_names_f_dc[i] = "f_dc_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_f_dc,
        tinyply::Type::FLOAT32, this->features_dc_.size(0),
        reinterpret_cast<uint8_t*>(f_dc.data_ptr<float>()),
        tinyply::Type::INVALID, 0);


    std::size_t n_f_rest = this->features_rest_.size(1) * this->features_rest_.size(2);
    std::vector<std::string> property_names_f_rest(n_f_rest);
    for (int i = 0; i < n_f_rest; ++i)
        property_names_f_rest[i] = "f_rest_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_f_rest,
        tinyply::Type::FLOAT32, this->features_rest_.size(0),
        reinterpret_cast<uint8_t*>(f_rest.data_ptr<float>()),
        tinyply::Type::INVALID, 0);


    result_file.add_properties_to_element(
        "vertex", {"opacity"},
        tinyply::Type::FLOAT32, opacities.size(0),
        reinterpret_cast<uint8_t*>(opacities.data_ptr<float>()),
        tinyply::Type::INVALID, 0);


    std::size_t n_scale = scale.size(1);
    std::vector<std::string> property_names_scale(n_scale);
    for (int i = 0; i < n_scale; ++i)
        property_names_scale[i] = "scale_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_scale,
        tinyply::Type::FLOAT32, scale.size(0),
        reinterpret_cast<uint8_t*>(scale.data_ptr<float>()),
        tinyply::Type::INVALID, 0);


    std::size_t n_rotation = rotation.size(1);
    std::vector<std::string> property_names_rotation(n_rotation);
    for (int i = 0; i < n_rotation; ++i)
        property_names_rotation[i] = "rot_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_rotation,
        tinyply::Type::FLOAT32, rotation.size(0),
        reinterpret_cast<uint8_t*>(rotation.data_ptr<float>()),
        tinyply::Type::INVALID, 0);


    torch::Tensor ids_int32 = ids.to(torch::kInt32);
    result_file.add_properties_to_element(
        "vertex", {"id"},
        tinyply::Type::INT32, ids_int32.size(0),
        reinterpret_cast<uint8_t*>(ids_int32.data_ptr<int>()),
        tinyply::Type::INVALID, 0);


    result_file.write(outstream_binary, true);

    fb_binary.close();
}


void GaussianModel::trainingSetup()
{
    this->sparse_optimizer_.reset(new SparseGaussianAdam(Tensor_vec_xyz_, 0.0, 1e-15));
    sparse_optimizer_->param_groups()[0].options().set_lr(position_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_feature_dc_);
    sparse_optimizer_->param_groups()[1].options().set_lr(feature_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_feature_rest_);
    sparse_optimizer_->param_groups()[2].options().set_lr(feature_lr_ / 20.0);

    sparse_optimizer_->add_param_group(Tensor_vec_opacity_);
    sparse_optimizer_->param_groups()[3].options().set_lr(opacity_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_scaling_);
    sparse_optimizer_->param_groups()[4].options().set_lr(scaling_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_rotation_);
    sparse_optimizer_->param_groups()[5].options().set_lr(rotation_lr_);

    if (apply_exposure_)
    {
        this->exposure_optimizer_.reset(new torch::optim::Adam(Tensor_vec_exposure_, {}));
        exposure_optimizer_->param_groups()[0].options().set_lr(exposure_lr_);
    }


    int64_t N = this->xyz_.size(0);
    this->xyz_gradient_accum_ = torch::zeros({N, 2}, torch::kFloat32).cuda();
    this->denom_               = torch::zeros({N, 1}, torch::kFloat32).cuda();
    this->max_radii2D_         = torch::zeros({N},    torch::kFloat32).cuda();
    this->newborn_steps_left_  = torch::zeros({N}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
}


void GaussianModel::densificationPostfix(
    torch::Tensor& new_xyz,
    torch::Tensor& new_features_dc,
    torch::Tensor& new_features_rest,
    torch::Tensor& new_opacities,
    torch::Tensor& new_scaling,
    torch::Tensor& new_rotation,
    torch::Tensor& new_ids,
    int newborn_boost_steps)
{
    const int64_t old_N = this->xyz_.size(0);

    std::vector<torch::Tensor> optimizable_tensors(6);



    std::vector<torch::Tensor> tensors_dict =
    {
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation
    };



    auto& param_groups = this->sparse_optimizer_->param_groups();
    auto& optimizer_state = this->sparse_optimizer_->get_state();


    for (int group_idx = 0; group_idx < 6; ++group_idx)
    {

        auto& group = param_groups[group_idx];
        assert(group.params().size() == 1);
        auto& extension_tensor = tensors_dict[group_idx];
        auto& param = group.params()[0];


        auto old_param_impl = param.unsafeGetTensorImpl();

        param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();

        group.params()[0] = param;


        auto new_param_impl = param.unsafeGetTensorImpl();

        auto state_it = optimizer_state.find(old_param_impl);
        if (state_it != optimizer_state.end())
        {

            auto stored_state = state_it->second;

            stored_state.exp_avg = torch::cat({stored_state.exp_avg.clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0);
            stored_state.exp_avg_sq = torch::cat({stored_state.exp_avg_sq.clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0);

            optimizer_state.erase(state_it);

            optimizer_state[new_param_impl] = stored_state;
        }
        else
        {

            State new_state;
            new_state.step = 0;

            new_state.exp_avg = torch::zeros_like(param, torch::MemoryFormat::Preserve);
            new_state.exp_avg_sq = torch::zeros_like(param, torch::MemoryFormat::Preserve);
            new_state.initialized = true;


            optimizer_state[new_param_impl] = new_state;
        }


        optimizable_tensors[group_idx] = param;
    }



    this->xyz_ = optimizable_tensors[0];
    this->features_dc_ = optimizable_tensors[1];
    this->features_rest_ = optimizable_tensors[2];
    this->opacity_ = optimizable_tensors[3];
    this->scaling_ = optimizable_tensors[4];
    this->rotation_ = optimizable_tensors[5];


    GAUSSIAN_MODEL_TENSORS_TO_VEC


    this->global_ids_ = torch::cat({this->global_ids_, new_ids}, 0);


    {
        auto steps_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        if (!newborn_steps_left_.defined() || newborn_steps_left_.numel() == 0 || newborn_steps_left_.size(0) != old_N) {
            newborn_steps_left_ = torch::zeros({old_N}, steps_opts);
        }
        const int init_steps = std::max(0, newborn_boost_steps);
        const int64_t added = (int64_t)new_xyz.size(0);
        auto appended = torch::full({added}, init_steps, steps_opts);
        newborn_steps_left_ = torch::cat({newborn_steps_left_, appended}, 0);
    }


    int64_t added = (int64_t)new_xyz.size(0);
    if (xyz_gradient_accum_.defined() && xyz_gradient_accum_.size(0) > 0) {
            this->xyz_gradient_accum_ = torch::cat({
                this->xyz_gradient_accum_,
                torch::zeros({added, 2}, torch::kFloat32).cuda()
            }, 0);
            this->denom_ = torch::cat({
                this->denom_,
                torch::zeros({added, 1}, torch::kFloat32).cuda()
            }, 0);
        this->max_radii2D_ = torch::cat({
            this->max_radii2D_,
            torch::zeros({added}, torch::kFloat32).cuda()
        }, 0);
    }
}



 void GaussianModel::prune(torch::Tensor& keep_mask)
 {
     std::vector<torch::Tensor> optimizable_tensors(6);
     std::vector<torch::Tensor> tensors_dict = {
         this->xyz_,
         this->features_dc_,
         this->features_rest_,
         this->opacity_,
         this->scaling_,
         this->rotation_
     };

     auto& param_groups = this->sparse_optimizer_->param_groups();
     auto& optimizer_state = this->sparse_optimizer_->get_state();

     for (int group_idx = 0; group_idx < 6; ++group_idx)
     {
         auto& group = param_groups[group_idx];
         auto& param = group.params()[0];
         auto old_param_impl = param.unsafeGetTensorImpl();


         param = tensors_dict[group_idx].index({keep_mask, torch::indexing::Slice()}).detach().contiguous().requires_grad_();


         group.params()[0] = param;
         auto new_param_impl = param.unsafeGetTensorImpl();


         auto state_it = optimizer_state.find(old_param_impl);
         if (state_it != optimizer_state.end())
         {
             auto stored_state = state_it->second;

             stored_state.exp_avg = stored_state.exp_avg.index({keep_mask, torch::indexing::Slice()}).detach().contiguous();
             stored_state.exp_avg_sq = stored_state.exp_avg_sq.index({keep_mask, torch::indexing::Slice()}).detach().contiguous();

             optimizer_state.erase(state_it);
             optimizer_state[new_param_impl] = stored_state;
         }

         optimizable_tensors[group_idx] = param;
     }

     this->xyz_ = optimizable_tensors[0];
     this->features_dc_ = optimizable_tensors[1];
     this->features_rest_ = optimizable_tensors[2];
     this->opacity_ = optimizable_tensors[3];
     this->scaling_ = optimizable_tensors[4];
     this->rotation_ = optimizable_tensors[5];

    GAUSSIAN_MODEL_TENSORS_TO_VEC


     this->global_ids_ = this->global_ids_.index({keep_mask, torch::indexing::Slice()});


    if (xyz_gradient_accum_.defined() && xyz_gradient_accum_.size(0) == keep_mask.size(0)) {
        this->xyz_gradient_accum_ = this->xyz_gradient_accum_.index(
            {keep_mask, torch::indexing::Slice()});
        this->denom_ = this->denom_.index(
            {keep_mask, torch::indexing::Slice()});
        this->max_radii2D_ = this->max_radii2D_.index({keep_mask});
    }
    if (newborn_steps_left_.defined() && newborn_steps_left_.size(0) == keep_mask.size(0)) {
        newborn_steps_left_ = newborn_steps_left_.index({keep_mask});
    }
 }




// Back-project normalized screen-space directions into the world frame.
static torch::Tensor computeWorldDir(
    const torch::Tensor& grad_dir_2d,
    float fx, float fy,
    const Eigen::Matrix3d& R_cw)
{
    int M = (int)grad_dir_2d.size(0);
    auto R_cw_f = R_cw.cast<float>();

    auto R_wc = torch::tensor(
        {(float)R_cw_f(0,0), (float)R_cw_f(1,0), (float)R_cw_f(2,0),
         (float)R_cw_f(0,1), (float)R_cw_f(1,1), (float)R_cw_f(2,1),
         (float)R_cw_f(0,2), (float)R_cw_f(1,2), (float)R_cw_f(2,2)},
        torch::kFloat32).reshape({3, 3}).cuda();


    auto dir_cam = torch::zeros({M, 3}, torch::kFloat32).cuda();
    dir_cam.index_put_({torch::indexing::Slice(), 0},
        grad_dir_2d.index({torch::indexing::Slice(), 0}) / fx);
    dir_cam.index_put_({torch::indexing::Slice(), 1},
        grad_dir_2d.index({torch::indexing::Slice(), 1}) / fy);



    auto dir_cam_norm = torch::nn::functional::normalize(
        dir_cam, torch::nn::functional::NormalizeFuncOptions().dim(1));
    return torch::matmul(dir_cam_norm, R_wc.t());
}





void GaussianModel::addDensificationStats(
    const torch::Tensor& screenspace_points,
    const torch::Tensor& visibility_mask,
    const torch::Tensor& radii,
    const torch::Tensor& screenspace_points_local,
    const torch::Tensor& screenspace_points_mask)
{

    auto vis_f = visibility_mask;
    this->max_radii2D_.index_put_(
        {vis_f},
        torch::max(this->max_radii2D_.index({vis_f}),
                   radii.index({vis_f}).to(torch::kFloat32))
    );


    auto grad_src = screenspace_points.grad();
    if (!grad_src.defined() && screenspace_points_local.defined()) {
        grad_src = screenspace_points_local.grad();
        if (grad_src.defined() && screenspace_points_mask.defined()) {
            auto N_full = screenspace_points.size(0);
            auto grad_full = torch::zeros({N_full, 3}, grad_src.options());
            grad_full = grad_full.index_put({screenspace_points_mask}, grad_src, /*accumulate=*/false);
            grad_src = grad_full;
        }
    }

    if (grad_src.defined()) {
        auto grad_xy = grad_src
                         .index({torch::indexing::Slice(),
                                 torch::indexing::Slice(0, 2)})
                         .detach();
        this->xyz_gradient_accum_.index_put_(
            {vis_f},
            this->xyz_gradient_accum_.index({vis_f}) + grad_xy.index({vis_f})
        );
        this->denom_.index_put_(
            {vis_f},
            this->denom_.index({vis_f}) + 1.0f
        );
    }
}


static torch::Tensor buildChildOpacityFromParent(
    const torch::Tensor& parent_opacity_logit,
    double opacity_scale,
    double min_alpha)
{
    auto parent_alpha = torch::sigmoid(parent_opacity_logit.detach());
    const float safe_scale = std::max(0.0f, static_cast<float>(opacity_scale));
    const float safe_min_alpha = std::clamp(static_cast<float>(min_alpha), 1e-5f, 0.95f);
    auto child_alpha = torch::clamp(parent_alpha * safe_scale, 1e-5f, 0.95f);

    auto floor_alpha = torch::full_like(child_alpha, safe_min_alpha);
    child_alpha = torch::where(parent_alpha >= floor_alpha,
                               torch::maximum(child_alpha, floor_alpha),
                               child_alpha);
    return general_utils::inverse_sigmoid(child_alpha);
}




// Clone compact Gaussians along the dominant image-space gradient direction.
void GaussianModel::densifyAndClone(
    torch::Tensor& grads,
    double grad_threshold,
    double extent,
    const std::shared_ptr<Camera>& cam)
{
    auto grad_norms = torch::norm(grads, 2, /*dim=*/1);
    auto selected   = grad_norms >= (float)grad_threshold;
    auto max_scale  = std::get<0>(torch::max(this->getScaling(), 1));
    selected = torch::logical_and(selected,
                                  max_scale <= (float)(percent_dense_ * extent));

    int num_sel = selected.sum().item<int>();
    if (num_sel == 0) return;


    auto sel_grads   = grads.index({selected});
    auto sel_norms   = grad_norms.index({selected}).unsqueeze(1).clamp_min(1e-8f);
    auto grad_dir_2d = sel_grads / sel_norms;

    auto dir_world = computeWorldDir(grad_dir_2d, cam->fx_, cam->fy_, cam->R_cw_);
    auto alpha = (float)densify_alpha_ *
                 max_scale.index({selected}).unsqueeze(1).detach();


    auto new_xyz         = this->xyz_.index({selected}).detach() + alpha * dir_world;
    auto new_features_dc = this->features_dc_.index({selected}).detach();
    auto new_features_rest = this->features_rest_.index({selected}).detach();
    auto new_opacity = buildChildOpacityFromParent(
        this->opacity_.index({selected}),
        densify_new_opacity_scale_,
        densify_new_opacity_min_);
    auto new_scaling     = this->scaling_.index({selected}).detach();
    auto new_rotation    = this->rotation_.index({selected}).detach();

    int64_t num_new = (int64_t)num_sel;
    auto new_ids = torch::arange(max_id_, max_id_ + num_new,
                                  torch::kInt64).view({num_new, 1}).cuda();
    max_id_ += num_new;

    this->densificationPostfix(new_xyz, new_features_dc, new_features_rest,
                               new_opacity, new_scaling, new_rotation, new_ids,
                               densify_newborn_boost_steps_);


    grads = torch::cat({grads, torch::zeros({num_new, 2}, grads.options())}, 0);
}






// Split large Gaussians by moving the parent and spawning one mirrored child.
void GaussianModel::densifyAndSplit(
    torch::Tensor& grads,
    double grad_threshold,
    double extent,
    const std::shared_ptr<Camera>& cam)
{
    auto grad_norms = torch::norm(grads, 2, 1);
    auto selected   = grad_norms >= (float)grad_threshold;
    auto max_scale  = std::get<0>(torch::max(this->getScaling(), 1));
    selected = torch::logical_and(selected,
                                  max_scale > (float)(percent_dense_ * extent));

    int num_sel = selected.sum().item<int>();
    if (num_sel == 0) return;


    auto sel_grads   = grads.index({selected});
    auto sel_norms   = grad_norms.index({selected}).unsqueeze(1).clamp_min(1e-8f);
    auto grad_dir_2d = sel_grads / sel_norms;

    auto dir_world = computeWorldDir(grad_dir_2d, cam->fx_, cam->fy_, cam->R_cw_);
    auto alpha = (float)densify_alpha_ *
                 max_scale.index({selected}).unsqueeze(1).detach();


    {
        torch::NoGradGuard no_grad;


        auto base_xyz = this->xyz_.index({selected}).detach();
        this->xyz_.index_put_({selected},
            (base_xyz + alpha * dir_world).detach());


        auto new_scale_log = torch::log(
            this->getScaling().index({selected}).detach() / (0.8f * 2.0f));
        this->scaling_.index_put_({selected}, new_scale_log.detach());
    }



    auto new_xyz = this->xyz_.index({selected}).detach()
                   - 2.0f * alpha * dir_world.detach();


    auto new_scaling_val  = this->scaling_.index({selected}).detach();
    auto new_rotation     = this->rotation_.index({selected}).detach();
    auto new_features_dc  = this->features_dc_.index({selected}).detach();
    auto new_features_rest= this->features_rest_.index({selected}).detach();
    auto new_opacity = buildChildOpacityFromParent(
        this->opacity_.index({selected}),
        densify_new_opacity_scale_,
        densify_new_opacity_min_);

    int64_t num_new = (int64_t)num_sel;
    auto new_ids = torch::arange(max_id_, max_id_ + num_new,
                                  torch::kInt64).view({num_new, 1}).cuda();
    max_id_ += num_new;


    this->densificationPostfix(new_xyz, new_features_dc, new_features_rest,
                               new_opacity, new_scaling_val, new_rotation, new_ids,
                               densify_newborn_boost_steps_);


    grads = torch::cat({grads, torch::zeros({num_new, 2}, grads.options())}, 0);

}




// Run densification and then prune Gaussians that fall below the opacity floor.
void GaussianModel::densifyAndPrune(
    double max_grad, double min_opacity, double extent,
    int max_screen_size,
    const std::shared_ptr<Camera>& cam)
{

    auto grads = this->xyz_gradient_accum_ /
                 this->denom_.clamp_min(1.0f);
    grads = torch::nan_to_num(grads, 0.0, 0.0, 0.0);
    int64_t old_N = this->opacity_.size(0);

    std::cout << "grads statics: min " << grads.min().item<double>()
              << ", max " << grads.max().item<double>()
              << ", mean " << grads.mean().item<double>() << std::endl;


    this->densifyAndClone(grads, max_grad, extent, cam);
    this->densifyAndSplit(grads, max_grad, extent, cam);





    auto prune_mask = (this->getOpacity() < (float)min_opacity).squeeze();
    auto num_opacity = prune_mask.sum().item<int>();
    std::cout << num_opacity << " gaussians will be prune for too low opacity" <<std::endl;






    auto keep_mask = torch::logical_not(prune_mask);
    this->prune(keep_mask);


    int64_t new_N = this->xyz_.size(0);
    this->xyz_gradient_accum_ = torch::zeros({new_N, 2}, torch::kFloat32).cuda();
    this->denom_               = torch::zeros({new_N, 1}, torch::kFloat32).cuda();
    this->max_radii2D_         = torch::zeros({new_N},    torch::kFloat32).cuda();

    c10::cuda::CUDACachingAllocator::emptyCache();
    std::cout << "\033[1;36m[Densify] add GS=" << new_N - old_N << "\033[0m" << std::endl;
}

// Extend the map from the latest keyframe using depth-completed point proposals.
void extend(const std::shared_ptr<Dataset>& dataset, std::shared_ptr<GaussianModel>& pc,
            std::shared_ptr<SPNetWrapper> spnet,
            std::shared_ptr<DA3Wrapper> da3)
{

    torch::NoGradGuard no_grad;


    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();


    std::shared_ptr<Camera> viewpoint_cam = dataset->train_cameras_.back();

    int deg_2 = (pc->sh_degree_ + 1) * (pc->sh_degree_ + 1);
    float focal = static_cast<float>((viewpoint_cam->fx_ + viewpoint_cam->fy_) / 2.0);
    torch::Tensor rgb = viewpoint_cam->original_image_.cuda();
    int H = viewpoint_cam->image_height_;
    int W = viewpoint_cam->image_width_;


    torch::Tensor sparse_depth = process_lidar_projection(dataset, viewpoint_cam);


    DepthCompletionParams params;
    params.epsilon_1 = pc->spnet_epsilon_1_;
    params.epsilon_2 = pc->spnet_epsilon_2_;
    params.patch_size = pc->spnet_patch_size_;
    params.dilate_radius = pc->spnet_dilate_radius_;
    params.depth_grad_threshold = pc->spnet_depth_grad_threshold_;
    if (spnet) params.max_depth = spnet->get_max_depth();

    SPNetPointsResult sp_result;

    if (spnet != nullptr && spnet->is_loaded()) {

        sp_result = get_spnet_points(*spnet, rgb, sparse_depth, params, viewpoint_cam);

        if (sp_result.fused_points.size(0) == 0) {
            return;
        }
    } else {
        std::cerr << "SPNet not loaded, cannot generate new points." << std::endl;
        return;
    }


    torch::Tensor da3_normals_all;
    torch::Tensor da3_valid_mask_all;
    if (da3 != nullptr && da3->is_loaded()) {
        auto da3_guidance = build_da3_guidance(
            *da3, rgb, sparse_depth, viewpoint_cam,
            pc->da3_conf_threshold_, pc->da3_min_depth_, pc->da3_max_depth_, pc->da3_align_min_points_,
            pc->spnet_enforce_normal_face_camera_, pc->da3_save_depth_vis_, pc->da3_debug_dir_);
        std::tie(da3_normals_all, da3_valid_mask_all) =
            sample_da3_guidance_by_pixels(da3_guidance, sp_result.fused_pixels);
    } else {
        da3_normals_all = torch::zeros({sp_result.fused_points.size(0), 3},
                                       torch::TensorOptions().dtype(torch::kFloat32).device(sp_result.fused_points.device()));
        da3_valid_mask_all = torch::zeros({sp_result.fused_points.size(0), 1},
                                          torch::TensorOptions().dtype(torch::kFloat32).device(sp_result.fused_points.device()));
    }


    // Build a sparse spatial prior from image gradients and optional voxel thinning.
    double grad_threshold = pc->color_gradient_threshold_;
    torch::Tensor grad_mask;
    torch::Tensor pixel_grads;
    {

        auto gray_img = 0.299 * rgb.select(0, 0) + 0.587 * rgb.select(0, 1) + 0.114 * rgb.select(0, 2);
        auto padded_gray = torch::nn::functional::pad(
            gray_img.unsqueeze(0).unsqueeze(0),
            torch::nn::functional::PadFuncOptions({1, 1, 1, 1}).mode(torch::kReplicate)).squeeze(0).squeeze(0);

        auto grad_x = (padded_gray.index({torch::indexing::Slice(1, -1), torch::indexing::Slice(2, torch::indexing::None)}) -
                       padded_gray.index({torch::indexing::Slice(1, -1), torch::indexing::Slice(0, -2)})) / 2.0;
        auto grad_y = (padded_gray.index({torch::indexing::Slice(2, torch::indexing::None), torch::indexing::Slice(1, -1)}) -
                       padded_gray.index({torch::indexing::Slice(0, -2), torch::indexing::Slice(1, -1)})) / 2.0;
        auto grad_mag = torch::sqrt(grad_x * grad_x + grad_y * grad_y);

        auto px = sp_result.fused_pixels.index({torch::indexing::Slice(), 0}).clamp(0, W - 1).to(torch::kLong);
        auto py = sp_result.fused_pixels.index({torch::indexing::Slice(), 1}).clamp(0, H - 1).to(torch::kLong);
        pixel_grads = grad_mag.index({py, px});
    }

    if (grad_threshold > 0.0) {
        grad_mask = pixel_grads > grad_threshold;
    } else {
        grad_mask = torch::ones({sp_result.fused_points.size(0)}, torch::TensorOptions().dtype(torch::kBool).device(rgb.device()));
    }


    double voxel_size = pc->voxel_size_;
    torch::Tensor voxel_mask;
    if (voxel_size > 0.0) {
        auto points = sp_result.fused_points;
        auto voxel_coords = torch::floor(points / voxel_size).to(torch::kLong);

        torch::Tensor unique_voxels, inverse_indices, counts;
        std::tie(unique_voxels, inverse_indices, counts) = torch::unique_dim(voxel_coords, 0, true, true, true);



        auto N = points.size(0);
        auto num_voxels = unique_voxels.size(0);


        auto sort_res = torch::sort(inverse_indices);
        auto sorted_inv = std::get<0>(sort_res);
        auto sorted_idx = std::get<1>(sort_res);


        auto diff = torch::cat({torch::ones({1}, torch::TensorOptions().dtype(torch::kBool).device(rgb.device())),
                                sorted_inv.index({torch::indexing::Slice(1, torch::indexing::None)}) !=
                                sorted_inv.index({torch::indexing::Slice(0, -1)})}, 0);


        auto keep_indices = sorted_idx.index({diff});

        voxel_mask = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(rgb.device()));
        voxel_mask.index_put_({keep_indices}, true);
    } else {
        voxel_mask = torch::ones({sp_result.fused_points.size(0)}, torch::TensorOptions().dtype(torch::kBool).device(rgb.device()));
    }


    auto spatial_mask = torch::logical_or(grad_mask, voxel_mask);


    // Keep proposals that are under-explained by the current rendering.
    auto render_pkg = render_2d(viewpoint_cam, pc, bg, 1.0f, pc->if_tileCull_in_extend_);
    auto rendered_alpha = render_pkg.rendered_alpha.squeeze();

    double alpha_threshold_base = pc->alpha_threshold_;
    double error_threshold = pc->hiColorLoss_threshold_;
    double dark_color_threshold = pc->dark_color_threshold_;


    auto error_map = torch::abs(render_pkg.rendered_image - rgb).mean(0);


    auto pixels = sp_result.fused_pixels;


    auto x_coords = pixels.index({torch::indexing::Slice(), 0}).clamp(0, W - 1).to(torch::kLong);
    auto y_coords = pixels.index({torch::indexing::Slice(), 1}).clamp(0, H - 1).to(torch::kLong);

    auto color_r = rgb.select(0, 0).index({y_coords, x_coords});
    auto color_g = rgb.select(0, 1).index({y_coords, x_coords});
    auto color_b = rgb.select(0, 2).index({y_coords, x_coords});

    auto luminance = (color_r + color_g + color_b) / 3.0f * 255.0f;

    auto dynamic_alpha_threshold = torch::ones_like(luminance) * alpha_threshold_base;

    if (dark_color_threshold > 0.0) {
        auto dark_mask = luminance < dark_color_threshold;
        dynamic_alpha_threshold = torch::where(dark_mask, dynamic_alpha_threshold * 0.7f, dynamic_alpha_threshold);
    }


    auto is_transparent = rendered_alpha.index({y_coords, x_coords}) < dynamic_alpha_threshold;


    auto has_high_error = error_map.index({y_coords, x_coords}) > error_threshold;


    auto rendering_mask = torch::logical_or(is_transparent, has_high_error);
    auto regress_mask = pc->if_full_regress_
        ? rendering_mask
        : torch::logical_and(rendering_mask, spatial_mask);

    bool use_regress = pc->use_Gaussian_regress_ && pc->feature_extractor_ && pc->feature_extractor_->is_loaded()
        && pc->gaussian_regressor_ && pc->gaussian_regressor_->is_loaded();

    int num_regress_new = regress_mask.sum().item<int>();


    // The optional traditional branch only consumes points rejected by the main branch.
    bool enable_traditional_extra = use_regress && pc->enable_regress_high_grad_traditional_init_;
    if (enable_traditional_extra && pc->generate_dataset_) {
        std::cout << "[TraditionalExtra] 已开启，但当前为数据集生成模式。为避免污染回归数据集，本功能已跳过。" << std::endl;
        enable_traditional_extra = false;
    }
    if (enable_traditional_extra && voxel_size <= 0.0) {
        std::cout << "[TraditionalExtra] 已开启，但当前未启用体素滤波。本功能自动不生效。" << std::endl;
        enable_traditional_extra = false;
    }

    torch::Tensor traditional_extra_mask = torch::zeros_like(rendering_mask);
    int traditional_candidate_num = 0;
    int traditional_extra_num = 0;
    double traditional_grad_threshold = 0.0;
    if (enable_traditional_extra) {

        auto traditional_candidate_mask = torch::logical_and(
            rendering_mask,
            torch::logical_and(torch::logical_not(regress_mask), torch::logical_not(voxel_mask)));
        traditional_candidate_num = traditional_candidate_mask.sum().item<int>();

        if (traditional_candidate_num > 0) {
            auto candidate_grads = pixel_grads.index({traditional_candidate_mask});
            double grad_min = candidate_grads.min().item<double>();
            double grad_max = candidate_grads.max().item<double>();
            double keep_ratio = std::clamp(pc->regress_high_grad_keep_ratio_, 0.0, 1.0);


            traditional_grad_threshold = grad_min + (1.0 - keep_ratio) * (grad_max - grad_min);
            traditional_extra_mask = torch::logical_and(
                traditional_candidate_mask,
                pixel_grads >= traditional_grad_threshold);
            traditional_extra_num = traditional_extra_mask.sum().item<int>();
        }

        std::cout << "[TraditionalExtra] candidate=" << traditional_candidate_num
                  << ", dynamic_grad_threshold=" << traditional_grad_threshold
                  << ", selected=" << traditional_extra_num << std::endl;
    }

    int total_num = num_regress_new + traditional_extra_num;
    if (total_num == 0) return;


    // Materialize the selected candidate sets for initialization.
    auto new_points = sp_result.fused_points.index({regress_mask});
    auto new_colors_rgb = sp_result.fused_colors.index({regress_mask});
    auto new_depths = sp_result.fused_depths.index({regress_mask});
    auto new_normals = da3_normals_all.index({regress_mask});
    auto new_pixels = sp_result.fused_pixels.index({regress_mask});
    auto new_valid_normal_mask = da3_valid_mask_all.index({regress_mask});
    auto new_fused_color = RGB2SH(new_colors_rgb);


    auto tensor_opts_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto traditional_points = torch::empty({0, 3}, tensor_opts_float);
    auto traditional_colors_rgb = torch::empty({0, 3}, tensor_opts_float);
    auto traditional_depths = torch::empty({0, 1}, tensor_opts_float);
    auto traditional_normals = torch::empty({0, 3}, tensor_opts_float);
    auto traditional_pixels = torch::empty({0, 2}, tensor_opts_float);
    auto traditional_valid_normal_mask = torch::empty({0, 1}, tensor_opts_float);
    if (traditional_extra_num > 0) {
        traditional_points = sp_result.fused_points.index({traditional_extra_mask});
        traditional_colors_rgb = sp_result.fused_colors.index({traditional_extra_mask});
        traditional_depths = sp_result.fused_depths.index({traditional_extra_mask});
        traditional_normals = da3_normals_all.index({traditional_extra_mask});
        traditional_pixels = sp_result.fused_pixels.index({traditional_extra_mask});
        traditional_valid_normal_mask = da3_valid_mask_all.index({traditional_extra_mask});
    }

    if (pc->extend_debug_) {
        std::string debug_dir = pc->dataset_path_;
        if (debug_dir.empty()) debug_dir = "/tmp";
        debug_dir += "/extend_debug";
        if (!fs::exists(debug_dir)) {
            fs::create_directories(debug_dir);
        }
        fs::create_directories(debug_dir + "/render");
        fs::create_directories(debug_dir + "/alpha");
        fs::create_directories(debug_dir + "/marked");
    }



    torch::Tensor main_fused_point_cloud = torch::empty({0, 3}, tensor_opts_float);
    torch::Tensor main_features_dc = torch::empty({0, 1, 3}, tensor_opts_float);
    torch::Tensor main_features_rest = torch::empty({0, deg_2 - 1, 3}, tensor_opts_float);
    torch::Tensor main_scales = torch::empty({0, 2}, tensor_opts_float);
    torch::Tensor main_opacities = torch::empty({0, 1}, tensor_opts_float);
    torch::Tensor main_rots = torch::empty({0, 4}, tensor_opts_float);

    torch::Tensor extra_fused_point_cloud = torch::empty({0, 3}, tensor_opts_float);
    torch::Tensor extra_features_dc = torch::empty({0, 1, 3}, tensor_opts_float);
    torch::Tensor extra_features_rest = torch::empty({0, deg_2 - 1, 3}, tensor_opts_float);
    torch::Tensor extra_scales = torch::empty({0, 2}, tensor_opts_float);
    torch::Tensor extra_opacities = torch::empty({0, 1}, tensor_opts_float);
    torch::Tensor extra_rots = torch::empty({0, 4}, tensor_opts_float);

    if (use_regress) {
        if (num_regress_new > 0) {
            RegressionInput rg_input;
            rg_input.success = false;


            rg_input = prepare_rg_input(pc.get(), dataset, viewpoint_cam,
                                        new_points, new_depths, new_valid_normal_mask,
                                        new_pixels, new_normals, new_fused_color, torch::Tensor());

            if (rg_input.success) {
                auto res = regress_gaussians(pc.get(), viewpoint_cam, rg_input, focal);
                if (res.success) {
                    auto kept_new_points = filter_tensor_by_keep_mask(new_points, res.keep_mask);
                    auto kept_new_fused_color = filter_tensor_by_keep_mask(new_fused_color, res.keep_mask);
                    auto kept_new_depths = filter_tensor_by_keep_mask(new_depths, res.keep_mask);
                    auto kept_new_normals = filter_tensor_by_keep_mask(new_normals, res.keep_mask);
                    auto kept_new_pixels = filter_tensor_by_keep_mask(new_pixels, res.keep_mask);
                    auto kept_new_valid_normal_mask = filter_tensor_by_keep_mask(new_valid_normal_mask, res.keep_mask);
                    const int64_t kept_regress_num = res.kept_count;

                    main_fused_point_cloud = kept_new_points;
                    main_scales = res.scales;
                    main_opacities = res.opacities;
                    main_features_dc = pc->apply_regressed_sh_dc_
                        ? res.features_dc.transpose(1, 2).contiguous()
                        : kept_new_fused_color.unsqueeze(1).contiguous();
                    main_features_rest = pc->apply_regressed_sh_rest_
                        ? res.features_rest.transpose(1, 2).contiguous()
                        : torch::zeros({kept_regress_num, deg_2 - 1, 3}, kept_new_fused_color.options());

                    if (pc->apply_regressed_rotation_) {
                        main_rots = res.rots;
                    } else {
                        main_rots = compute_rotation_from_normals(
                            kept_new_normals,
                            kept_new_valid_normal_mask,
                            kept_new_pixels,
                            kept_new_depths,
                            viewpoint_cam);
                    }

                    if (pc->apply_regressed_position_ && res.positions.defined()) {
                        main_fused_point_cloud = res.positions;
                    }
                    new_points = kept_new_points;
                    new_fused_color = kept_new_fused_color;
                    new_depths = kept_new_depths;
                    new_normals = kept_new_normals;
                    new_pixels = kept_new_pixels;
                    new_valid_normal_mask = kept_new_valid_normal_mask;
                    std::cout << "[MLP] Regressed " << kept_regress_num << " Gaussians." << std::endl;
                }
                else {
                    std::cerr << "[GaussianModel] Regression failed!" << std::endl;
                    return;
                }
            }
            else {
                std::cerr << "[GaussianModel] Regression input prepare failed!" << std::endl;
                return;
            }
        }
        if (traditional_extra_num > 0) {
            auto extra_fused_color = RGB2SH(traditional_colors_rgb);
            torch::Tensor extra_features = torch::zeros({traditional_extra_num, 3, deg_2}, torch::kFloat32).cuda();
            extra_features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}, extra_fused_color);

            extra_features_dc = extra_features.index({
                torch::indexing::Slice(),
                torch::indexing::Slice(),
                torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous();
            extra_features_rest = extra_features.index({
                torch::indexing::Slice(),
                torch::indexing::Slice(),
                torch::indexing::Slice(1, deg_2)}).transpose(1, 2).contiguous();

            extra_scales = torch::log(pc->scaling_scale_ * traditional_depths / focal);
            extra_scales = torch::clamp(extra_scales, std::log(1e-4), std::log(pc->clamp_scale_));
            extra_scales = extra_scales.repeat({1, 2});

            extra_opacities = general_utils::inverse_sigmoid(
                0.1f * torch::ones({traditional_extra_num, 1}, torch::kFloat32).cuda());

            extra_rots = compute_rotation_from_normals(
                traditional_normals,
                traditional_valid_normal_mask,
                traditional_pixels,
                traditional_depths,
                viewpoint_cam);
            extra_fused_point_cloud = traditional_points;

            std::cout << "[TraditionalExtra] Initialized " << traditional_extra_num
                      << " Gaussians with traditional init." << std::endl;
        }
    } else {



        torch::Tensor features = torch::zeros({num_regress_new, 3, deg_2}, torch::kFloat32).cuda();
        features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}, new_fused_color);

        main_features_dc = features.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(),
            torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous();
        main_features_rest = features.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(),
            torch::indexing::Slice(1, deg_2)}).transpose(1, 2).contiguous();

        main_scales = torch::log(pc->scaling_scale_ * new_depths / focal);
        main_scales = torch::clamp(main_scales, std::log(1e-4), std::log(pc->clamp_scale_));
        main_scales = main_scales.repeat({1, 2});

        main_opacities = general_utils::inverse_sigmoid(0.1f * torch::ones({num_regress_new, 1}, torch::kFloat32).cuda());

        main_rots = compute_rotation_from_normals(
            new_normals,
            new_valid_normal_mask,
            new_pixels,
            new_depths,
            viewpoint_cam);
        main_fused_point_cloud = new_points;
    }

    torch::Tensor fused_point_cloud;
    torch::Tensor features_dc;
    torch::Tensor features_rest;
    torch::Tensor scales;
    torch::Tensor opacities;
    torch::Tensor rots;

    if (main_fused_point_cloud.size(0) > 0 && extra_fused_point_cloud.size(0) > 0) {
        fused_point_cloud = torch::cat({main_fused_point_cloud, extra_fused_point_cloud}, 0);
        features_dc = torch::cat({main_features_dc, extra_features_dc}, 0);
        features_rest = torch::cat({main_features_rest, extra_features_rest}, 0);
        scales = torch::cat({main_scales, extra_scales}, 0);
        opacities = torch::cat({main_opacities, extra_opacities}, 0);
        rots = torch::cat({main_rots, extra_rots}, 0);
    } else if (main_fused_point_cloud.size(0) > 0) {
        fused_point_cloud = main_fused_point_cloud;
        features_dc = main_features_dc;
        features_rest = main_features_rest;
        scales = main_scales;
        opacities = main_opacities;
        rots = main_rots;
    } else {
        fused_point_cloud = extra_fused_point_cloud;
        features_dc = extra_features_dc;
        features_rest = extra_features_rest;
        scales = extra_scales;
        opacities = extra_opacities;
        rots = extra_rots;
    }

    if  (pc->generate_dataset_){
        viewpoint_cam->raw_points_ = new_points.cpu();
        viewpoint_cam->raw_depths_ = new_depths.cpu();
        viewpoint_cam->raw_normals_ = new_normals.cpu();
        viewpoint_cam->raw_pixels_ = new_pixels.cpu();
        viewpoint_cam->raw_valid_mask_ = new_valid_normal_mask.cpu();
        viewpoint_cam->raw_colors_ = new_fused_color.cpu();
        viewpoint_cam->raw_focal_ = focal;
    }

    const int64_t total_inserted = fused_point_cloud.size(0);
    if (total_inserted == 0) {
        viewpoint_cam->added_ids_.clear();

        int current_idx = dataset->train_cameras_.size() - 1;
        int limit_idx = current_idx - pc->slide_window_size_;
        if (limit_idx >= 0 && limit_idx < dataset->train_cameras_.size()) {
            auto& cam_to_clear = dataset->train_cameras_[limit_idx];
            if (cam_to_clear->feature_map_.defined()) {
                cam_to_clear->feature_map_ = torch::Tensor();
            }
        }

        dataset->pointcloud_.clear();
        dataset->pointcolor_.clear();
        dataset->pointdepth_.clear();

        std::cout << "[InsertSummary] regress=0, traditional_extra=0, total=0 (all regressed candidates were pruned)." << std::endl;
        return;
    }

    int64_t start_id = pc->max_id_;
    torch::Tensor new_ids = torch::arange(start_id, start_id + total_inserted, torch::kInt64).view({total_inserted, 1}).cuda();
    pc->max_id_ += total_inserted;

    auto ids_cpu = new_ids.cpu().contiguous();
    int64_t* ids_ptr = ids_cpu.data_ptr<int64_t>();
    viewpoint_cam->added_ids_.assign(ids_ptr, ids_ptr + total_inserted);

    pc->densificationPostfix(fused_point_cloud, features_dc, features_rest, opacities, scales, rots, new_ids);

    std::cout << "[InsertSummary] regress=" << main_fused_point_cloud.size(0)
              << ", traditional_extra=" << traditional_extra_num
              << ", total=" << fused_point_cloud.size(0) << std::endl;
    std::cout << std::fixed << std::setprecision(2)
            << "\033[1;32m Insert " << double(fused_point_cloud.size(0)) / 1000
            << "k GS" << ",\033[0m";

    int current_idx = dataset->train_cameras_.size() - 1;
    int limit_idx = current_idx - pc->slide_window_size_;
    if (limit_idx >= 0 && limit_idx < dataset->train_cameras_.size()) {
        auto& cam_to_clear = dataset->train_cameras_[limit_idx];

        if (cam_to_clear->feature_map_.defined()) {
            cam_to_clear->feature_map_ = torch::Tensor();
        }
    }

    dataset->pointcloud_.clear();
    dataset->pointcolor_.clear();
    dataset->pointdepth_.clear();
}


// Build the optimization list from recency, densification boost, and loss priority.
double optimize(const std::shared_ptr<Dataset>& dataset, std::shared_ptr<GaussianModel>& pc)
{

    pc->t_start_ = std::chrono::steady_clock::now();
    int updated_num = 0;
    int densify_executed = 0;
    std::vector<int> opt_list;
    int max_iters = 100;

    int train_camera_num = dataset->train_cameras_.size();
    int test_camera_num = dataset->test_cameras_.size();
    int total_camera_num = pc->generate_dataset_ ? train_camera_num + test_camera_num : train_camera_num;


    if (pc->keyframe_loss_.size() < total_camera_num) {
        pc->keyframe_loss_.resize(total_camera_num, 10000.0);
        pc->keyframe_train_times_.resize(total_camera_num, 0);
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    if (pc->last_densify_round_.size() < total_camera_num) {
        pc->last_densify_round_.resize(total_camera_num, -1000000000);
        pc->densify_selected_count_.resize(total_camera_num, 0);
        pc->neighbor_last_densify_train_stamp_.resize(total_camera_num, 0);
        pc->neighbor_last_densify_round_.resize(total_camera_num, -1000000000);
    }
    pc->optimize_round_ += 1;


    while (!pc->recent_densify_centers_.empty() &&
           pc->recent_densify_centers_.front().second < pc->optimize_round_) {
        pc->recent_densify_centers_.pop_front();
    }

    struct DensifyCandidate {
        int idx;
        double score;
    };
    std::vector<DensifyCandidate> densify_candidates;
    densify_candidates.reserve(train_camera_num);
    int max_staleness = 1;
    int max_train_gap = 1;

    const int min_train_gap = std::max(1, pc->densify_min_train_after_covis_);
    for (int i = 0; i < train_camera_num; ++i) {

        if (pc->keyframe_train_times_[i] < pc->densify_from_train_times_) continue;
        if (pc->densification_interval_ <= 0) continue;
        if (pc->keyframe_train_times_[i] % pc->densification_interval_ != 0) continue;

        const int train_gap = pc->keyframe_train_times_[i] - pc->neighbor_last_densify_train_stamp_[i];
        if (train_gap < min_train_gap) continue;
        int staleness = pc->optimize_round_ - pc->last_densify_round_[i];
        max_train_gap = std::max(max_train_gap, train_gap);
        max_staleness = std::max(max_staleness, staleness);
    }
    for (int i = 0; i < train_camera_num; ++i) {
        if (pc->keyframe_train_times_[i] < pc->densify_from_train_times_) continue;
        if (pc->densification_interval_ <= 0) continue;
        if (pc->keyframe_train_times_[i] % pc->densification_interval_ != 0) continue;
        const int train_gap = pc->keyframe_train_times_[i] - pc->neighbor_last_densify_train_stamp_[i];
        if (train_gap < min_train_gap) continue;

        const double stale_norm = static_cast<double>(pc->optimize_round_ - pc->last_densify_round_[i]) /
                                  static_cast<double>(max_staleness);
        const double train_gap_norm = static_cast<double>(train_gap) / static_cast<double>(max_train_gap);


        const double score = uni01(gen) + 0.5 * stale_norm - 0.1 * pc->densify_selected_count_[i]
                           + pc->densify_train_gate_alpha_ * train_gap_norm;
        densify_candidates.push_back({i, score});
    }
    std::sort(densify_candidates.begin(), densify_candidates.end(),
              [](const DensifyCandidate& a, const DensifyCandidate& b) {
                  return a.score > b.score;
              });


    const int min_gap = std::max(1, pc->densify_index_gap_);
    const int max_densify = std::max(0, pc->densify_max_per_round_);
    std::unordered_set<int> selected_densify_set;
    std::vector<int> selected_densify_list;
    for (const auto& c : densify_candidates) {
        bool conflict = false;
        for (int chosen_idx : selected_densify_list) {
            if (std::abs(c.idx - chosen_idx) < min_gap) {
                conflict = true;
                break;
            }
        }
        if (conflict) continue;
        selected_densify_set.insert(c.idx);
        selected_densify_list.push_back(c.idx);
        if ((int)selected_densify_list.size() >= max_densify) break;
    }
    if (!selected_densify_list.empty()) {
        std::cout << "[DensifySelect] round " << pc->optimize_round_ << ", selected idx:";
        for (int i : selected_densify_list) std::cout << " " << i;
        std::cout << " (gap>=" << min_gap << ")" << std::endl;
        std::cout << "[DensifySelect] train_gap:";
        for (int i : selected_densify_list) {
            int train_gap = pc->keyframe_train_times_[i] - pc->neighbor_last_densify_train_stamp_[i];
            std::cout << " [" << i << ":" << train_gap << "]";
        }
        std::cout << " (min_required=" << min_train_gap << ")" << std::endl;
    }



    int start_idx_p1 = std::max(0, train_camera_num - pc->slide_window_size_);
    std::vector<int> p1_candidates;
    p1_candidates.reserve(pc->slide_window_size_);
    std::vector<char> in_p1(train_camera_num, 0);
    for (int i = start_idx_p1; i < train_camera_num; ++i) {
        p1_candidates.push_back(i);
        in_p1[i] = 1;
    }

    std::vector<int> p_boost_forced;
    std::vector<int> p_boost_candidates;
    std::vector<char> in_boost(train_camera_num, 0);
    for (int idx : selected_densify_list) {
        if (idx < 0 || idx >= train_camera_num) continue;
        if (in_p1[idx] || in_boost[idx]) continue;
        in_boost[idx] = 1;
        p_boost_forced.push_back(idx);
    }
    const int boost_radius = std::max(0, pc->post_densify_window_radius_);
    for (const auto& item : pc->recent_densify_centers_) {
        const int center = item.first;
        const int left = std::max(0, center - boost_radius);
        const int right = std::min(train_camera_num - 1, center + boost_radius);
        for (int i = left; i <= right; ++i) {
            if (in_p1[i] || in_boost[i]) continue;
            in_boost[i] = 1;
            p_boost_candidates.push_back(i);
        }
    }
    std::shuffle(p_boost_forced.begin(), p_boost_forced.end(), gen);
    std::shuffle(p_boost_candidates.begin(), p_boost_candidates.end(), gen);


    std::vector<int> p2_candidates;
    std::vector<int> p3_candidates;
    for (int i = 0; i < train_camera_num; ++i) {
        if (in_p1[i] || in_boost[i]) continue;
        if (pc->keyframe_loss_[i] > pc->hiloss_threshold_) p2_candidates.push_back(i);
        else p3_candidates.push_back(i);
    }

    if (pc->generate_dataset_) {
        for (int i = train_camera_num; i < total_camera_num; ++i) {
            p3_candidates.push_back(i);
        }
    }
    std::shuffle(p2_candidates.begin(), p2_candidates.end(), gen);
    std::shuffle(p3_candidates.begin(), p3_candidates.end(), gen);


    auto append_with_budget = [&](const std::vector<int>& src, int budget = -1) {
        if ((int)opt_list.size() >= max_iters) return;
        int take = std::min((int)src.size(), max_iters - (int)opt_list.size());
        if (budget >= 0) take = std::min(take, budget);
        opt_list.insert(opt_list.end(), src.begin(), src.begin() + take);
    };

    append_with_budget(p1_candidates);
    append_with_budget(p_boost_forced);
    int remaining_boost_budget = std::max(0, pc->post_densify_boost_budget_) - (int)p_boost_forced.size();
    append_with_budget(p_boost_candidates, std::max(0, remaining_boost_budget));
    append_with_budget(p2_candidates);
    append_with_budget(p3_candidates);

    if (!selected_densify_list.empty()) {
        std::vector<char> in_opt(total_camera_num, 0);
        for (int idx : opt_list) in_opt[idx] = 1;
        int missing = 0;
        for (int idx : selected_densify_list) {
            if (idx < 0 || idx >= train_camera_num) continue;
            if (!in_opt[idx]) missing++;
        }
        if (missing > 0) {
            std::cout << "[DensifySelect] warning: " << missing
                      << " selected frames are not in opt_list this round." << std::endl;
        }
    }


    torch::cuda::synchronize();
    pc->t_end_ = std::chrono::steady_clock::now();
    pc->t_optlist_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();


    pc->t_start_ = std::chrono::steady_clock::now();
    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();
    torch::cuda::synchronize();
    pc->t_end_ = std::chrono::steady_clock::now();
    pc->t_tocuda_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();

    // Run render-loss-backprop-update for each selected camera.
    for (int idx : opt_list)
    {

        pc->t_start_ = std::chrono::steady_clock::now();
        const std::shared_ptr<Camera>& viewpoint_cam = (idx < train_camera_num)
            ? dataset->train_cameras_[idx]
            : dataset->test_cameras_[idx - train_camera_num];

        auto gt_image = viewpoint_cam->original_image_.to(torch::kCUDA, /*non_blocking=*/true);
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_tocuda_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();


        pc->t_start_ = std::chrono::steady_clock::now();
        auto render_pkg = render_2d(viewpoint_cam, pc, bg, 1.0f, false);
        auto rendered_image = render_pkg.rendered_image;
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_forward_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();


        if (idx < train_camera_num) {
            runTrainVisualEvalIfNeeded(dataset, pc, viewpoint_cam, idx);
        }

        if (pc->extend_debug_ && pc->keyframe_train_times_[idx] == 0) {
            std::string debug_dir = pc->dataset_path_;
            if (debug_dir.empty()) debug_dir = "/tmp";
            debug_dir += "/extend_debug";
            if (!fs::exists(debug_dir)) {
                fs::create_directories(debug_dir);
            }
            fs::create_directories(debug_dir + "/initial_render");


            torch::Tensor r_img_tensor = render_pkg.rendered_image.detach().cpu().permute({1, 2, 0}).contiguous();
            r_img_tensor = r_img_tensor.mul(255).clamp(0, 255).to(torch::kU8);
            cv::Mat r_mat(viewpoint_cam->image_height_, viewpoint_cam->image_width_, CV_8UC3, r_img_tensor.data_ptr<uint8_t>());
            cv::cvtColor(r_mat, r_mat, cv::COLOR_RGB2BGR);
            cv::imwrite(debug_dir + "/initial_render/" + viewpoint_cam->image_name_, r_mat);
        }

        pc->t_start_ = std::chrono::steady_clock::now();

        auto Ll1 = loss_utils::l1_loss(rendered_image, gt_image);

        float lambda_dssim = pc->lambda_dssim_;
        torch::Tensor ssim_value;
        torch::Tensor rendered_image_unsq = rendered_image.unsqueeze(0);
        torch::Tensor gt_image_unsq = gt_image.unsqueeze(0);
        ssim_value = loss_utils::fused_ssim(rendered_image_unsq, gt_image_unsq);

        auto loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value);


        if (pc->keyframe_train_times_[idx] > pc->train_times_threshold_) {

            auto dist_loss = pc->lambda_dist_ * loss_utils::distortion_loss(render_pkg.rendered_distortion);
            loss = loss + dist_loss;

        auto normal_loss = pc->lambda_normal_ * loss_utils::normal_consistency_loss(
            render_pkg.rendered_normal, render_pkg.surf_normal, render_pkg.rendered_alpha);
        loss = loss + normal_loss;
        }

        if(pc->keyframe_train_times_[idx] == 0){
            std::cout <<"\033[1;35m"<<"[Regressor Supervisor] frame idx: "<<idx<<",train times: "<<pc->keyframe_train_times_[idx]<<", loss: "<<loss.item<double>()<<"\033[0m"<<std::endl;
        }
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_forward_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();


        pc->t_start_ = std::chrono::steady_clock::now();
        loss.backward();
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_backward_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();


        // Boost only newborn visible points so the higher position LR decays by effective updates.
        if (pc->densify_newborn_pos_lr_scale_ > 1.0 &&
            pc->newborn_steps_left_.defined() &&
            pc->newborn_steps_left_.size(0) == pc->getXYZ().size(0))
        {
            torch::NoGradGuard no_grad;
            auto xyz_grad = pc->xyz_.grad();
            if (xyz_grad.defined()) {
                auto newborn_mask = pc->newborn_steps_left_ > 0;
                if (newborn_mask.any().item<bool>()) {
                    xyz_grad.index_put_(
                        {newborn_mask, torch::indexing::Slice()},
                        xyz_grad.index({newborn_mask, torch::indexing::Slice()}) *
                        static_cast<float>(pc->densify_newborn_pos_lr_scale_));
                }
            }
        }

        {
            torch::NoGradGuard no_grad;
            pc->addDensificationStats(
                render_pkg.screenspace_points,
                render_pkg.visibility_mask,
                render_pkg.radii,
                render_pkg.screenspace_points_local,
                render_pkg.screenspace_points_mask
            );
        }


        pc->t_start_ = std::chrono::steady_clock::now();
        auto visible = render_pkg.visibility_mask;
        updated_num += visible.sum().item<int>();

        pc->sparse_optimizer_->set_visibility_and_N(visible, pc->getXYZ().size(0));
        pc->sparse_optimizer_->step();
        pc->sparse_optimizer_->zero_grad(true);


        if (pc->apply_exposure_)
        {
            pc->exposure_optimizer_->step();
            pc->exposure_optimizer_->zero_grad(true);
        }


        if (pc->newborn_steps_left_.defined() && pc->newborn_steps_left_.size(0) == visible.size(0)) {
            torch::NoGradGuard no_grad;
            auto vis_mask = visible.to(torch::kBool);
            auto active_mask = (pc->newborn_steps_left_ > 0) & vis_mask;
            if (active_mask.any().item<bool>()) {
                pc->newborn_steps_left_.index_put_(
                    {active_mask},
                    pc->newborn_steps_left_.index({active_mask}) - 1);
            }
        }


        pc->keyframe_train_times_[idx]++;
        pc->keyframe_loss_[idx] = loss.item<double>();


        // Densification eligibility is decided globally before this loop.
        if (selected_densify_set.count(idx) > 0)
        {
            torch::NoGradGuard no_grad;
            densify_executed++;
            std::cout << "frame idx " << idx
                      << " (trained " << pc->keyframe_train_times_[idx]
                      << " times) is densifying" << std::endl;
            pc->densifyAndPrune(
                pc->densify_grad_threshold_,
                pc->opacity_cull_threshold_,
                pc->scene_extent_,
                500,
                viewpoint_cam
            );
            if (pc->extend_debug_) {
                std::string debug_dir = pc->dataset_path_;
                if (debug_dir.empty()) debug_dir = "/tmp";
                debug_dir += "/extend_debug";
                if (!fs::exists(debug_dir)) {
                    fs::create_directories(debug_dir);
                }
                std::string this_dir = debug_dir + "/frame" + std::to_string(idx);
                if (!fs::exists(this_dir)) {
                    fs::create_directories(this_dir);
                }


                torch::Tensor r_img_tensor = render_pkg.rendered_image.detach().cpu().permute({1, 2, 0}).contiguous();
                r_img_tensor = r_img_tensor.mul(255).clamp(0, 255).to(torch::kU8);
                cv::Mat r_mat(viewpoint_cam->image_height_, viewpoint_cam->image_width_, CV_8UC3, r_img_tensor.data_ptr<uint8_t>());
                cv::cvtColor(r_mat, r_mat, cv::COLOR_RGB2BGR);
                cv::imwrite(this_dir + "/" + viewpoint_cam->image_name_+ "/" + std::to_string(pc->keyframe_train_times_[idx]), r_mat);
            }

            pc->last_densify_round_[idx] = pc->optimize_round_;
            pc->densify_selected_count_[idx] += 1;


            const int covis_radius = std::max(0, pc->densify_covis_window_);
            const int left = std::max(0, idx - covis_radius);
            const int right = std::min(train_camera_num - 1, idx + covis_radius);
            for (int j = left; j <= right; ++j) {
                pc->neighbor_last_densify_train_stamp_[j] = pc->keyframe_train_times_[j];
                pc->neighbor_last_densify_round_[j] = pc->optimize_round_;
            }
            for (auto it = pc->recent_densify_centers_.begin(); it != pc->recent_densify_centers_.end();) {
                if (it->first == idx) it = pc->recent_densify_centers_.erase(it);
                else ++it;
            }

            const int keep_rounds = std::max(1, pc->post_densify_boost_rounds_);
            pc->recent_densify_centers_.emplace_back(idx, pc->optimize_round_ + keep_rounds - 1);
        }


        if (pc->keyframe_train_times_[idx] % 5 == 0 && pc->if_prune_) {

            torch::NoGradGuard no_grad;
            auto scales = pc->getScaling();
            auto max_vals = std::get<0>(torch::max(scales, -1));
            auto min_vals = std::get<0>(torch::min(scales, -1));
            auto ratio = max_vals / min_vals;


            auto keep_mask = (ratio <= pc->scale_ratio_threshold_);
            int n_prune = (pc->getXYZ().size(0) - keep_mask.sum().item<long>());

            pc->prune(keep_mask);
        }

        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_step_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
    }


    if (!selected_densify_list.empty()) {
        std::cout << "[DensifySelect] selected=" << selected_densify_list.size()
                  << ", executed=" << densify_executed << std::endl;
    }
    if (opt_list.empty()) return 0.0;
    return updated_num / opt_list.size();
}

void runTrainVisualEvalIfNeeded(const std::shared_ptr<Dataset>& dataset,
                                std::shared_ptr<GaussianModel>& pc,
                                const std::shared_ptr<Camera>& train_camera,
                                int train_camera_idx)
{
    if (!pc->enable_train_visual_eval_) return;
    if (train_camera_idx < 0) return;
    if (train_camera_idx >= static_cast<int>(dataset->train_cameras_.size())) return;
    if (pc->train_visual_eval_frame_ids_.empty()) return;


    // Track only requested training frames and save one result per train_times value.
    warnRequestedTestFrames(dataset, pc);

    if (pc->train_visual_eval_last_saved_times_.size() < dataset->train_cameras_.size()) {
        pc->train_visual_eval_last_saved_times_.resize(dataset->train_cameras_.size(), -1);
    }

    const int frame_id = parseFrameIdFromImageName(train_camera->image_name_);


    int save_every_k_times = pc->train_visual_eval_every_k_train_times_;
    if (save_every_k_times <= 0) {
        save_every_k_times = 1;
    }

    const int train_times = pc->keyframe_train_times_[train_camera_idx];
    if (train_times < 0) return;
    if (train_times % save_every_k_times != 0) return;


    if (pc->train_visual_eval_last_saved_times_[train_camera_idx] == train_times) return;

    std::string eval_root = pc->result_path_;
    if (eval_root.empty()) {
        eval_root = pc->dataset_path_;
    }
    if (eval_root.empty()) {
        eval_root = "/tmp";
    }

    const fs::path base_dir = resolveTrainVisualEvalOutputDir(eval_root, pc->train_visual_eval_output_dir_);
    const fs::path train_dir = base_dir / getImageStem(train_camera->image_name_);
    const std::string train_times_stem = formatTrainTimesStem(train_times);
    const fs::path render_path = train_dir / (train_times_stem + ".jpg");
    const fs::path metrics_path = train_dir / (train_times_stem + "_metrics.txt");

    fs::create_directories(train_dir);

    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();

    auto lpips_module = loadLpipsModuleCached(pc->lpips_path_);
    const bool has_lpips = (lpips_module != nullptr);


    // Re-render the current state of the selected training frame and log quality metrics.
    auto render_pkg = render_2d(train_camera, pc, bg, 1.0f, false);
    auto rendered_image = render_pkg.rendered_image.clamp(0, 1);
    auto gt_image = train_camera->original_image_.to(torch::kCUDA).clamp(0, 1);

    const double psnr = loss_utils::psnr(rendered_image, gt_image).mean().item<double>();
    const double ssim = loss_utils::ssim(rendered_image, gt_image).item<double>();

    double lpips = std::numeric_limits<double>::quiet_NaN();
    if (has_lpips) {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(rendered_image.unsqueeze(0));
        inputs.push_back(gt_image.unsqueeze(0));
        lpips = lpips_module->forward(inputs).toTensor().item<double>();
    }

    {
        std::ofstream ofs(metrics_path);
        if (ofs.is_open()) {
            ofs << "image_name: " << train_camera->image_name_ << "\n";
            ofs << "frame_id: " << frame_id << "\n";
            ofs << "train_times: " << train_times << "\n";
            ofs << "optimize_round: " << pc->optimize_round_ << "\n";
            ofs << "psnr: " << psnr << "\n";
            ofs << "ssim: " << ssim << "\n";
            ofs << "lpips: " << lpips << "\n";
        }
    }

    appendTrainVisualEvalHistory(train_dir, train_times, pc->optimize_round_, psnr, ssim, lpips);
    pc->train_visual_eval_last_saved_times_[train_camera_idx] = train_times;

    std::cout << std::fixed << std::setprecision(4)
              << "[TrainVisualEval] 保存 "
              << train_camera->image_name_
              << " 的训练过程结果, train_times=" << train_times
              << ", PSNR=" << psnr
              << ", SSIM=" << ssim
              << ", LPIPS=" << lpips
              << ", 输出文件=" << render_path.string()
              << std::endl;
}



// Evaluate train and test views with a consistent rendering setup.
void evaluateVisualQuality(const std::shared_ptr<Dataset>& dataset,
                            std::shared_ptr<GaussianModel>& pc,
                            const std::string& result_path,
                            const std::string& lpips_path,
                            const bool save_image)
{

    std::cout << "Evaluate Visual Quality"<<std::endl;
    std::cout << "Number of Final Gaussians: " << pc->getXYZ().size(0) << std::endl;
    std::cout << "[Eval] save_image=" << (save_image ? "true" : "false") << std::endl;



    if (!fs::exists(result_path)) {
        fs::create_directories(result_path);
    }

    std::string render_dir_path = result_path + "/render";
    if (fs::exists(render_dir_path)) fs::remove_all(render_dir_path);
    fs::create_directories(render_dir_path);

    std::string gt_dir_path = result_path + "/gt";
    if (fs::exists(gt_dir_path)) fs::remove_all(gt_dir_path);
    fs::create_directories(gt_dir_path);

    const fs::path visual_quality_root = fs::path(result_path) / "visual_quality";
    const fs::path train_metrics_csv_path = visual_quality_root / "train" / "frame_metrics.csv";
    const fs::path test_metrics_csv_path = visual_quality_root / "test" / "frame_metrics.csv";


    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();


    auto lpips_module = loadLpipsModuleCached(lpips_path);
    if (lpips_module == nullptr) {
        std::cerr << "[Eval] LPIPS 模型不可用，将在逐帧结果中写入 NaN。" << std::endl;
    }

    {
        const auto train_eval = evaluateCameraSplit(
            dataset->train_cameras_,
            "Train",
            pc,
            bg,
            lpips_module,
            save_image,
            render_dir_path,
            gt_dir_path);
        savePerFrameMetricsCsv(train_metrics_csv_path, train_eval.records);


        std::cout << std::fixed << std::setprecision(2) << "AUTO_TUNE_TRAIN_PSNR " << train_eval.mean_psnr << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "AUTO_TUNE_TRAIN_SSIM " << train_eval.mean_ssim << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "AUTO_TUNE_TRAIN_LPIPS " << train_eval.mean_lpips << std::endl;
        std::cout << "[Eval] Train 逐帧指标已保存到 " << train_metrics_csv_path << std::endl;
    }

    {
        const auto test_eval = evaluateCameraSplit(
            dataset->test_cameras_,
            "Test",
            pc,
            bg,
            lpips_module,
            save_image,
            render_dir_path,
            gt_dir_path);
        savePerFrameMetricsCsv(test_metrics_csv_path, test_eval.records);


        std::cout << std::fixed << std::setprecision(2) << "AUTO_TUNE_NOVEL_PSNR " << test_eval.mean_psnr << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "AUTO_TUNE_NOVEL_SSIM " << test_eval.mean_ssim << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "AUTO_TUNE_NOVEL_LPIPS " << test_eval.mean_lpips << std::endl;
        std::cout << "[Eval] Test 逐帧指标已保存到 " << test_metrics_csv_path << std::endl;
    }

    std::cout << "[Eval] 如需绘制 train/test 逐帧指标曲线，可执行: python3 src/Gaussian-LIC/scripts/plot_visual_quality_metrics.py --result_root "
              << result_path << std::endl;
}
