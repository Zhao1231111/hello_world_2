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

#include "mapping.h"
#include "gaussian.h"
#include "pose_optimizer.h"
#include "pose_eval_recorder.h"
#include "model/depth_densifier/spnet_wrapper.h"
#include "model/depth_densifier/da3_wrapper.h"

#include <atomic>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <streambuf>

class TeeBuf : public std::streambuf {
public:
    TeeBuf(std::streambuf* sb1, std::streambuf* sb2) : sb1(sb1), sb2(sb2) {}
protected:
    virtual int overflow(int c) override {
        if (c == EOF) return !EOF;
        int const r1 = sb1->sputc(c);
        int const r2 = sb2->sputc(c);
        return (r1 == EOF || r2 == EOF) ? EOF : c;
    }
    virtual int sync() override {
        int const r1 = sb1->pubsync();
        int const r2 = sb2->pubsync();
        return (r1 == 0 && r2 == 0) ? 0 : -1;
    }
private:
    std::streambuf *sb1, *sb2;
};

std::mutex m_buf;
std::condition_variable con;

std::queue<sensor_msgs::PointCloud2ConstPtr> point_buf;
std::queue<geometry_msgs::PoseStampedConstPtr> pose_buf;
std::queue<sensor_msgs::ImageConstPtr> image_buf;

std::atomic<bool> exit_flag(false);
std::atomic<double> last_point_time(0.0);
std::atomic<bool> gaussians_initialized(false);

void pointCallback(const sensor_msgs::PointCloud2ConstPtr& point_msg)
{
    m_buf.lock();
    point_buf.push(point_msg);
    last_point_time = ros::Time::now().toSec();
    m_buf.unlock();
}

void poseCallback(const geometry_msgs::PoseStampedConstPtr& pose_msg)
{
    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
}

void imageCallback(const sensor_msgs::ImageConstPtr& image_msg)
{
    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();
}

// Align point, pose, and image messages around the point-cloud timestamp.
bool getAlignedData(Frame& cur_frame)
{
    if (point_buf.empty() || pose_buf.empty() || image_buf.empty())
    {
        return false;
    }

    double frame_time = point_buf.front()->header.stamp.toSec();

    while (1)
    {
        if (pose_buf.front()->header.stamp.toSec() < frame_time - 0.01)
        {
            pose_buf.pop();
            if (pose_buf.empty())
            {
                return false;
            }
        }
        else break;
    }
    if (pose_buf.front()->header.stamp.toSec() > frame_time + 0.01)
    {
        point_buf.pop();
        return false;
    }

    while (1)
    {
        if (image_buf.front()->header.stamp.toSec() < frame_time - 0.01)
        {
            image_buf.pop();
            if (image_buf.empty())
            {
                return false;
            }
        }
        else break;
    }
    if (image_buf.front()->header.stamp.toSec() > frame_time + 0.01)
    {
        point_buf.pop();
        return false;
    }

    auto cur_point = point_buf.front();
    auto cur_pose = pose_buf.front();
    auto cur_image = image_buf.front();

    cur_frame.point_msg = cur_point;
    cur_frame.pose_msg = cur_pose;
    cur_frame.image_msg = cur_image;

    point_buf.pop();
    pose_buf.pop();
    image_buf.pop();

    return true;
}

void printStatistics(torch::Tensor vals)
{
    double min_val = vals.min().item<double>();
    double max_val = vals.max().item<double>();
    int bins = 10;

    auto hist = torch::histc(vals, bins, min_val, max_val);

    double step = (max_val - min_val) / bins;

    for (int i = 0; i < bins; ++i) {
        double bin_start = min_val + i * step;
        double bin_end = (i == bins - 1) ? max_val : (min_val + (i + 1) * step);

        int count = static_cast<int>(hist[i].item<float>());

        printf("[%8.4f, %8.4f]: %d\n", bin_start, bin_end, count);
    }
}

std::vector<int> parseFrameIdListParam(const std::string& text)
{
    std::string normalized = text;
    for (char& ch : normalized) {
        if (ch == ',' || ch == ';') ch = ' ';
    }

    std::stringstream ss(normalized);
    std::vector<int> frame_ids;
    int frame_id = 0;
    while (ss >> frame_id) {
        frame_ids.push_back(frame_id);
    }
    return frame_ids;
}


void mapping(const YAML::Node& node, const std::string& result_path, const std::string& lpips_path)
{
    torch::jit::setGraphExecutorOptimize(false);

    Params prm(node);
    prm.dataset_path_ = node["dataset_path"].as<std::string>();
    std::string ros_dataset_path;
    if (ros::param::get("~dataset_path", ros_dataset_path) && !ros_dataset_path.empty()) {
        prm.dataset_path_ = ros_dataset_path;
        std::cout << "\033[1;32m [Config] Overrode dataset_path from ROS launch param: " << ros_dataset_path << " \033[0m" << std::endl;
    }
    prm.generate_dataset_ = node["generate_dataset"] ? node["generate_dataset"].as<bool>() : false;
    prm.dataset_target_train_times = node["dataset_target_train_times"] ? node["dataset_target_train_times"].as<int>() : 100;
    bool ros_generate_dataset;
    if (ros::param::get("~generate_dataset", ros_generate_dataset)) {
        prm.generate_dataset_ = ros_generate_dataset;
        std::cout << "\033[1;32m [Config] Overrode generate_dataset from ROS launch param: " << (ros_generate_dataset ? "true" : "false") << " \033[0m" << std::endl;
    }
    int ros_dataset_target_train_times;
    if (ros::param::get("~dataset_target_train_times", ros_dataset_target_train_times)) {
        prm.dataset_target_train_times = ros_dataset_target_train_times;
        std::cout << "\033[1;32m [Config] Overrode dataset_target_train_times from ROS launch param: "
                  << ros_dataset_target_train_times << " \033[0m" << std::endl;
    }
    bool ros_use_Gaussian_regress_;
    if (ros::param::get("~use_Gaussian_regress", ros_use_Gaussian_regress_)) {
        prm.use_Gaussian_regress = ros_use_Gaussian_regress_;
        std::cout << "\033[1;32m [Config] Overrode use_Gaussian_regress from ROS launch param: " << (ros_use_Gaussian_regress_ ? "true" : "false") << " \033[0m" << std::endl;
    }
    double ros_opacity_prune_forRegress;
    if (ros::param::get("~opacity_prune_forRegress", ros_opacity_prune_forRegress)) {
        prm.opacity_prune_forRegress = ros_opacity_prune_forRegress;
        std::cout << "\033[1;32m [Config] Overrode opacity_prune_forRegress from ROS launch param: "
                  << ros_opacity_prune_forRegress << " \033[0m" << std::endl;
    }
    double ros_opacity_modifier;
    if (ros::param::get("~opacity_modifier", ros_opacity_modifier)) {
        prm.opacity_modifier = ros_opacity_modifier;
        std::cout << "\033[1;32m [Config] Overrode opacity_modifier from ROS launch param: " << ros_opacity_modifier << " \033[0m" << std::endl;
    }
    bool ros_enable_train_visual_eval;
    if (ros::param::get("~enable_train_visual_eval", ros_enable_train_visual_eval)) {
        prm.enable_train_visual_eval = ros_enable_train_visual_eval;
        std::cout << "\033[1;32m [Config] Overrode enable_train_visual_eval from ROS launch param: "
                  << (ros_enable_train_visual_eval ? "true" : "false") << " \033[0m" << std::endl;
    }
    int ros_train_visual_eval_every_k_train_times;
    if (ros::param::get("~train_visual_eval_every_k_train_times", ros_train_visual_eval_every_k_train_times)) {
        prm.train_visual_eval_every_k_train_times = ros_train_visual_eval_every_k_train_times;
        std::cout << "\033[1;32m [Config] Overrode train_visual_eval_every_k_train_times from ROS launch param: "
                  << ros_train_visual_eval_every_k_train_times << " \033[0m" << std::endl;
    }
    std::string ros_train_visual_eval_output_dir;
    if (ros::param::get("~train_visual_eval_output_dir", ros_train_visual_eval_output_dir) &&
        !ros_train_visual_eval_output_dir.empty()) {
        prm.train_visual_eval_output_dir = ros_train_visual_eval_output_dir;
        std::cout << "\033[1;32m [Config] Overrode train_visual_eval_output_dir from ROS launch param: "
                  << ros_train_visual_eval_output_dir << " \033[0m" << std::endl;
    }
    std::string ros_train_visual_eval_frame_ids;
    if (ros::param::get("~train_visual_eval_frame_ids", ros_train_visual_eval_frame_ids) &&
        !ros_train_visual_eval_frame_ids.empty()) {
        prm.train_visual_eval_frame_ids = parseFrameIdListParam(ros_train_visual_eval_frame_ids);
        std::cout << "\033[1;32m [Config] Overrode train_visual_eval_frame_ids from ROS launch param: "
                  << ros_train_visual_eval_frame_ids << " \033[0m" << std::endl;
    }
    std::shared_ptr<GaussianModel> gaussians = std::make_shared<GaussianModel>(prm);
    gaussians->result_path_ = result_path;
    gaussians->lpips_path_ = lpips_path;
    std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(prm);
    gaussians->setDA3DebugDir(result_path + "/da3_depth_vis");
    PoseEvalRecorder pose_eval_recorder(result_path);

    std::shared_ptr<SPNetWrapper> spnet = nullptr;
    if (node["spnet_model_path"]) {
        std::string spnet_path = node["spnet_model_path"].as<std::string>();
        float max_depth = node["spnet_max_depth"] ? node["spnet_max_depth"].as<float>() : 100.0f;
        spnet = std::make_shared<SPNetWrapper>(spnet_path, max_depth);
        if (spnet->is_loaded()) {
            std::cout << "\033[1;32m[Mapping] SPNet loaded for depth completion\033[0m" << std::endl;
        } else {
            spnet = nullptr;
        }
    }

    std::shared_ptr<DA3Wrapper> da3 = nullptr;
    if (node["da3_model_path"]) {
        std::string da3_path = node["da3_model_path"].as<std::string>();
        da3 = std::make_shared<DA3Wrapper>(da3_path);
        if (da3->is_loaded()) {
            std::cout << "\033[1;32m[Mapping] DA3 loaded for normal initialization\033[0m" << std::endl;
        } else {
            da3 = nullptr;
        }
    }

    bool generate_dataset = prm.generate_dataset_;

    std::chrono::steady_clock::time_point t_start, t_end;
    double total_mapping_time = 0;
    double total_adding_time = 0;
    double total_extending_time = 0;

    Frame cur_frame;
    while (!exit_flag)
    {
        m_buf.lock();
        bool align_flag = getAlignedData(cur_frame);
        m_buf.unlock();
        if (!align_flag) continue;

        t_start = std::chrono::steady_clock::now();
        dataset->addFrame(cur_frame);
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
        const bool is_current_keyframe = dataset->is_keyframe_current_;
        auto current_camera = is_current_keyframe ? dataset->train_cameras_.back() : dataset->test_cameras_.back();
        const int current_frame_idx = dataset->all_frame_num_ - 1;
        PoseEvalFrameStatus pose_eval_status;
        pose_eval_status.frame_idx = current_frame_idx;
        pose_eval_status.is_keyframe = is_current_keyframe;
        if (is_current_keyframe)
        {
            total_adding_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
            std::cout << "\033[1;33m     Cur Frame " << current_frame_idx << ",\033[0m";
        }

        if (!gaussians->is_init_)
        {
            // Skip non-keyframes until the map exists.
            if (!is_current_keyframe)
            {
                pose_eval_recorder.recordFrame(current_camera, pose_eval_status);
                continue;
            }

            gaussians->is_init_ = true;
            gaussians_initialized = true;
            gaussians->initialize(dataset, spnet, da3);
            gaussians->trainingSetup();
            pose_eval_recorder.recordFrame(current_camera, pose_eval_status);
        }
        else
        {
            // Run the same pose refinement on both keyframes and non-keyframes.
            const int current_keyframe_idx = static_cast<int>(dataset->train_cameras_.size()) - 1;
            if (gaussians->enable_pose_refinement_ &&
                current_keyframe_idx >= gaussians->pose_refine_start_frame_ &&
                !current_camera->pose_refined_ &&
                gaussians->getXYZ().size(0) > 100)
            {
                pose_eval_status.pose_refine_attempted = true;
                pose_opt::PoseFilterParams pose_params;
                pose_params.max_iterations = gaussians->pose_max_iterations_;
                pose_params.img_point_cov = gaussians->pose_img_point_cov_;
                pose_params.init_cov = gaussians->pose_init_cov_;
                pose_params.conv_thresh_rot = gaussians->pose_conv_thresh_rot_;
                pose_params.conv_thresh_pos = gaussians->pose_conv_thresh_pos_;
                pose_params.min_valid_pixels = gaussians->pose_min_valid_pixels_;
                pose_params.alpha_threshold = gaussians->pose_alpha_threshold_;
                pose_params.huber_delta = gaussians->pose_huber_delta_;
                pose_params.rerender_each_iter = gaussians->pose_rerender_each_iter_;
                pose_params.max_step_rot_deg = gaussians->pose_max_step_rot_deg_;
                pose_params.max_step_trans_cm = gaussians->pose_max_step_trans_cm_;
                pose_params.rmse_increase_tolerance_ratio = gaussians->pose_rmse_increase_tolerance_ratio_;
                pose_params.min_alpha_coverage_ratio = gaussians->pose_min_alpha_coverage_ratio_;
                pose_params.alpha_erode_radius = gaussians->pose_alpha_erode_radius_;

                auto pose_result = pose_opt::refineCurrentFramePose(current_camera, gaussians, pose_params);
                pose_eval_status.pose_refine_success = pose_result.success;
                pose_eval_status.valid_pixels = pose_result.valid_pixels;
                pose_eval_status.rmse = pose_result.rmse;
                pose_eval_status.alpha_coverage_ratio = pose_result.alpha_coverage_ratio;
                pose_eval_status.rejected_by_guard = pose_result.rejected_by_guard;
                pose_eval_status.step_was_clamped = pose_result.step_was_clamped;
                pose_eval_status.guard_reason = pose_result.guard_reason;
                if (pose_result.success)
                {
                    current_camera->setPoseFromCw(pose_result.R_cw, pose_result.t_cw);
                    current_camera->pose_cov_ = pose_result.post_cov;
                    current_camera->pose_refined_ = true;

                    const Eigen::Vector3d delta_t = pose_result.t_cw - current_camera->t_cw_pred_;
                    const Eigen::Matrix3d delta_R = pose_result.R_cw * current_camera->R_cw_pred_.transpose();
                    const Eigen::Vector3d delta_rot = pose_opt::Log(delta_R);
                    std::cout << "\033[1;36m[PoseOpt] "
                              << (is_current_keyframe ? "keyframe " : "frame ")
                              << current_frame_idx
                              << ": iters=" << pose_result.iterations
                              << ", valid_pixels=" << pose_result.valid_pixels
                              << ", rmse=" << pose_result.rmse
                              << ", alpha_cov=" << pose_result.alpha_coverage_ratio
                              << ", |delta_R|=" << delta_rot.norm() * 57.29577951308232 << " deg"
                              << ", |delta_t|=" << delta_t.norm() * 1000.0 << " mm"
                              << ", step_clamped=" << (pose_result.step_was_clamped ? 1 : 0)
                              << ", guard=" << (pose_result.guard_reason.empty() ? "none" : pose_result.guard_reason)
                              << "\033[0m" << std::endl;
                }
                else
                {
                    std::cout << "\033[1;33m[PoseOpt] "
                              << (is_current_keyframe ? "keyframe " : "frame ")
                              << current_frame_idx
                              << " skipped, valid_pixels=" << pose_result.valid_pixels
                              << ", rmse=" << pose_result.rmse
                              << ", alpha_cov=" << pose_result.alpha_coverage_ratio
                              << ", step_clamped=" << (pose_result.step_was_clamped ? 1 : 0)
                              << ", guard=" << (pose_result.guard_reason.empty() ? "none" : pose_result.guard_reason)
                              << "\033[0m" << std::endl;
                }
            }

            pose_eval_recorder.recordFrame(current_camera, pose_eval_status);

            if (is_current_keyframe)
            {
                t_start = std::chrono::steady_clock::now();
                extend(dataset, gaussians, spnet, da3);
                torch::cuda::synchronize();
                t_end = std::chrono::steady_clock::now();
                total_extending_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
            }
            else
            {
                // Non-keyframes only update their current pose estimate.
                continue;
            }
        }

        t_start = std::chrono::steady_clock::now();
        double updated_num = optimize(dataset, gaussians);
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
        total_mapping_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        std::cout << std::fixed << std::setprecision(2)
                  << "\033[1;36m Update " << updated_num / 10000
                  << "w GS per Iter \033[0m" << std::endl;
    }

    std::cout << "Runtime Statistics"<<std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Total Mapping Time: " << total_mapping_time << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Forward: " << gaussians->t_forward_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Backward: " << gaussians->t_backward_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Step: " << gaussians->t_step_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "CPU2GPU: " << gaussians->t_tocuda_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Total Adding Time: " << total_adding_time << "s" << std::endl;

    std::cout << std::fixed << std::setprecision(2) << "Total Extending Time: " << total_extending_time << "s" << std::endl;
    evaluateVisualQuality(dataset, gaussians, result_path, lpips_path, false);

    if (generate_dataset) {
        std::cout << "[Dataset] Saving Final Dataset..." << std::endl;
        evaluateVisualQuality(dataset, gaussians, result_path, lpips_path, true);
        gaussians->saveDataset(prm.dataset_path_, dataset);
    }
    else {
        evaluateVisualQuality(dataset, gaussians, result_path, lpips_path, true);
        gaussians->saveMap(result_path);
    }

    std::cout << "Gaussian-LIC Done!" << std::endl;
}

int main(int argc, char** argv)
{
    std::cout << "Gaussian-LIC Ready!\n\n\n";
    ros::init(argc, argv, "gaussianlic");
    ros::NodeHandle nh("~");
    ros::Rate loop_rate(1000);
    image_transport::ImageTransport it_(nh);

    ros::Subscriber sub_point = nh.subscribe("/points_for_gs", 10000, pointCallback);
    ros::Subscriber sub_pose = nh.subscribe("/pose_for_gs", 10000, poseCallback);
    image_transport::Subscriber image_sub = it_.subscribe("/image_for_gs", 10000, imageCallback);

    std::string config_path;
    nh.param<std::string>("config_path", config_path, "");
    YAML::Node config_node = YAML::LoadFile(config_path);
    std::string result_path;
    nh.param<std::string>("result_path", result_path, "");
    std::string lpips_path;
    nh.param<std::string>("lpips_path", lpips_path, "");
    std::string dataset_path;
    nh.param<std::string>("dataset_path", dataset_path, "");
    if (dataset_path.empty()) {
        dataset_path = config_node["dataset_path"].as<std::string>();
    }
    bool generate_dataset;
    if (nh.hasParam("generate_dataset")) {
        nh.getParam("generate_dataset", generate_dataset);
    }
    else {
        generate_dataset = config_node["generate_dataset"].as<bool>();
    }

    if (dataset_path.empty()) {
        std::cerr << "Error: dataset_path is empty!" << std::endl;
        return 1;
    }
    std::string log_file_path;
    if(generate_dataset)
    {
        log_file_path = dataset_path + "/gaussian_lic_log.txt";
    }
    else
    {
        log_file_path = result_path + "/gaussian_lic_log.txt";
    }

    if (!std::filesystem::exists(dataset_path)) {
        std::filesystem::create_directories(dataset_path);
    }
    if (!std::filesystem::exists(result_path)) {
        std::filesystem::create_directories(result_path);
    }

    std::ofstream log_file(log_file_path);
    TeeBuf* tee_buf = nullptr;
    std::streambuf* old_cout_buf = nullptr;

    if (log_file.is_open()) {
        tee_buf = new TeeBuf(std::cout.rdbuf(), log_file.rdbuf());
        old_cout_buf = std::cout.rdbuf(tee_buf);
        std::cout << "\n[Logging] Output will be saved to: " << log_file_path << std::endl;
    } else {
        std::cout << "\n[Warning] Could not open log file at: " << log_file_path << " (result_path might be empty or invalid)" << std::endl;
    }
    std::thread mapping_process(mapping, config_node, result_path, lpips_path);
    std::thread monitor_thread([](){
        while (!exit_flag)
        {
            double now = ros::Time::now().toSec();
            if (gaussians_initialized && (now - last_point_time > 5.0) && image_buf.empty())
            {
                exit_flag = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });

    ros::spin();

    mapping_process.join();
    monitor_thread.join();

    if (old_cout_buf) {
        std::cout.rdbuf(old_cout_buf);
        delete tee_buf;
    }
    return 0;
}
