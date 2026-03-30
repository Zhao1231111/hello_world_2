/**
 * @file depth_completion.h
 * @brief 深度补全流水线 (Gaussian-LIC2 Algorithm 1)
 * 
 * 实现 Gaussian-LIC2 论文描述的深度补全算法：
 * 1. 调用 SPNet 进行深度补全
 * 2. 失败检测 (epsilon_1 阈值)
 * 3. 边缘过滤 (深度梯度检查)
 * 4. 可选 LiDAR Mask 膨胀
 * 5. 30x30 Patch 采样
 * 6. 反投影生成补充点云
 */

#pragma once

#include <vector>
#include <utility>
#include <string>
#include <torch/torch.h>
#include "spnet_wrapper.h"
#include "da3_wrapper.h"

// Forward declaration
class Camera;

/**
 * @brief 深度补全结果结构体
 */
struct DepthCompletionResult
{
    bool success;                          ///< 补全是否成功
    torch::Tensor dense_depth;             ///< 稠密深度图 (H, W)
    torch::Tensor filtered_dense_depth;    ///< 过滤后的稠密深度图 (H, W)，用于法线/旋转初始化
    torch::Tensor filtered_mask;           ///< 经过过滤后的有效像素 mask (H, W)
    
    /// 补充点像素坐标列表 (u, v)
    std::vector<std::pair<int, int>> supplement_pixels;
    
    /// 补充点深度 (从 Dcf[A] 获取)
    std::vector<float> supplement_depths;
    
    /// 补充点颜色 (从 RGB[A] 获取)
    std::vector<std::array<float, 3>> supplement_colors;
};

/**
 * @brief 深度补全参数
 */
struct DepthCompletionParams
{
    float epsilon_1 = 0.3f;           ///< 失败检测阈值 (30% mean change)
    float epsilon_2 = 80.0f;          ///< 最大采样深度 (m)
    int patch_size = 30;              ///< patch 大小
    int dilate_radius = 0;            ///< LiDAR mask 膨胀半径 (默认=0 不膨胀)
    float depth_grad_threshold = 0.5f;///< 深度梯度阈值 (用于边缘过滤)
    float max_depth = 100.0f;         ///< 最大有效深度（室外）
};

/**
 * @brief 为关键帧执行深度补全
 * 
 * 实现 Gaussian-LIC2 论文 Algorithm 1 描述的深度补全流程：
 * 1. 调用 SPNet 进行深度补全
 * 2. 检测补全是否失败 (mean depth change > epsilon_1)
 * 3. 过滤边缘区域 (高深度梯度)
 * 4. 可选: 膨胀 LiDAR 点 mask
 * 5. 在无 LiDAR 深度的 30x30 patch 中采样最小深度像素
 * 6. 返回补充点列表
 * 
 * @param spnet SPNet 封装器引用
 * @param rgb RGB 图像 (3, H, W) [0, 1]
 * @param sparse_depth 稀疏 LiDAR 深度图 (H, W)
 * @param params 深度补全参数
 * @return DepthCompletionResult 补全结果
 */
DepthCompletionResult complete_depth_for_keyframe(
    SPNetWrapper& spnet,
    const torch::Tensor& rgb,
    const torch::Tensor& sparse_depth,
    const DepthCompletionParams& params
);

/**
 * @brief 检测深度补全是否失败
 * 
 * 通过比较补全前后的平均深度变化来判断补全是否成功。
 * 如果变化超过 epsilon_1 阈值，认为补全失败。
 * 
 * @param sparse_depth 原始稀疏深度
 * @param dense_depth 补全后的稠密深度
 * @param epsilon_1 阈值
 * @return true 如果补全失败
 */
bool detect_completion_failure(
    const torch::Tensor& sparse_depth,
    const torch::Tensor& dense_depth,
    float epsilon_1
);

/**
 * @brief 计算深度梯度并过滤边缘
 * 
 * 使用中心差分计算深度梯度，过滤梯度过大的区域（边缘）。
 * 
 * @param depth 深度图 (H, W)
 * @param gradient_threshold 梯度阈值
 * @return torch::Tensor 有效区域 mask (H, W)
 */
torch::Tensor filter_depth_edges(
    const torch::Tensor& depth,
    float gradient_threshold
);

/**
 * @brief 膨胀 LiDAR 有效区域 mask
 * 
 * 使用 max pooling 实现形态学膨胀。当 dilate_radius=0 时返回原始 mask。
 * 
 * @param lidar_mask LiDAR 有效像素 mask (H, W)
 * @param dilate_radius 膨胀半径 (0=不膨胀)
 * @return torch::Tensor 膨胀后的 mask (H, W)
 */
torch::Tensor dilate_mask(
    const torch::Tensor& lidar_mask,
    int dilate_radius
);

/**
 * @brief 在无 LiDAR 深度的 patch 中采样最小深度像素
 * 
 * 将图像划分为 patch_size x patch_size 的网格。
 * 对于没有 LiDAR 深度的 patch，在过滤后的稠密深度图中
 * 选择深度最小且满足条件的像素加入补充列表。
 * 
 * @param sparse_depth 稀疏 LiDAR 深度 (H, W)
 * @param filtered_dense_depth 过滤后的稠密深度 (H, W)
 * @param dilated_lidar_mask 膨胀后的 LiDAR mask (H, W)
 * @param rgb RGB 图像 (3, H, W)
 * @param params 参数
 * @return DepthCompletionResult 采样结果
 */
DepthCompletionResult sample_from_patches(
    const torch::Tensor& sparse_depth,
    const torch::Tensor& filtered_dense_depth,
    const torch::Tensor& dilated_lidar_mask,
    const torch::Tensor& rgb,
    const DepthCompletionParams& params
);
/**
 * @brief SPNet点生成结果
 */
struct SPNetPointsResult
{
    torch::Tensor fused_points;        ///< (N, 3) 世界坐标系下的点
    torch::Tensor fused_colors;        ///< (N, 3) RGB颜色 [0, 1]
    torch::Tensor fused_depths;        ///< (N, 1) 深度值
    torch::Tensor fused_pixels;        ///< (N, 2) 像素坐标 (u, v)
};

/**
 * @brief 执行 SPNet 补全并生成 3D 点云及属性 (Unified SPNet Logic)
 * 
 * 封装了深度补全与补点点云生成逻辑（不再负责法线计算）。
 * 
 * @param spnet SPNet 模型
 * @param rgb RGB 图像 (3, H, W)
 * @param sparse_depth 稀疏 LiDAR 深度 (H, W)
 * @param params 补全参数
 * @param cam 当前相机对象 (用于反投影和坐标变换)
 * @return SPNetPointsResult // 包含 points/colors/depths/pixels
 */
SPNetPointsResult get_spnet_points(
    SPNetWrapper& spnet,
    const torch::Tensor& rgb,
    const torch::Tensor& sparse_depth,
    const DepthCompletionParams& params,
    const std::shared_ptr<Camera>& cam
);

/**
 * @brief DA3 深度引导结果（用于法线初始化）
 */
struct DA3GuidanceResult
{
    bool success = false;                 ///< 是否成功得到可用的 DA3 引导结果
    torch::Tensor aligned_depth;          ///< (H, W) 与稀疏深度最小二乘对齐后的 DA3 深度
    torch::Tensor normal_map;             ///< (H, W, 3) 相机坐标系法线图（仅有效像素有意义）
    torch::Tensor normal_valid_mask;      ///< (H, W, 1) 法线有效掩码（float: 1/0）
};

/**
 * @brief 构建 DA3 法线引导（推理 + 最小二乘对齐 + 法线与掩码）
 *
 * 注意：
 * 1) 深度对齐采用“仿射 a*x+b 优先，必要时回退到纯尺度 a*x”；
 * 2) 法线有效条件严格使用：
 *    min_depth < d < max_depth 且 conf >= threshold（若启用阈值）；
 * 3) 可选保存对齐后的深度可视化图 *_da3_aligned_ls.png。
 */
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
    const std::string& depth_vis_dir
);

/**
 * @brief 按像素坐标采样 DA3 法线与有效掩码（用于 fused 点）
 *
 * @param guidance DA3 引导结果（全分辨率）
 * @param pixels (N, 2) 像素坐标，顺序为 (u, v)
 * @return pair{normals, valid_mask}
 *         normals: (N, 3) 相机系法线
 *         valid_mask: (N, 1) float 掩码（1/0）
 */
std::pair<torch::Tensor, torch::Tensor> sample_da3_guidance_by_pixels(
    const DA3GuidanceResult& guidance,
    const torch::Tensor& pixels
);
