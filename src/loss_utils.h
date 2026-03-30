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

#include <vector>

#include <torch/torch.h>

#include <fused-ssim/ssim.h>

/**
 * @brief 损失函数工具命名空间
 * 包含各种图像质量评估和损失计算函数
 */
namespace loss_utils
{

/**
 * @brief 计算L1损失（绝对值损失）
 * @param network_output 网络输出张量
 * @param gt 真实标签张量
 * @return L1损失值
 */
inline torch::Tensor l1_loss(torch::Tensor &network_output, torch::Tensor &gt)
{
    return torch::abs(network_output - gt).mean();
}

/**
 * @brief 计算峰值信噪比（PSNR）
 * @param img1 第一张图像张量
 * @param img2 第二张图像张量
 * @return PSNR值
 */
inline torch::Tensor psnr(torch::Tensor &img1, torch::Tensor &img2)
{
    auto mse = torch::pow(img1 - img2, 2).mean();
    return 10.0f * torch::log10(1.0f / mse);
}

/**
 * @brief PSNR计算的Python参考实现
 * 该函数用于批量计算多张图像的PSNR值
 *
 * 数学公式：
 * PSNR = 20 * log10(1.0 / sqrt(MSE))
 * 其中MSE是均方误差
 *
 * @note 这个函数是Gaussian Splatting中使用的PSNR计算方式，
 *       与标准PSNR略有不同，主要用于批量图像评估
 */

/**
 * @brief 计算高斯散射版本的PSNR（用于批量处理）
 * @param img1 第一批图像张量
 * @param img2 第二批图像张量
 * @return PSNR平均值
 */
inline torch::Tensor psnr_gaussian_splatting(torch::Tensor &img1, torch::Tensor &img2)
{
    auto mse = torch::pow(img1 - img2, 2).view({img1.size(0) , -1}).mean(1, /*keepdim=*/true);
    return 20.0f * torch::log10(1.0f / torch::sqrt(mse)).mean();
}

/**
 * @brief 生成一维高斯核
 * @param window_size 窗口大小
 * @param sigma 高斯分布的标准差
 * @param device_type 设备类型（默认为CUDA）
 * @return 归一化后的一维高斯核张量
 */
inline torch::Tensor gaussian(
    int window_size,
    float sigma,
    torch::DeviceType device_type = torch::kCUDA)
{
    std::vector<float> gauss_values(window_size);
    for (int x = 0; x < window_size; ++x) {
        int temp = x - window_size / 2;
        gauss_values[x] = std::exp(-temp * temp / (2.0f * sigma * sigma));
    }
    torch::Tensor gauss = torch::tensor(
        gauss_values,
        torch::TensorOptions().device(device_type));
    return gauss / gauss.sum();
}

/**
 * @brief 创建二维高斯窗口（用于SSIM计算）
 * @param window_size 窗口大小
 * @param channel 通道数
 * @param device_type 设备类型（默认为CUDA）
 * @return 可导变量形式的二维高斯窗口
 */
inline torch::autograd::Variable create_window(
    int window_size,
    int64_t channel,
    torch::DeviceType device_type = torch::kCUDA)
{
    auto _1D_window = gaussian(window_size, 1.5f, device_type).unsqueeze(1);
    auto _2D_window = _1D_window.mm(_1D_window.t()).to(torch::kFloat).unsqueeze(0).unsqueeze(0);
    auto window = torch::autograd::Variable(_2D_window.expand({channel, 1, window_size, window_size}).contiguous());
    return window;
}

/**
 * @brief SSIM（结构相似性指数）相关函数
 *
 * SSIM是一种用于衡量两幅图像相似度的指标，相比于简单的像素差异，
 * SSIM更符合人眼的视觉特性。它考虑了亮度、对比度和结构三个方面。
 *
 * 数学公式：
 * SSIM(x,y) = (2μxμy + C1)(2σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
 *
 * 其中：
 * - μx, μy: 图像x和y的均值
 * - σx², σy²: 图像x和y的方差
 * - σxy: 图像x和y的协方差
 * - C1, C2: 稳定性常量，防止分母为零
 */

/**
 * @brief SSIM内部计算函数
 *
 * 该函数实现SSIM的核心计算逻辑，使用滑动窗口方式计算局部相似度。
 * 通过二维卷积计算局部统计量（均值、方差、协方差），然后计算SSIM值。
 *
 * 计算步骤：
 * 1. 使用高斯窗口计算局部均值 μ1, μ2
 * 2. 计算局部方差 σ1², σ2² 和协方差 σ12
 * 3. 使用SSIM公式计算相似度映射
 * 4. 可选择对结果取平均
 *
 * @param img1 第一张图像张量，形状为 (batch, channel, height, width)
 * @param img2 第二张图像张量，形状为 (batch, channel, height, width)
 * @param window 高斯权重窗口，用于局部统计计算
 * @param window_size 窗口大小（通常为11）
 * @param channel 图像通道数（RGB为3，灰度为1）
 * @param size_average 是否对所有像素的SSIM值取平均（true返回标量，false返回SSIM映射）
 * @return 如果size_average=true，返回平均SSIM值；否则返回SSIM值映射
 */
inline torch::Tensor _ssim(
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::autograd::Variable &window,
    int window_size,
    int64_t channel,
    bool size_average = true)
{
    int window_size_half = window_size / 2;

    // 计算局部均值：使用高斯加权卷积
    auto mu1 = torch::nn::functional::conv2d(img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel));
    auto mu2 = torch::nn::functional::conv2d(img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel));

    // 计算均值的平方和乘积，用于后续计算
    auto mu1_sq = mu1.pow(2);
    auto mu2_sq = mu2.pow(2);
    auto mu1_mu2 = mu1 * mu2;

    // 计算局部方差：E[X²] - (E[X])²
    auto sigma1_sq = torch::nn::functional::conv2d(img1 * img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel))
                    - mu1_sq;
    auto sigma2_sq = torch::nn::functional::conv2d(img2 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel))
                    - mu2_sq;

    // 计算协方差：E[XY] - E[X]E[Y]
    auto sigma12 = torch::nn::functional::conv2d(img1 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel))
                    - mu1_mu2;

    // SSIM稳定性常量，防止数值不稳定
    auto C1 = 0.01 * 0.01;  // 亮度稳定性常量
    auto C2 = 0.03 * 0.03;  // 对比度稳定性常量

    // 计算SSIM映射：SSIM公式实现
    // SSIM = (2μ1μ2 + C1)(2σ12 + C2) / ((μ1² + μ2² + C1)(σ1² + σ2² + C2))
    auto ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

    if (size_average)
        return ssim_map.mean();
    else
        return ssim_map.mean(1).mean(1).mean(1);
}

/**
 * @brief 计算结构相似性指数（SSIM）
 * @param img1 第一张图像
 * @param img2 第二张图像
 * @param device_type 设备类型（默认为CUDA）
 * @param window_size 窗口大小（默认为11）
 * @param size_average 是否对结果取平均（默认为true）
 * @return SSIM值
 */
inline torch::Tensor ssim(
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::DeviceType device_type = torch::kCUDA,
    int window_size = 11,
    bool size_average = true)
{
    auto channel = img1.size(-3);
    auto window = create_window(window_size, channel, device_type);

    // window = window.to(img1.device());
    window = window.type_as(img1);

    return _ssim(img1, img2, window, window_size, channel, size_average);
}

/**
 * @brief SSIM计算中的稳定性常量
 *
 * 这些常量用于防止在计算SSIM时出现数值不稳定情况，
 * 特别是在图像区域对比度很低（接近常数）时。
 *
 * C1 = (0.01)^2 = 0.0001 - 用于亮度比较的稳定性常量
 * C2 = (0.03)^2 = 0.0009 - 用于对比度比较的稳定性常量
 *
 * 这些值基于经验选择，在标准SSIM实现中被广泛使用。
 */
const float C1 = std::pow(0.01, 2);
const float C2 = std::pow(0.03, 2);

/**
 * @brief 融合SSIM相关函数
 *
 * FusedSSIM是使用CUDA加速的SSIM实现，相比纯PyTorch实现有更好的性能。
 * 它通过自定义的自动微分操作来实现高效的梯度计算。
 */

/**
 * @brief 融合SSIM映射类，用于CUDA加速的SSIM计算
 * 该类继承自torch::autograd::Function，实现自动微分
 */
class FusedSSIMMap : public torch::autograd::Function<FusedSSIMMap> 
{
public:
    /**
     * @brief 前向传播函数
     *
     * 该函数执行SSIM的前向计算，使用CUDA加速的fusedssim函数。
     * 同时保存反向传播所需的中间变量和梯度信息。
     *
     * @param ctx 自动微分上下文，用于保存状态和梯度信息
     * @param C1 SSIM亮度稳定性常量 (通常为0.0001)
     * @param C2 SSIM对比度稳定性常量 (通常为0.0009)
     * @param img1 第一张输入图像张量，形状为 (N, C, H, W)
     * @param img2 第二张输入图像张量，形状为 (N, C, H, W)
     * @return SSIM相似度映射，形状与输入图像相同
     */
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, float C1, float C2, 
                                 torch::Tensor& img1, torch::Tensor& img2) 
    {
        // 设置计算参数
        std::string padding = "same";  // 使用same padding保持输出尺寸
        bool train = true;             // 启用训练模式，计算梯度信息

        // 调用CUDA加速的SSIM计算函数
        // 返回值包含：SSIM映射和反向传播所需的梯度中间变量
        auto result = fusedssim(C1, C2, img1, img2, train);
        torch::Tensor ssim_map = std::get<0>(result);         // SSIM相似度映射
        torch::Tensor dm_dmu1 = std::get<1>(result);          // d(SSIM)/d(μ1) - 对均值1的梯度
        torch::Tensor dm_dsigma1_sq = std::get<2>(result);    // d(SSIM)/d(σ1²) - 对方差1的梯度
        torch::Tensor dm_dsigma12 = std::get<3>(result);      // d(SSIM)/d(σ12) - 对协方差的梯度

        // 如果使用valid padding，需要裁剪输出以匹配输入尺寸
        if (padding == "valid")
        {
            ssim_map = ssim_map.slice(2, 5, -5).slice(3, 5, -5);
        }

        // 保存反向传播所需的变量和参数
        ctx->save_for_backward({img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12});
        ctx->saved_data["C1"] = C1;
        ctx->saved_data["C2"] = C2;
        ctx->saved_data["padding"] = padding;

        return ssim_map;
    }

    /**
     * @brief 反向传播函数
     *
     * 计算SSIM损失相对于输入图像的梯度。通过保存的前向传播中间变量
     * 和预计算的梯度信息，高效地计算反向传播梯度。
     *
     * @param ctx 自动微分上下文，包含前向传播保存的状态
     * @param grad_output 从损失函数传播下来的梯度，形状与SSIM映射相同
     * @return 输入梯度的元组：(dL/dC1, dL/dC2, dL/dimg1, dL/dimg2)
     *         前两个梯度为空张量，最后两个是相对于输入图像的梯度
     */
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_output) 
    {
        // 恢复前向传播保存的变量
        auto saved = ctx->get_saved_variables();
        torch::Tensor img1 = saved[0];           // 输入图像1（已分离梯度）
        torch::Tensor img2 = saved[1];           // 输入图像2
        torch::Tensor dm_dmu1 = saved[2];        // SSIM对均值1的导数
        torch::Tensor dm_dsigma1_sq = saved[3];  // SSIM对方差1的导数
        torch::Tensor dm_dsigma12 = saved[4];    // SSIM对协方差的导数

        // 恢复保存的参数
        float C1 = static_cast<float>(ctx->saved_data["C1"].toDouble());
        float C2 = static_cast<float>(ctx->saved_data["C2"].toDouble());
        std::string padding = ctx->saved_data["padding"].toStringRef();

        // 获取损失函数传播下来的梯度
        torch::Tensor dL_dmap = grad_output[0];

        // 处理padding模式：如果使用valid padding，需要相应调整梯度
        if (padding == "valid")
        {
            dL_dmap = torch::zeros_like(img1);
            dL_dmap.slice(2, 5, -5).slice(3, 5, -5) = grad_output[0];
        }

        // 调用CUDA加速的反向传播函数，计算相对于输入图像的梯度
        torch::Tensor grad = fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);

        // 返回梯度元组：(dL/dC1, dL/dC2, dL/dimg1, dL/dimg2, dL/dxxx, dL/dxxx)
        // C1和C2的梯度为空（不参与优化），只有img1的梯度被计算
        return {torch::Tensor(), torch::Tensor(), grad, torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

/**
 * @brief 计算融合SSIM（使用CUDA加速）
 *
 * 这是一个高层接口函数，用于计算两张图像之间的SSIM相似度。
 * 使用预定义的稳定性常量C1和C2，通过FusedSSIMMap类的apply方法
 * 执行自动微分的SSIM计算。
 *
 * 该函数主要用于：
 * 1. Gaussian Splatting的渲染质量评估
 * 2. 训练过程中的损失计算
 * 3. 图像相似度度量
 *
 * @param img1 第一张输入图像，形状为 (N, C, H, W) 或 (C, H, W)
 * @param img2 第二张输入图像，形状需要与img1匹配
 * @return 平均SSIM值，范围为[0, 1]，1表示完全相似，0表示完全不同
 *
 * @note 返回的是平均值而不是SSIM映射，适用于损失函数计算
 * @note 使用全局常量C1和C2作为稳定性参数
 */
inline torch::Tensor fused_ssim(torch::Tensor& img1, torch::Tensor& img2) 
{    
    torch::Tensor map = FusedSSIMMap::apply(C1, C2, img1, img2);
    return map.mean();
}

// ======================= 2DGS 正则化损失函数 =======================

/**
 * @brief 从深度图计算表面法线 (2DGS 专用)
 * 
 * 当前实现采用“相机坐标系三维点邻域叉积”：
 * 1. 先将每个像素反投影为相机系 3D 点 p(u,v)；
 * 2. 用中心差分构造切向量：
 *      dx = p(u+1,v) - p(u-1,v)
 *      dy = p(u,v+1) - p(u,v-1)
 * 3. 法线 n = normalize(dx x dy)。
 * 
 * 相比 n = (-dz/dx/fx, -dz/dy/fy, 1) 近似形式，此实现几何一致性更好，
 * 对大平面区域更稳定，适合用于旋转初始化与法线监督。
 * 
 * @param depth 深度图张量，形状为 (H, W)
 * @param fx 相机 x 方向焦距
 * @param fy 相机 y 方向焦距
 * @param cx 相机 x 方向主点
 * @param cy 相机 y 方向主点
 * @return torch::Tensor 法线图，形状为 (3, H, W)，已归一化
 */
inline torch::Tensor depth_to_normal(
    const torch::Tensor& depth,
    float fx, float fy, float cx, float cy)
{
    int H = depth.size(0);
    int W = depth.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(depth.device());
    auto z = depth.to(torch::kFloat32);

    // 极小分辨率无法做中心差分，回退到默认 +Z 法线，避免数值不稳定。
    if (H < 3 || W < 3) {
        auto normal_small = torch::zeros({3, H, W}, options);
        normal_small.index_put_({2, torch::indexing::Slice(), torch::indexing::Slice()}, 1.0f);
        return normal_small;
    }

    // === Step 1: 像素反投影到相机系 3D 点 ===
    auto y_coords = torch::arange(0, H, options).view({H, 1}).expand({H, W});
    auto x_coords = torch::arange(0, W, options).view({1, W}).expand({H, W});
    auto x = (x_coords - cx) * z / fx;
    auto y = (y_coords - cy) * z / fy;
    auto points = torch::stack({x, y, z}, 0);  // (3, H, W)

    // === Step 2: 中心差分构造切向量 ===
    auto dx = torch::zeros_like(points);
    auto dy = torch::zeros_like(points);
    dx.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, W - 1)},
        points.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(2, W)}) -
        points.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, W - 2)})
    );
    dy.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(1, H - 1), torch::indexing::Slice()},
        points.index({torch::indexing::Slice(), torch::indexing::Slice(2, H), torch::indexing::Slice()}) -
        points.index({torch::indexing::Slice(), torch::indexing::Slice(0, H - 2), torch::indexing::Slice()})
    );

    // === Step 3: 叉积求法线并归一化 ===
    auto normal_hwc = torch::cross(dx.permute({1, 2, 0}), dy.permute({1, 2, 0}), 2);  // (H, W, 3)
    auto normal = normal_hwc.permute({2, 0, 1});  // (3, H, W)

    // 仅在有效深度区域保留法线，避免无效区域影响后续监督。
    auto valid_depth = torch::isfinite(z) & (z > 0.0f);  // (H, W)
    normal = normal * valid_depth.unsqueeze(0).to(torch::kFloat32);

    auto norm = torch::sqrt((normal * normal).sum(0, true) + 1e-8f);  // (1, H, W)
    normal = normal / norm;

    return normal;
}

/**
 * @brief 2DGS 法线一致性损失 (Normal Consistency Loss)
 * 
 * 计算渲染法线与从深度图推导的表面法线之间的一致性损失。
 * 损失值为 1 - cos(angle)，其中 angle 是两个法线之间的夹角。
 * 
 * @param rendered_normal 渲染法线图，形状为 (3, H, W)
 * @param surf_normal 伪表面法线图 (GT)，形状为 (3, H, W)
 * @param alpha 渲染 Alpha 图，形状为 (H, W)，用于掩码
 * @return torch::Tensor 法线一致性损失标量
 */
inline torch::Tensor normal_consistency_loss(
    const torch::Tensor& rendered_normal,
    const torch::Tensor& surf_normal,
    const torch::Tensor& alpha)
{
    // 计算法线一致性: 1 - cos(angle) = 1 - dot(n1, n2)
    // rendered_normal 和 surf_normal 都应该是归一化的
    auto cos_similarity = (rendered_normal * surf_normal).sum(0);  // (H, W)
    auto normal_error = 1.0f - cos_similarity;  // (H, W)
    
    // 使用 alpha 作为掩码，只在有效区域计算损失
    auto mask = alpha > 0.5f;
    auto masked_error = normal_error * mask.to(torch::kFloat32);
    
    // 返回平均损失
    auto valid_count = mask.sum().clamp_min(1.0f);
    return masked_error.sum() / valid_count;
}

/**
 * @brief 2DGS 深度畸变损失 (Distortion Loss)
 * 
 * 直接对渲染器输出的畸变图求平均作为损失。
 * 该损失用于惩罚不合理的深度分布，减少深度伪影。
 * 
 * @param rendered_distortion 渲染畸变图，形状为 (H, W)
 * @return torch::Tensor 畸变损失标量
 */
inline torch::Tensor distortion_loss(const torch::Tensor& rendered_distortion)
{
    return rendered_distortion.mean();
}

}
