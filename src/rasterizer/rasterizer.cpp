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

#include "rasterizer.h"

/**
 * @brief 高斯光栅化算子前向传播 (PyTorch Autograd 扩展)
 *
 * 该函数是连接 PyTorch 和 CUDA 底层实现的核心桥梁。它调用底层 CUDA 核函数进行渲染，
 * 并负责管理和保存反向传播所需的中间缓冲区（Buffers）。
 *
 * @param ctx 自动求导上下文，用于保存状态和张量
 * @param means3D 3D中心位置 (N, 3)
 * @param means2D 2D中心位置 (N, 3) - 仅作为梯度占位符
 * @param dc 球谐函数直流分量 (N, 1, 3)
 * @param sh 球谐函数高阶分量 (N, 15, 3)
 * @param colors_precomp 预计算颜色（可选）
 * @param opacities 不透明度 (N, 1)
 * @param scales 缩放参数 (N, 3)
 * @param rotations 旋转参数 (N, 4)
 * @param cov3Ds_precomp 预计算3D协方差（可选）
 * @param raster_settings 渲染全局配置
 *
 * @return torch::autograd::tensor_list 返回包含：
 *         0: color - 渲染生成的RGB图像 (3, H, W)
 *         1: radii - 高斯点在屏幕上的半径 (N,)
 *         2: final_T - 最终透射率/深度图 (H, W)
 */
torch::autograd::tensor_list
GaussianRasterizerFunction::forward(
    torch::autograd::AutogradContext *ctx,
    torch::Tensor means3D,
    torch::Tensor means2D,
    torch::Tensor dc,
    torch::Tensor sh,
    torch::Tensor colors_precomp,
    torch::Tensor opacities,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor cov3Ds_precomp,
    GaussianRasterizationSettings raster_settings)
{
    // === 调用底层 CUDA 光栅化算子 ===
    auto rasterization_result = RasterizeGaussiansCUDA(
        raster_settings.bg_,
        means3D,
        colors_precomp,
        opacities,
        scales,
        rotations,
        raster_settings.scale_modifier_,
        cov3Ds_precomp,
        raster_settings.viewmatrix_,
        raster_settings.projmatrix_,
        raster_settings.tanfovx_,
        raster_settings.tanfovy_,
        raster_settings.image_height_,
        raster_settings.image_width_,
        raster_settings.limx_neg_,
        raster_settings.limx_pos_,
        raster_settings.limy_neg_,
        raster_settings.limy_pos_,
        dc,
        sh,
        raster_settings.sh_degree_,
        raster_settings.campos_,
        raster_settings.prefiltered_,
        raster_settings.debug_,
        raster_settings.no_color_
    );

    // === 解析渲染结果与缓冲区 ===
    auto num_rendered = std::get<0>(rasterization_result);    // 成功光栅化的高斯实例数
    auto num_buckets = std::get<1>(rasterization_result);     // 渲染过程中分配的 Bucket 数量
    auto color = std::get<2>(rasterization_result);           // 渲染图像
    auto final_T = std::get<3>(rasterization_result);         // 最终透射率图
    auto radii = std::get<4>(rasterization_result);           // 投影半径
    
    // 以下 Buffer 存储了前向传播的中间状态，必须保存用于反向梯度计算
    auto geomBuffer = std::get<5>(rasterization_result);       // 几何计算缓存
    auto binningBuffer = std::get<6>(rasterization_result);    // 排序缓存
    auto imgBuffer = std::get<7>(rasterization_result);        // 图像处理缓存
    auto sampleBuffer = std::get<8>(rasterization_result);     // 采样/梯度缓存

    // === 保存上下文数据用于反向传播 ===
    ctx->saved_data["num_rendered"] = num_rendered;
    ctx->saved_data["num_buckets"] = num_buckets;
    ctx->saved_data["scale_modifier"] = raster_settings.scale_modifier_;
    ctx->saved_data["tanfovx"] = raster_settings.tanfovx_;
    ctx->saved_data["tanfovy"] = raster_settings.tanfovy_;
    ctx->saved_data["sh_degree"] = raster_settings.sh_degree_;
    ctx->saved_data["limx_neg"] = raster_settings.limx_neg_;
    ctx->saved_data["limx_pos"] = raster_settings.limx_pos_;
    ctx->saved_data["limy_neg"] = raster_settings.limy_neg_;
    ctx->saved_data["limy_pos"] = raster_settings.limy_pos_;
    ctx->saved_data["lambda_erank"] = raster_settings.lambda_erank_;

    // 保存张量：注意 detach 后的张量可以安全地跨 Pass 传递
    ctx->save_for_backward({raster_settings.bg_,
                            raster_settings.viewmatrix_,
                            raster_settings.projmatrix_,
                            raster_settings.campos_,
                            colors_precomp,
                            means3D,
                            scales,
                            rotations,
                            cov3Ds_precomp,
                            radii,
                            dc,
                            sh,
                            geomBuffer,
                            binningBuffer,
                            imgBuffer,
                            sampleBuffer});

    return {color, radii, final_T};
}

/**
 * @brief 高斯光栅化算子反向传播
 *
 * 通过链式法则计算损失函数对于所有可优化输入参数（位置、旋转、缩放等）的梯度。
 *
 * @param ctx 前向传播保存的上下文
 * @param grad_outputs 从下层传播回来的梯度元组 (dL/dcolor, dL/dradii, dL/dfinal_T)
 *
 * @return torch::autograd::tensor_list 返回所有输入参数的梯度列表
 */
torch::autograd::tensor_list
GaussianRasterizerFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::tensor_list grad_outputs)
{
    // === 恢复前向传播保存的元数据和张量 ===
    auto num_rendered = ctx->saved_data["num_rendered"].toInt();
    auto num_buckets = ctx->saved_data["num_buckets"].toInt();
    auto scale_modifier = static_cast<float>(ctx->saved_data["scale_modifier"].toDouble());
    auto tanfovx = static_cast<float>(ctx->saved_data["tanfovx"].toDouble());
    auto tanfovy = static_cast<float>(ctx->saved_data["tanfovy"].toDouble());
    auto sh_degree = ctx->saved_data["sh_degree"].toInt();
    auto limx_neg = static_cast<float>(ctx->saved_data["limx_neg"].toDouble());
    auto limx_pos = static_cast<float>(ctx->saved_data["limx_pos"].toDouble());
    auto limy_neg = static_cast<float>(ctx->saved_data["limy_neg"].toDouble());
    auto limy_pos = static_cast<float>(ctx->saved_data["limy_pos"].toDouble());
    auto lambda_erank = static_cast<float>(ctx->saved_data["lambda_erank"].toDouble());

    auto saved = ctx->get_saved_variables();
    auto bg = saved[0];
    auto viewmatrix = saved[1];
    auto projmatrix = saved[2];
    auto campos = saved[3];
    auto colors_precomp = saved[4];
    auto means3D = saved[5];
    auto scales = saved[6];
    auto rotations = saved[7];
    auto cov3Ds_precomp = saved[8];
    auto radii = saved[9];
    auto dc = saved[10];
    auto sh = saved[11];
    auto geomBuffer = saved[12];
    auto binningBuffer = saved[13];
    auto imgBuffer = saved[14];
    auto sampleBuffer = saved[15];

    // 获取损失相对于输出颜色的梯度
    auto dL_dcolor = grad_outputs[0];

    // === 调用底层 CUDA 反向传播算子 ===
    auto rasterization_backward_result = RasterizeGaussiansBackwardCUDA(
        bg,
        means3D,
        radii,
        colors_precomp,
        scales,
        rotations,
        scale_modifier,
        cov3Ds_precomp,
        viewmatrix,
        projmatrix,
        tanfovx,
        tanfovy,
        limx_neg,
        limx_pos,
        limy_neg,
        limy_pos,
        dL_dcolor,
        dc,
        sh,
        sh_degree,
        campos,
        geomBuffer,
        num_rendered,
        binningBuffer,
        imgBuffer,
        num_buckets,
        sampleBuffer,
        lambda_erank,
        false
    );

    // === 根据参数顺序返回梯度元组 ===
    return {
        std::get<3>(rasterization_backward_result)/*dL_dmeans3D*/,
        std::get<0>(rasterization_backward_result)/*dL_dmeans2D*/,
        std::get<5>(rasterization_backward_result)/*dL_ddc*/,
        std::get<6>(rasterization_backward_result)/*dL_dsh*/,
        std::get<1>(rasterization_backward_result)/*dL_dcolors_precomp*/,
        std::get<2>(rasterization_backward_result)/*dL_dopacities*/,
        std::get<7>(rasterization_backward_result)/*dL_dscales*/,
        std::get<8>(rasterization_backward_result)/*dL_drotations*/,
        std::get<4>(rasterization_backward_result)/*dL_dcov3Ds_precomp*/,
        torch::Tensor()/*dL_draster_setting*/
    };
}

/**
 * @brief GaussianRasterizer 前向逻辑封装
 * 
 * 该方法通过 apply() 调用自定义的 Autograd Function，实现可微分渲染。
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
GaussianRasterizer::forward(
    torch::Tensor means3D,
    torch::Tensor means2D,
    torch::Tensor opacities,
    torch::Tensor dc,
    torch::Tensor shs,
    torch::Tensor colors_precomp,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor cov3D_precomp)

{
    auto raster_settings = this->raster_settings_;
    
    // === 确定初始化配置 ===
    torch::TensorOptions options;
    colors_precomp = torch::tensor({}, options.device(torch::kCUDA));
    cov3D_precomp = torch::tensor({}, options.device(torch::kCUDA));

    // 调用 apply() 触发自定义算子的执行
    auto result = GaussianRasterizerFunction::apply(
        means3D,
        means2D,
        dc,
        shs,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3D_precomp,
        raster_settings
    );
    
    // 返回 (color, radii, final_T)
    return std::make_tuple(result[0], result[1], result[2]);
}