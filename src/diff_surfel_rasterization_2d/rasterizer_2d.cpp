/*
 * 2D Gaussian Splatting Rasterizer - Autograd Layer Implementation
 * 
 * 本文件仿照 Gaussian-LIC 的 rasterizer.cpp 编写，
 * 为 2DGS 提供与 3DGS 完全一致的封装层次。
 */

#include "rasterizer_2d.h"

/**
 * @brief 2D高斯光栅化算子前向传播 (PyTorch Autograd 扩展)
 *
 * 该函数是连接 PyTorch 和 CUDA 底层实现的核心桥梁。它调用底层 CUDA 核函数进行渲染，
 * 并负责管理和保存反向传播所需的中间缓冲区（Buffers）。
 *
 * @param ctx 自动求导上下文，用于保存状态和张量
 * @param means3D 3D中心位置 (N, 3)
 * @param means2D 2D中心位置 (N, 2) - 仅作为梯度占位符
 * @param sh 完整SH系数 (N, K, 3)
 * @param colors_precomp 预计算颜色（可选）
 * @param opacities 不透明度 (N, 1)
 * @param scales 2D缩放参数 (N, 2) - 与3DGS的(N,3)不同！
 * @param rotations 旋转参数 (N, 4)
 * @param transMat_precomp 预计算变换矩阵（可选）
 * @param raster_settings 渲染全局配置
 *
 * @return torch::autograd::tensor_list 返回包含：
 *         0: color - 渲染生成的RGB图像 (3, H, W)
 *         1: out_others - 辅助输出 (7, H, W) [depth, alpha, normal(3), median_depth, distortion]
 *         2: radii - 高斯点在屏幕上的半径 (N,)
 */
torch::autograd::tensor_list
GaussianRasterizer2DFunction::forward(
    torch::autograd::AutogradContext *ctx,
    torch::Tensor means3D,
    torch::Tensor means2D,
    torch::Tensor sh,
    torch::Tensor colors_precomp,
    torch::Tensor opacities,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor transMat_precomp,
    GaussianRasterization2DSettings raster_settings)
{
    // === 调用底层 CUDA 2DGS 光栅化算子 ===
    auto rasterization_result = RasterizeGaussiansCUDA(
        raster_settings.bg_,
        means3D,
        colors_precomp,
        opacities,
        scales,
        rotations,
        raster_settings.scale_modifier_,
        transMat_precomp,
        raster_settings.viewmatrix_,
        raster_settings.projmatrix_,
        raster_settings.tanfovx_,
        raster_settings.tanfovy_,
        raster_settings.image_height_,
        raster_settings.image_width_,
        sh,
        raster_settings.sh_degree_,
        raster_settings.campos_,
        raster_settings.prefiltered_,
        raster_settings.use_tile_culling_,
        raster_settings.debug_
    );

    // === 解析渲染结果与缓冲区 ===
    auto num_rendered = std::get<0>(rasterization_result);    // 成功光栅化的高斯实例数
    auto color = std::get<1>(rasterization_result);           // 渲染图像 (3, H, W)
    auto out_others = std::get<2>(rasterization_result);      // 辅助输出 (7, H, W)
    auto radii = std::get<3>(rasterization_result);           // 投影半径 (N,)
    
    // 以下 Buffer 存储了前向传播的中间状态，必须保存用于反向梯度计算
    auto geomBuffer = std::get<4>(rasterization_result);       // 几何计算缓存
    auto binningBuffer = std::get<5>(rasterization_result);    // 排序缓存
    auto imgBuffer = std::get<6>(rasterization_result);        // 图像处理缓存

    // === 保存上下文数据用于反向传播 ===
    ctx->saved_data["num_rendered"] = num_rendered;
    ctx->saved_data["scale_modifier"] = raster_settings.scale_modifier_;
    ctx->saved_data["tanfovx"] = raster_settings.tanfovx_;
    ctx->saved_data["tanfovy"] = raster_settings.tanfovy_;
    ctx->saved_data["sh_degree"] = raster_settings.sh_degree_;
    ctx->saved_data["image_height"] = raster_settings.image_height_;
    ctx->saved_data["image_width"] = raster_settings.image_width_;

    // 保存张量：注意这些张量在反向传播时会被恢复
    ctx->save_for_backward({
        raster_settings.bg_,
        raster_settings.viewmatrix_,
        raster_settings.projmatrix_,
        raster_settings.campos_,
        colors_precomp,
        means3D,
        scales,
        rotations,
        transMat_precomp,
        radii,
        sh,
        geomBuffer,
        binningBuffer,
        imgBuffer
    });

    return {color, out_others, radii};
}

/**
 * @brief 2D高斯光栅化算子反向传播
 *
 * 通过链式法则计算损失函数对于所有可优化输入参数（位置、旋转、缩放等）的梯度。
 *
 * @param ctx 前向传播保存的上下文
 * @param grad_outputs 从下层传播回来的梯度元组 (dL/dcolor, dL/dout_others, dL/dradii)
 *
 * @return torch::autograd::tensor_list 返回所有输入参数的梯度列表
 */
torch::autograd::tensor_list
GaussianRasterizer2DFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::tensor_list grad_outputs)
{
    // === 恢复前向传播保存的元数据 ===
    auto num_rendered = ctx->saved_data["num_rendered"].toInt();
    auto scale_modifier = static_cast<float>(ctx->saved_data["scale_modifier"].toDouble());
    auto tanfovx = static_cast<float>(ctx->saved_data["tanfovx"].toDouble());
    auto tanfovy = static_cast<float>(ctx->saved_data["tanfovy"].toDouble());
    auto sh_degree = ctx->saved_data["sh_degree"].toInt();
    auto image_height = ctx->saved_data["image_height"].toInt();
    auto image_width = ctx->saved_data["image_width"].toInt();

    // === 恢复前向传播保存的张量 ===
    auto saved = ctx->get_saved_variables();
    auto bg = saved[0];
    auto viewmatrix = saved[1];
    auto projmatrix = saved[2];
    auto campos = saved[3];
    auto colors_precomp = saved[4];
    auto means3D = saved[5];
    auto scales = saved[6];
    auto rotations = saved[7];
    auto transMat_precomp = saved[8];
    auto radii = saved[9];
    auto sh = saved[10];
    auto geomBuffer = saved[11];
    auto binningBuffer = saved[12];
    auto imgBuffer = saved[13];

    // 获取损失相对于输出的梯度
    auto dL_dcolor = grad_outputs[0];      // (3, H, W)
    auto dL_dout_others = grad_outputs[1]; // (7, H, W) - 包含深度梯度等
    
    // 从 out_others 梯度中提取深度梯度 (第0个通道)
    auto dL_ddepths = dL_dout_others.index({0, torch::indexing::Slice(), torch::indexing::Slice()});

    // === 调用底层 CUDA 反向传播算子 ===
    auto rasterization_backward_result = RasterizeGaussiansBackwardCUDA(
        bg,
        means3D,
        radii,
        colors_precomp,
        scales,
        rotations,
        scale_modifier,
        transMat_precomp,
        viewmatrix,
        projmatrix,
        tanfovx,
        tanfovy,
        dL_dcolor,
        dL_dout_others,  // 2DGS 特有：深度梯度
        sh,
        sh_degree,
        campos,
        geomBuffer,
        num_rendered,
        binningBuffer,
        imgBuffer,
        false  // debug
    );

    // === 解析反向传播结果 ===
    // 返回顺序参考 rasterize_points.h 中的定义
    auto dL_dmeans2D = std::get<0>(rasterization_backward_result);
    auto dL_dcolors = std::get<1>(rasterization_backward_result);
    auto dL_dopacity = std::get<2>(rasterization_backward_result);
    auto dL_dmeans3D = std::get<3>(rasterization_backward_result);
    auto dL_dtransMat = std::get<4>(rasterization_backward_result);
    auto dL_dsh = std::get<5>(rasterization_backward_result);
    auto dL_dscales = std::get<6>(rasterization_backward_result);
    auto dL_drots = std::get<7>(rasterization_backward_result);
    auto dL_dnormal = std::get<8>(rasterization_backward_result);

    // === 根据 forward 参数顺序返回梯度元组 ===
    // forward 参数顺序: means3D, means2D, sh, colors_precomp, opacities, scales, rotations, transMat_precomp, settings
    return {
        dL_dmeans3D,          // dL/d(means3D)
        dL_dmeans2D,          // dL/d(means2D)
        dL_dsh,               // dL/d(sh)
        dL_dcolors,           // dL/d(colors_precomp)
        dL_dopacity,          // dL/d(opacities)
        dL_dscales,           // dL/d(scales)
        dL_drots,             // dL/d(rotations)
        dL_dtransMat,         // dL/d(transMat_precomp)
        torch::Tensor()       // dL/d(raster_settings) - 不可微
    };
}

/**
 * @brief GaussianRasterizer2D 前向逻辑封装
 * 
 * 该方法通过 apply() 调用自定义的 Autograd Function，实现可微分渲染。
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
GaussianRasterizer2D::forward(
    torch::Tensor means3D,
    torch::Tensor means2D,
    torch::Tensor opacities,
    torch::Tensor sh,
    torch::Tensor colors_precomp,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor transMat_precomp)
{
    auto raster_settings = this->raster_settings_;
    
    // === 确保空张量在正确设备上 ===
    torch::TensorOptions options;
    if (colors_precomp.numel() == 0) {
        colors_precomp = torch::tensor({}, options.device(torch::kCUDA));
    }
    if (transMat_precomp.numel() == 0) {
        transMat_precomp = torch::tensor({}, options.device(torch::kCUDA));
    }

    // === 调用 apply() 触发自定义算子的执行 ===
    auto result = GaussianRasterizer2DFunction::apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        transMat_precomp,
        raster_settings
    );
    
    // 返回 (color, out_others, radii)
    return std::make_tuple(result[0], result[1], result[2]);
}
