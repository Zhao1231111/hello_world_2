/*
 * 2D Gaussian Splatting Rasterizer - Autograd Layer Implementation
 */

#include "rasterizer_2d.h"


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

    auto num_rendered = std::get<0>(rasterization_result);
    auto color = std::get<1>(rasterization_result);
    auto out_others = std::get<2>(rasterization_result);
    auto radii = std::get<3>(rasterization_result);

    auto geomBuffer = std::get<4>(rasterization_result);
    auto binningBuffer = std::get<5>(rasterization_result);
    auto imgBuffer = std::get<6>(rasterization_result);

    ctx->saved_data["num_rendered"] = num_rendered;
    ctx->saved_data["scale_modifier"] = raster_settings.scale_modifier_;
    ctx->saved_data["tanfovx"] = raster_settings.tanfovx_;
    ctx->saved_data["tanfovy"] = raster_settings.tanfovy_;
    ctx->saved_data["sh_degree"] = raster_settings.sh_degree_;
    ctx->saved_data["image_height"] = raster_settings.image_height_;
    ctx->saved_data["image_width"] = raster_settings.image_width_;

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


torch::autograd::tensor_list
GaussianRasterizer2DFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::tensor_list grad_outputs)
{
    auto num_rendered = ctx->saved_data["num_rendered"].toInt();
    auto scale_modifier = static_cast<float>(ctx->saved_data["scale_modifier"].toDouble());
    auto tanfovx = static_cast<float>(ctx->saved_data["tanfovx"].toDouble());
    auto tanfovy = static_cast<float>(ctx->saved_data["tanfovy"].toDouble());
    auto sh_degree = ctx->saved_data["sh_degree"].toInt();
    auto image_height = ctx->saved_data["image_height"].toInt();
    auto image_width = ctx->saved_data["image_width"].toInt();

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

    auto dL_dcolor = grad_outputs[0];
    auto dL_dout_others = grad_outputs[1];

    auto dL_ddepths = dL_dout_others.index({0, torch::indexing::Slice(), torch::indexing::Slice()});

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
        dL_dout_others,
        sh,
        sh_degree,
        campos,
        geomBuffer,
        num_rendered,
        binningBuffer,
        imgBuffer,
        false
    );

    auto dL_dmeans2D = std::get<0>(rasterization_backward_result);
    auto dL_dcolors = std::get<1>(rasterization_backward_result);
    auto dL_dopacity = std::get<2>(rasterization_backward_result);
    auto dL_dmeans3D = std::get<3>(rasterization_backward_result);
    auto dL_dtransMat = std::get<4>(rasterization_backward_result);
    auto dL_dsh = std::get<5>(rasterization_backward_result);
    auto dL_dscales = std::get<6>(rasterization_backward_result);
    auto dL_drots = std::get<7>(rasterization_backward_result);
    auto dL_dnormal = std::get<8>(rasterization_backward_result);

    return {
        dL_dmeans3D,
        dL_dmeans2D,
        dL_dsh,
        dL_dcolors,
        dL_dopacity,
        dL_dscales,
        dL_drots,
        dL_dtransMat,
        torch::Tensor()
    };
}


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

    torch::TensorOptions options;
    if (colors_precomp.numel() == 0) {
        colors_precomp = torch::tensor({}, options.device(torch::kCUDA));
    }
    if (transMat_precomp.numel() == 0) {
        transMat_precomp = torch::tensor({}, options.device(torch::kCUDA));
    }

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

    return std::make_tuple(result[0], result[1], result[2]);
}
