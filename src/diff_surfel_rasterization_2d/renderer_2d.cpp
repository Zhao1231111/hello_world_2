/*
 * 2D Gaussian Splatting Renderer - High-Level Interface
 * 
 * 本文件仿照 Gaussian-LIC 的 renderer.cpp 编写，
 * 提供与 3DGS render() 函数完全一致的调用模式。
 */

#include "renderer_2d.h"
#include "rasterizer_2d.h"
#include "../camera.h"
#include "../gaussian.h"
#include "../loss_utils.h"
#include "../tensor_utils.h"

#include <cmath>
#include <array>

namespace
{

/**
 * @brief 构造 so(3) 的反对称矩阵
 */
Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& vec)
{
    Eigen::Matrix3d mat;
    mat << 0.0, -vec.z(), vec.y(),
           vec.z(), 0.0, -vec.x(),
          -vec.y(), vec.x(), 0.0;
    return mat;
}

/**
 * @brief 通过 Rodrigues 公式实现 \f$\mathrm{so}(3)\rightarrow \mathrm{SO}(3)\f$
 */
Eigen::Matrix3d so3Exp(const Eigen::Vector3d& omega)
{
    const double theta = omega.norm();
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    if (theta < 1e-12)
    {
        return I + skewSymmetric(omega);
    }

    const Eigen::Matrix3d K = skewSymmetric(omega / theta);
    return I + std::sin(theta) * K + (1.0 - std::cos(theta)) * K * K;
}

/**
 * @brief 基于左扰动 \f$\mathbf{T}'_{cw}=\exp(\delta\xi^\wedge)\mathbf{T}_{cw}\f$ 生成扰动相机
 */
std::shared_ptr<Camera> makePerturbedCamera(
    const std::shared_ptr<Camera>& base_camera,
    const Eigen::Matrix<double, 6, 1>& delta_xi)
{
    std::shared_ptr<Camera> perturbed_camera = std::make_shared<Camera>(*base_camera);

    const Eigen::Vector3d delta_rot = delta_xi.block<3, 1>(0, 0);
    const Eigen::Vector3d delta_pos = delta_xi.block<3, 1>(3, 0);
    const Eigen::Matrix3d dR = so3Exp(delta_rot);

    // [公式] 左扰动：R' = Exp(delta_phi) R,  t' = Exp(delta_phi) t + delta_rho
    const Eigen::Matrix3d new_R_cw = dR * base_camera->R_cw_;
    const Eigen::Vector3d new_t_cw = dR * base_camera->t_cw_ + delta_pos;
    perturbed_camera->setPoseFromCw(new_R_cw, new_t_cw);

    return perturbed_camera;
}

/**
 * @brief 计算当前帧灰度梯度掩码，用于抑制低纹理区域的位姿残差
 */
torch::Tensor buildGradientMask(const torch::Tensor& gt_image, double gradient_epsilon)
{
    auto gray_image = 0.299 * gt_image.select(0, 0)
                    + 0.587 * gt_image.select(0, 1)
                    + 0.114 * gt_image.select(0, 2);

    auto padded_gray = torch::nn::functional::pad(
        gray_image.unsqueeze(0).unsqueeze(0),
        torch::nn::functional::PadFuncOptions({1, 1, 1, 1}).mode(torch::kReplicate)).squeeze(0).squeeze(0);

    auto grad_x = 0.5 * (
        padded_gray.index({torch::indexing::Slice(1, -1), torch::indexing::Slice(2, torch::indexing::None)}) -
        padded_gray.index({torch::indexing::Slice(1, -1), torch::indexing::Slice(0, -2)}));
    auto grad_y = 0.5 * (
        padded_gray.index({torch::indexing::Slice(2, torch::indexing::None), torch::indexing::Slice(1, -1)}) -
        padded_gray.index({torch::indexing::Slice(0, -2), torch::indexing::Slice(1, -1)}));

    auto grad_norm = torch::sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12);
    return grad_norm > gradient_epsilon;
}

/**
 * @brief 计算逐像素 Huber 权重
 */
torch::Tensor buildHuberWeight(const torch::Tensor& residual_norm, double delta)
{
    if (delta <= 0.0)
    {
        return torch::ones_like(residual_norm);
    }

    auto abs_residual = residual_norm.abs();
    auto safe_abs_residual = abs_residual.clamp_min(1e-12);
    return torch::where(abs_residual <= delta,
                        torch::ones_like(abs_residual),
                        torch::full_like(abs_residual, delta) / safe_abs_residual);
}

/**
 * @brief 对二值 alpha 掩码做小半径腐蚀，去掉 render 边界的半透明过渡带
 */
torch::Tensor erodeBinaryMask(const torch::Tensor& binary_mask, int radius)
{
    if (radius <= 0)
    {
        return binary_mask;
    }

    auto mask_float = binary_mask.to(torch::kFloat32).unsqueeze(0).unsqueeze(0);
    auto inverted = 1.0f - mask_float;
    const int kernel_size = radius * 2 + 1;
    auto dilated_inverted = torch::nn::functional::max_pool2d(
        inverted,
        torch::nn::functional::MaxPool2dFuncOptions(kernel_size).stride(1).padding(radius));
    auto eroded = (1.0f - dilated_inverted).squeeze(0).squeeze(0) > 0.5f;
    return eroded;
}

}  // namespace

/**
 * @brief 2D高斯点云渲染主函数
 *
 * 将2D高斯点云从指定视角渲染成2D图像。这是2D Gaussian Splatting的核心渲染函数，
 * 完全仿照3DGS的render()函数，使用相同的调用模式。
 *
 * @param viewpoint_camera 当前视角的相机参数
 * @param pc 高斯模型，包含所有高斯点的参数
 * @param bg_color 背景颜色 (R, G, B)
 * @param scaling_modifier 缩放因子修改器，用于控制高斯大小
 * @param use_tile_culling 是否使用Tile Culling
 *
 * @return RenderResult2D 包含渲染结果和中间变量的结构体
 */
RenderResult2D render_2d(
    const std::shared_ptr<Camera>& viewpoint_camera,
    const std::shared_ptr<GaussianModel>& pc,
    const torch::Tensor& bg_color,
    float scaling_modifier,
    bool use_tile_culling,
    bool debug_mode,
    const torch::Tensor& render_mask,
    bool compute_extras
)
{
    // === 计算相机视场角参数 ===
    const float tan_fovx = std::tan(viewpoint_camera->FoVx_ * 0.5f);
    const float tan_fovy = std::tan(viewpoint_camera->FoVy_ * 0.5f);

    // === 构建2DGS光栅化设置（仿照3DGS的GaussianRasterizationSettings）===
    auto raster_settings = GaussianRasterization2DSettings(
        viewpoint_camera->image_height_,
        viewpoint_camera->image_width_,
        tan_fovx,
        tan_fovy,
        const_cast<torch::Tensor&>(bg_color),
        scaling_modifier,
        viewpoint_camera->world_view_transform_,
        viewpoint_camera->full_proj_transform_,
        pc->sh_degree_,
        viewpoint_camera->camera_center_,
        false,  // prefiltered
        use_tile_culling,  
        debug_mode   // debug
    );

    // === 创建光栅化器实例 ===
    GaussianRasterizer2D rasterizer(raster_settings);

    // === 准备高斯点参数 ===
    auto means3D = pc->getXYZ();
    
    // === 提前安全退出：如果当前完全没有高斯点，直接返回背景图像 ===
    if (means3D.size(0) == 0 || means3D.dim() < 2) {
        int H = viewpoint_camera->image_height_;
        int W = viewpoint_camera->image_width_;
        auto bg_image = bg_color.view({3, 1, 1}).expand({3, H, W}).clone();
        
        RenderResult2D empty_result;
        empty_result.rendered_image = bg_image;
        empty_result.visibility_mask = torch::zeros({0}, torch::kBool).cuda();
        empty_result.rendered_depth = torch::zeros({H, W}, torch::kFloat32).cuda();
        empty_result.rendered_alpha = torch::zeros({H, W}, torch::kFloat32).cuda();
        empty_result.rendered_normal = torch::zeros({3, H, W}, torch::kFloat32).cuda();
        empty_result.rendered_median_depth = torch::zeros({H, W}, torch::kFloat32).cuda();
        empty_result.rendered_distortion = torch::zeros({H, W}, torch::kFloat32).cuda();
        empty_result.surf_normal = torch::zeros({3, H, W}, torch::kFloat32).cuda();
        empty_result.radii = torch::zeros({0}, torch::kInt32).cuda();
        empty_result.screenspace_points = torch::zeros({0, 3}, torch::kFloat32).cuda();
        return empty_result;
    }

    auto opacity = pc->getOpacity();                // (N, 1)
    auto rotations = pc->getRotation();             // (N, 4)
    
    // 2DGS 使用 2D 缩放，从 3D 缩放中提取前两个分量
    auto scales_3d = pc->getScaling();              // (N, 3) 或 (N, 2)
    torch::Tensor scales;
    if (scales_3d.dim() > 1 && scales_3d.size(1) == 3) {
        scales = scales_3d.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}).contiguous();
    } else {
        scales = scales_3d;  // 已经是 (N, 2)
    }

    // === 准备 SH 系数 ===
    auto sh = pc->getFeatures();  // (N, K, 3) 使用新增的 getFeatures() 方法

    // === 【新增】法线朝向裁剪（可选）并与外部 render_mask 组合 ===
    torch::Tensor combined_mask;
    if (pc->enable_normal_cull_) {
        // 将局部 +Z 轴旋转到世界，再变换到相机系，获取相机系法线 z 分量
        auto qw = rotations.index({torch::indexing::Slice(), 0});
        auto qx = rotations.index({torch::indexing::Slice(), 1});
        auto qy = rotations.index({torch::indexing::Slice(), 2});
        auto qz = rotations.index({torch::indexing::Slice(), 3});

        auto nx = 2.0f * (qx * qz + qw * qy);
        auto ny = 2.0f * (qy * qz - qw * qx);
        auto nz = 1.0f - 2.0f * (qx * qx + qy * qy);
        auto normals_w = torch::stack({nx, ny, nz}, 1); // (N,3)

        auto R_cw = tensor_utils::EigenMatrix2TorchTensor(
            viewpoint_camera->R_cw_.cast<float>(), torch::kCUDA);
        auto normals_c = torch::matmul(normals_w, R_cw.t()); // (N,3)

        combined_mask = normals_c.index({torch::indexing::Slice(), 2}) <= pc->normal_z_cutoff_;
    }

    if (render_mask.defined() && render_mask.numel() > 0) {
        auto rm = render_mask.to(means3D.device(), /*non_blocking=*/true).to(torch::kBool);
        combined_mask = combined_mask.defined() ? (combined_mask & rm) : rm;
    }

    if (combined_mask.defined()) {
        if (!combined_mask.any().item<bool>()) {
            int H = viewpoint_camera->image_height_;
            int W = viewpoint_camera->image_width_;
            auto bg_image = bg_color.view({3, 1, 1}).expand({3, H, W}).clone();
            
            RenderResult2D empty_result;
            empty_result.rendered_image = bg_image;
            empty_result.visibility_mask = torch::zeros({0}, torch::kBool).cuda();
            empty_result.rendered_depth = torch::zeros({H, W}, torch::kFloat32).cuda();
            empty_result.rendered_alpha = torch::zeros({H, W}, torch::kFloat32).cuda();
            empty_result.rendered_normal = torch::zeros({3, H, W}, torch::kFloat32).cuda();
            empty_result.rendered_median_depth = torch::zeros({H, W}, torch::kFloat32).cuda();
            empty_result.rendered_distortion = torch::zeros({H, W}, torch::kFloat32).cuda();
            empty_result.surf_normal = torch::zeros({3, H, W}, torch::kFloat32).cuda();
            empty_result.radii = torch::zeros({0}, torch::kInt32).cuda();
            empty_result.screenspace_points = torch::zeros({0, 3}, torch::kFloat32).cuda();
            return empty_result;
        }

        means3D = means3D.index({combined_mask, torch::indexing::Slice()});
        opacity = opacity.index({combined_mask, torch::indexing::Slice()});
        rotations = rotations.index({combined_mask, torch::indexing::Slice()});
        scales = scales.index({combined_mask, torch::indexing::Slice()});
        sh = sh.index({combined_mask, torch::indexing::Slice(), torch::indexing::Slice()});
    }

    // === 再次安全退出：如果是局部剔除后没有任何点 ===
    if (means3D.size(0) == 0) {
        int H = viewpoint_camera->image_height_;
        int W = viewpoint_camera->image_width_;
        auto bg_image = bg_color.view({3, 1, 1}).expand({3, H, W}).clone();
        
        RenderResult2D empty_result;
        empty_result.rendered_image = bg_image;
        empty_result.visibility_mask = torch::zeros({0}, torch::kBool).cuda();
        empty_result.rendered_depth = torch::zeros({H, W}, torch::kFloat32).cuda();
        empty_result.rendered_alpha = torch::zeros({H, W}, torch::kFloat32).cuda();
        empty_result.rendered_normal = torch::zeros({3, H, W}, torch::kFloat32).cuda();
        empty_result.rendered_median_depth = torch::zeros({H, W}, torch::kFloat32).cuda();
        empty_result.rendered_distortion = torch::zeros({H, W}, torch::kFloat32).cuda();
        empty_result.surf_normal = torch::zeros({3, H, W}, torch::kFloat32).cuda();
        empty_result.radii = torch::zeros({0}, torch::kInt32).cuda();
        empty_result.screenspace_points = torch::zeros({0, 3}, torch::kFloat32).cuda();
        return empty_result;
    }

    // === 创建屏幕空间点占位符（用于梯度计算）===
    auto screenspace_points = torch::zeros_like(means3D,
        torch::TensorOptions().dtype(means3D.dtype()).requires_grad(true).device(torch::kCUDA));
    screenspace_points.retain_grad();
    
    // === 空张量占位符 ===
    torch::TensorOptions options;
    auto colors_precomp = torch::tensor({}, options.device(torch::kCUDA));
    auto transMat_precomp = torch::tensor({}, options.device(torch::kCUDA));

    // === 调用光栅化器 forward ===
    auto [rendered_image, out_others, radii] = rasterizer.forward(
        means3D,
        screenspace_points,
        opacity,
        sh,
        colors_precomp,
        scales,
        rotations,
        transMat_precomp
    );

    RenderResult2D result;
    result.rendered_image = rendered_image;
    // --- 将可见性和屏幕坐标恢复到全局尺寸，避免后端与 pc->xyz_ 尺寸不一致 ---
    if (combined_mask.defined()) {
        auto N_full = pc->getXYZ().size(0);
        auto full_radii = torch::zeros({N_full}, radii.options());
        auto full_screenspace = torch::zeros({N_full, 3}, screenspace_points.options()).requires_grad_(true);

        full_radii.index_put_({combined_mask}, radii);
        full_screenspace = full_screenspace.index_put({combined_mask}, screenspace_points, /*accumulate=*/false);

        result.radii = full_radii;
        result.visibility_mask = full_radii > 0;
        result.screenspace_points = full_screenspace;
        result.screenspace_points_mask = combined_mask;
        result.screenspace_points_local = screenspace_points;
    } else {
        result.radii = radii;
        result.visibility_mask = radii > 0;
        result.screenspace_points = screenspace_points;
    }
    if (result.screenspace_points.requires_grad()) {
        result.screenspace_points.retain_grad();
    }

    if (compute_extras) {
        // === 从 out_others 中提取各个辅助输出 ===
        // 1. 期望深度 (Expected Depth)
        auto render_depth_expected = out_others.index({0, torch::indexing::Slice(), torch::indexing::Slice()});
        
        // 2. Alpha 通道
        auto rendered_alpha = out_others.index({1, torch::indexing::Slice(), torch::indexing::Slice()});
        
        // 3. 渲染法线 (Rendered Normal)
        auto rendered_normal_view = out_others.index({torch::indexing::Slice(2, 5), torch::indexing::Slice(), torch::indexing::Slice()});
        auto R_cw = viewpoint_camera->world_view_transform_.index({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)});
        auto rendered_normal = torch::matmul(R_cw.t(), rendered_normal_view.reshape({3, -1})).reshape({3, viewpoint_camera->image_height_, viewpoint_camera->image_width_});

        // 4. 中值深度 (Median Depth)
        auto render_depth_median = out_others.index({5, torch::indexing::Slice(), torch::indexing::Slice()});
        render_depth_median = torch::nan_to_num(render_depth_median, 0.0, 0.0);
        
        // 5. 畸变图
        auto rendered_distortion = out_others.index({6, torch::indexing::Slice(), torch::indexing::Slice()});

        // === 【新增】伪表面属性计算 (复刻 2DGS 逻辑) ===
        render_depth_expected = render_depth_expected / (rendered_alpha + 1e-8);
        render_depth_expected = torch::nan_to_num(render_depth_expected, 0.0, 0.0);
        
        float depth_ratio = 1.0f; // Bounded scene default
        auto surf_depth = render_depth_expected * (1.0f - depth_ratio) + render_depth_median * depth_ratio;
        
        auto surf_normal = loss_utils::depth_to_normal(
            surf_depth, 
            viewpoint_camera->fx_, 
            viewpoint_camera->fy_, 
            viewpoint_camera->cx_, 
            viewpoint_camera->cy_
        );
        surf_normal = surf_normal * rendered_alpha.detach();

        result.rendered_depth = surf_depth;   
        result.rendered_alpha = rendered_alpha;
        result.rendered_normal = rendered_normal; 
        result.surf_normal = surf_normal;     
        
        result.rendered_median_depth = render_depth_median;
        result.rendered_distortion = rendered_distortion;
    }

    return result;
}

PoseLinearization2D linearize_pose_2d(
    const std::shared_ptr<Camera>& viewpoint_camera,
    const std::shared_ptr<GaussianModel>& pc,
    const torch::Tensor& gt_image,
    const torch::Tensor& bg_color,
    const RenderResult2D& base_render,
    const PoseLinearizationSettings2D& settings)
{
    torch::NoGradGuard no_grad;

    PoseLinearization2D result;

    // [公式] 当前帧残差：r_p = I_p - \hat{I}_p(\hat{\xi})
    auto gt_image_cuda = gt_image.to(torch::kCUDA, /*non_blocking=*/true).contiguous();
    auto residual_rgb = (gt_image_cuda - base_render.rendered_image).contiguous();
    auto alpha_mask_raw = (base_render.rendered_alpha > settings.alpha_threshold);
    auto alpha_mask = erodeBinaryMask(alpha_mask_raw, settings.alpha_erode_radius);
    auto gradient_mask = buildGradientMask(gt_image_cuda, settings.gradient_epsilon);
    auto residual_norm = torch::sqrt(torch::sum(residual_rgb * residual_rgb, 0) + 1e-12);
    auto huber_weight = buildHuberWeight(residual_norm, settings.huber_delta);

    // [公式] w_p = mask_alpha * mask_grad * huber_weight
    auto pixel_weight = alpha_mask.to(torch::kFloat32) *
                        gradient_mask.to(torch::kFloat32) *
                        huber_weight.to(torch::kFloat32);
    auto valid_mask = pixel_weight > 0.0f;

    result.alpha_pixels = alpha_mask.sum().item<int>();
    result.alpha_coverage_ratio =
        alpha_mask.numel() > 0 ? static_cast<double>(result.alpha_pixels) / static_cast<double>(alpha_mask.numel()) : 0.0;
    result.valid_pixels = valid_mask.sum().item<int>();
    result.valid_mask = valid_mask;
    result.residual_rgb = residual_rgb;

    if (result.valid_pixels <= 0)
    {
        result.jtj = torch::zeros({6, 6}, torch::TensorOptions().dtype(torch::kFloat64));
        result.jtr = torch::zeros({6}, torch::TensorOptions().dtype(torch::kFloat64));
        result.rmse = 0.0;
        return result;
    }

    auto residual_flat = residual_rgb.permute({1, 2, 0}).reshape({-1}).to(torch::kFloat64);
    auto weight_flat = pixel_weight.reshape({-1, 1}).repeat({1, 3}).reshape({-1}).to(torch::kFloat64);

    std::array<torch::Tensor, 6> jacobian_columns;
    for (int axis = 0; axis < 6; ++axis)
    {
        const double step = axis < 3 ? settings.rot_epsilon : settings.trans_epsilon;
        Eigen::Matrix<double, 6, 1> plus_delta = Eigen::Matrix<double, 6, 1>::Zero();
        plus_delta(axis) = step;

        torch::Tensor jacobian_rgb;
        if (settings.use_central_difference)
        {
            Eigen::Matrix<double, 6, 1> minus_delta = Eigen::Matrix<double, 6, 1>::Zero();
            minus_delta(axis) = -step;

            auto plus_camera = makePerturbedCamera(viewpoint_camera, plus_delta);
            auto minus_camera = makePerturbedCamera(viewpoint_camera, minus_delta);

            auto render_plus = render_2d(plus_camera, pc, bg_color, 1.0f, false, false, torch::Tensor(), false);
            auto render_minus = render_2d(minus_camera, pc, bg_color, 1.0f, false, false, torch::Tensor(), false);

            // [公式] J_k \approx (I(x+\epsilon e_k) - I(x-\epsilon e_k)) / (2\epsilon)
            jacobian_rgb = (render_plus.rendered_image - render_minus.rendered_image) / (2.0 * step);
        }
        else
        {
            auto plus_camera = makePerturbedCamera(viewpoint_camera, plus_delta);
            auto render_plus = render_2d(plus_camera, pc, bg_color, 1.0f, false, false, torch::Tensor(), false);

            // [公式] J_k \approx (I(x+\epsilon e_k) - I(x)) / \epsilon
            jacobian_rgb = (render_plus.rendered_image - base_render.rendered_image) / step;
        }

        jacobian_columns[axis] = jacobian_rgb.permute({1, 2, 0}).reshape({-1}).to(torch::kFloat64);
    }

    auto jacobian = torch::stack(jacobian_columns, 1);  // (3*H*W, 6)
    auto weighted_jacobian = jacobian * torch::sqrt(weight_flat).unsqueeze(1);
    auto weighted_residual = residual_flat * torch::sqrt(weight_flat);

    // [公式] JTJ = Σ J_p^T W_p J_p,  JTr = Σ J_p^T W_p r_p
    result.jtj = weighted_jacobian.transpose(0, 1).matmul(weighted_jacobian).cpu();
    result.jtr = weighted_jacobian.transpose(0, 1).matmul(weighted_residual.unsqueeze(1)).squeeze(1).cpu();

    auto weighted_residual_sq = (weight_flat * residual_flat * residual_flat).sum().item<double>();
    auto total_weight = weight_flat.sum().item<double>();
    result.rmse = total_weight > 0.0 ? std::sqrt(weighted_residual_sq / total_weight) : 0.0;

    return result;
}
