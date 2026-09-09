#pragma once

#include <cstdint>

/**
 * @brief 2DGS 像素光栅化阶段需要生成的输出集合
 *
 * 三个值同时作为 C++ 调用协议和 CUDA 模板参数使用。数值顺序表示输出能力递增，
 * 但调用方应使用枚举名而不是依赖整数比较。
 */
enum class RenderMode2D : std::int32_t
{
    RGB_ONLY = 0,       // 只生成 RGB；用于纯 photometric 优化和特征提取
    RGB_ALPHA = 1,      // 生成 RGB 与 alpha；用于补点、位姿和视觉评估
    FULL_GEOMETRY = 2,  // 生成 RGB、alpha、深度、法线、中值深度和 distortion
};

/**
 * @brief 返回不同模式实际需要分配的辅助输出通道数
 */
constexpr std::int64_t renderAuxChannelCount2D(RenderMode2D mode)
{
    switch (mode)
    {
        case RenderMode2D::RGB_ONLY: return 0;
        case RenderMode2D::RGB_ALPHA: return 1;
        case RenderMode2D::FULL_GEOMETRY: return 7;
    }
    return 0;
}
