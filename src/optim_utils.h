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

#include "diff_surfel_rasterization_2d/rasterize_points.h"
#include <unordered_map>
#include <vector>
#include <torch/torch.h>

/**
 * @brief Adam优化器的状态结构体
 * 存储每个参数张量的优化状态，包括步数和动量
 */
struct State
{
    int64_t step = 0;                ///< 当前优化步数，用于偏差校正
    torch::Tensor exp_avg;           ///< 一阶动量（梯度的指数移动平均）
    torch::Tensor exp_avg_sq;        ///< 二阶动量（梯度平方的指数移动平均）
    bool initialized = false;        ///< 状态是否已初始化的标志
};

/**
 * @brief 稀疏高斯Adam优化器的配置选项
 * 继承自PyTorch的OptimizerOptions，提供学习率和数值稳定性的配置
 */
struct SparseGaussianAdamOptions : public torch::optim::OptimizerOptions
{
public:
    /**
     * @brief 构造函数
     * @param lr 学习率，默认1e-3
     * @param eps 数值稳定性参数，用于防止除零，默认1e-8
     */
    SparseGaussianAdamOptions(double lr = 1e-3, double eps = 1e-8)
        : lr_(lr), eps_(eps) {}

    double lr_;   ///< 学习率
    double eps_;  ///< 数值稳定性参数

    /**
     * @brief 克隆配置选项
     * @return 新的配置选项对象
     */
    std::unique_ptr<OptimizerOptions> clone() const override
    {
        return std::make_unique<SparseGaussianAdamOptions>(*this);
    }

    /**
     * @brief 获取当前学习率
     * @return 当前学习率值
     */
    double get_lr() const override
    {
        return lr_;
    }

    /**
     * @brief 设置学习率
     * @param lr 新的学习率值
     */
    void set_lr(const double lr) override
    {
        lr_ = lr;
    }

    /**
     * @brief 获取数值稳定性参数
     * @return eps值
     */
    double get_eps() const
    {
        return eps_;
    }

    /**
     * @brief 设置数值稳定性参数
     * @param eps 新的eps值
     */
    void set_eps(const double eps)
    {
        eps_ = eps;
    }
};

/**
 * @brief 稀疏高斯Adam优化器
 * 专门为Gaussian Splatting设计的优化器，只更新可见的高斯点参数
 * 继承自PyTorch的Optimizer类，提供与PyTorch生态的兼容性
 */
class SparseGaussianAdam : public torch::optim::Optimizer
{
public:
    /**
     * @brief 构造函数
     * @param params 要优化的参数张量列表
     * @param lr 学习率
     * @param eps 数值稳定性参数
     */
    SparseGaussianAdam(const std::vector<torch::Tensor>& params, double lr, double eps)
        : torch::optim::Optimizer(
              {torch::optim::OptimizerParamGroup(params)},
              std::make_unique<SparseGaussianAdamOptions>(lr, eps)) {}

    /**
     * @brief 设置可见性和高斯点总数
     * 用于稀疏优化，只更新当前视角下可见的高斯点
     * @param visibility 可见性掩码张量，形状为(N,)，N为高斯点总数
     * @param N 高斯点总数
     */
    void set_visibility_and_N(const torch::Tensor& visibility, int64_t N)
    {
        visibility_ = visibility;
        N_ = N;
    }

    /**
     * @brief 执行一步优化
     * @param closure 可选的损失函数闭包，如果提供则先计算损失
     * @return 计算得到的损失值（如果提供了closure）
     */
    torch::Tensor step(LossClosure closure = nullptr) override
    {
        torch::Tensor loss;
        if (closure != nullptr)
        {
            loss = closure();
        }

        custom_step();

        return loss;
    }

    /**
     * @brief 获取优化器状态
     * @return 状态映射，每个参数张量对应一个State结构体
     */
    std::unordered_map<torch::TensorImpl*, State>& get_state()
    {
        return state_;
    }

private:
    /**
     * @brief 自定义优化步骤实现
     * 执行稀疏Adam优化算法的核心逻辑
     */
    void custom_step()
    {
        // 遍历每个参数组
        for (auto& group : param_groups_)
        {
            // 获取优化器选项
            auto& options = static_cast<SparseGaussianAdamOptions&>(group.options());
            double lr = options.get_lr();    // 学习率
            double eps = options.eps_;       // 数值稳定性参数

            // 确保每个参数组只有一个参数张量
            TORCH_CHECK(group.params().size() == 1, "More than one tensor in group");
            auto& param = group.params()[0];

            // 如果没有梯度，跳过此参数
            if (!param.grad().defined())
            {
                continue;
            }

            // 获取或初始化参数的状态
            auto& state = state_[param.unsafeGetTensorImpl()];
            if (!state.initialized)
            {
                state.step = 0;
                // 初始化动量张量，与参数张量形状相同
                state.exp_avg = torch::zeros_like(param, torch::MemoryFormat::Preserve);
                state.exp_avg_sq = torch::zeros_like(param, torch::MemoryFormat::Preserve);
                state.initialized = true;
            }

            auto& exp_avg = state.exp_avg;      // 一阶动量
            auto& exp_avg_sq = state.exp_avg_sq; // 二阶动量

            // 计算每个高斯点的参数维度 (M = 总参数数 / 高斯点数)
            int64_t M = param.numel() / N_;

            // 克隆梯度以避免修改原始梯度
            torch::Tensor grad = param.grad().clone();

            // 调用CUDA Adam更新函数
            adamUpdate(param, grad, exp_avg, exp_avg_sq, visibility_,
                    lr, 0.9, 0.999, eps, N_, M);

            // 更新步数
            state.step += 1;
        }
    }

    torch::Tensor visibility_;  ///< 可见性掩码，标记哪些高斯点在当前视角下可见
    int64_t N_;                 ///< 高斯点总数
    std::unordered_map<torch::TensorImpl*, State> state_;  ///< 参数状态映射，每个参数对应一个State
};