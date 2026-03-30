# 修改记录&笔记



## 1月25日
1. extend()和optimize（）待修改
```cpp
#ifdef USE_2DGS
    #include "diff_surfel_rasterization_2d/renderer_2d.h"
    auto render_pkg = render_2d(viewpoint_cam, pc, bg, 1.0f);
    auto rendered_alpha = render_pkg.rendered_alpha;
#else
    auto render_pkg = render(viewpoint_cam, pc, bg, pc->apply_exposure_, true);
    auto rendered_alpha = 1 - std::get<1>(render_pkg).squeeze(0);
#endif
```
2. 正则化损失应该在 optimize() 函数 中添加，在计算主损失（L1 + SSIM）之后：
```cpp
auto loss = (1.0f - lambda_dssim) * l1_loss + lambda_dssim * dssim_loss;
#ifdef USE_2DGS
    // 2DGS 特有的正则化项
    loss = loss + lambda_distortion * depth_distortion_loss(render_pkg.rendered_distortion, ...);
    loss = loss + lambda_normal * normal_consistency_loss(render_pkg.rendered_normal, ...);
#endif
loss.backward();  // 反向传播
```

3. 如对2DGS已有输出（depth, normal, distortion）计算损失，则不需要修改 backward.cu
4. SparseGaussianAdam稀疏优化器不需要修改：在optim_utils.h中已有对visibility_mask的过滤。原理为radii>0生成的mask，2dgs已有该逻辑，不需要修改
5. 2dgs渲染器文件调用层次与作用：
```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: 高层调用 (C++ 业务层)                                   │
│  文件：gaussian.cpp -> extend() / optimize()                    │
│  - 作用：决定何时渲染，计算 Loss，触发反向传播                    │
│  - 调用：render_2d(viewpoint_cam, pc, bg_color)                 │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2: 高层渲染接口                                           │
│  文件：renderer_2d.cpp                                          │
│  - 作用：Camera/GaussianModel 对象适配，参数预处理               │
│  - 操作：准备 GaussianRasterization2DSettings，提取 2D Scales    │
│  - 调用：GaussianRasterizer2D::forward()                        │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: Autograd 封装层 (PyTorch 自动求导桥梁) ⭐               │s
│  文件：rasterizer_2d.cpp -> GaussianRasterizer2DFunction        │
│  - 作用：继承 torch::autograd::Function，连接计算图              │
│  - 操作：forward() 保存 ctx 状态；backward() 实现梯度分发        │
│  - 调用：RasterizeGaussiansCUDA()                                │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4: CUDA 桥接层                                            │
│  文件：rasterize_points.cu                                       │
│  - 作用：PyTorch Tensor 转换为原始 CUDA 指针 (float*)            │
│  - 操作：管理 GPU 显存缓冲区 (Geom/Binning/Img Buffer)           │
│  - 调用：CudaRasterizer::Rasterizer::forward()                  │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5: CUDA 核心实现                                          │
│  文件：cuda_rasterizer/rasterizer_impl.cu                        │
│  - 作用：管理具体渲染子流程                                      │
│  - 操作：Preprocessing (预处理) -> Sorting (排序) -> Rendering (混合)
│  - 调用：cuda_rasterizer/forward.cu 中的内核                      │
└─────────────────────────────────────────────────────────────────┘

loss.backward()  (在 gaussian.cpp 中触发)
        ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: [rasterizer_2d.cpp] GaussianRasterizer2DFunction::backward()
│ - 从 AutogradContext 恢复状态：相机参数、各种显存 Buffer 指针
│ - 提取梯度：dL_dcolor (颜色梯度) 和 dL_ddepths (深度梯度)
│ - 作用：梯度分发，将 PyTorch 分散的梯度汇总并交给 CUDA
└─────────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4: [rasterize_points.cu] RasterizeGaussiansBackwardCUDA()  
│ - 数据转换：将输入梯度和 Buffer 从 Tensor 转换为原始 float* 指针
│ - 参数补全：根据 2D 渲染配置补全反向传播所需的常量参数
│ - 作用：作为 PyTorch(C++) 和底层 CUDA 之间的二进制桥梁
└─────────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4.5: [cuda_rasterizer/rasterizer.h] CudaRasterizer::Rasterizer::backward()
│ - 作用：核心层入口，定义了不依赖 PyTorch 的纯指针接口
└─────────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 5: [cuda_rasterizer/rasterizer_impl.cu] Rasterizer::backward()
│ - 逻辑调度：管理反向传播的两个核心子阶段
│ - 第一步：调用 BACKWARD::render() 内核 (计算每个像素对高斯的梯度)
│ - 第二步：调用 BACKWARD::preprocess() 内核 (将像素级梯度聚合到高斯参数上)
└─────────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 5.5: [cuda_rasterizer/backward.cu] BACKWARD Kernels
│ - render(): 通过反向 alpha 混合计算 dL/dColor, dL/dNormal, dL/dOpacity
│ - preprocess(): 处理几何导数，计算 dL/dMeans3D, dL/dScales, dL/dRotations
│ - 作用：在 GPU 上执行并行的导数链式法则计算
└─────────────────────────────────────────────────────────────────┘
        ↓
梯度填入 means3D, scales, rotations, opacity, sh 等张量的 .grad() 属性中

```

> ### **⚠️ 注意**  
> 1. 核对初始化方式：knn和基于深度
> 2. 核对两次剪枝
> 3. 核对extend()
> 4. gaussian-lic的CUDA负载均衡设计

### 关于剪枝：
gaussian-lic和2dgs都有第一个剪枝操作：视锥体剔除
但是gaussianlic在preprocessCUDA()函数中的第二个剪枝操作computeTilebasedCullingTileCount()在2dgs中没有对应的操作
询问gemini回答如加上这一个剪枝，不需要修改 backward.cu：
```
即便是一个非常细长且斜向的高斯面片，即便它只经过 Tile 的一个角落，AABB 依然会为该 Tile 预留内存，这在 2DGS 中会导致较大的冗余。

2. 必要性：是否有必要添加？
非常有必要，特别是为了解决大规模/长条状高斯的内存占用问题。

- 内存优化：2DGS 的面片特性使得 AABB 往往包含大量实际上不被高斯覆盖的 Tile。加入剔除可以显著减少 point_list 的副本数量，节省显存（Buffer）。
- 加速渲染：减少了排序压力以及 render 阶段每个像素需要遍历的高斯点。
3. 反向传播：加了是否影响，是否需要修改代码？
结论：不需要修改反向传播的核心数学逻辑，但必须严格保证前向传播的一致性。

- 数学逻辑：由于剔除发生在前向传播的“分块排序”之前，它表现为离散的实例选择。反向传播是基于前向传播生成的 point_list 进行的，如果一个 Tile 被剔除了，反向传播自然不会流经它。因此，backward.cu 里的梯度计算公式不需要改。
- 工程一致性（关键）：如果你打算添加此功能，必须同步修改两个地方：
  - preprocessCUDA：计算预留的 tiles_touched 数量。
  - rasterizer_impl.cu 中的 duplicateWithKeys：执行实际的 Tile ID 写入。 两者逻辑若不一致，会导致前缀和偏移错误，进而导致显存写越界或渲染花屏。
总结建议
如果你希望提升渲染性能并减少大场景下的 OOM 概率，建议移植。但移植时需将 3DGS 基于“2x2协方差阵”的剔除逻辑，转换为 2DGS 基于“3x3单应矩阵 (Homography)”的相交逻辑。
```

### 关于gaussian.cpp的gaussian scales 初始化：
gemini:
```
1. Scale 初始化对比：深度 vs KNN
Gaussian-LIC: 基于深度的初始化 (Depth-based)
公式: $$ s = \log \left( \alpha \cdot \frac{z}{f} \right) $$

$z$: 点的深度 (Depth)
$f$: 相机焦距 (Focal Length)
$\alpha$: 缩放系数 (scaling_scale_，通常为 2)
原理: 它是利用透视投影原理。在相机成像中，一个物体在图像上的大小与深度成反比。这个公式试图让所有的初始高斯在投影到屏幕上时拥有差不多相同的像素大小（例如半径为 2-3 个像素）。离得远的点 ($z$ 大) 初始尺寸大，离得近的点初始尺寸小。

为什么 Gaussian-LIC 选这种？

SLAM / 增量式特性：Gaussian-LIC 是逐帧（或逐关键帧）添加点的。每当你添加一个新点时，你非常清楚它是从哪一帧相机的哪个像素发射出去的，因此可以直接利用该帧的深度和焦距。
计算效率 ($O(1)$)：只需要当前点的信息，不需要知道周围点的信息。这对实时 SLAM 至关重要。
2DGS (及原始 3DGS): 基于 KNN 的初始化 (Density-based)
公式:
$$ s = \log(\sqrt{\text{mean\_dist}_{knn}}) $$
原理: 它假设点云的局部密度反映了物体表面的精细程度。

KNN (K-Nearest Neighbors)：即“K 个最近邻”。对于每一个点，去整个点云里找离它最近的 K 个点（通常 K=3）。
计算流程:
对点云构建空间索引（如 KD-Tree，或在 CUDA 中暴力计算）。
对每个点 $P_i$，计算它与最近 3 个点的距离平方平均值。
开根号得到平均距离，取对数作为 $scale$。
为什么 2DGS 选这种？

全局 SfM 初始化：2DGS 通常是在训练开始前一次性初始化所有点（来自 COLMAP 计算的稀疏点云）。
防空洞 (Hole Filling)：在点云稀疏的地方，KNN 距离大，初始化大高斯来填补空隙；在点云密集的地方（细节丰富），初始化小高斯来保留细节。它只关心几何分布，而不通过特定相机的视角来决定大小。
```

## 1月26日 - gaussian.cpp 全面适配 2DGS

### 修改内容总结

#### 1. gaussian.h
- **头文件替换**: `#include "rasterizer/renderer.h"` → `#include "diff_surfel_rasterization_2d/renderer_2d.h"`

#### 2. loss_utils.h - 新增 2DGS 正则化损失函数
- `depth_to_normal()`: 从深度图计算表面法线（使用中心差分计算梯度）
- `normal_consistency_loss()`: 法线一致性损失，计算渲染法线与深度法线的余弦距离
- `distortion_loss()`: 深度畸变损失，直接对 distortion 图求均值

#### 3. gaussian.cpp - initialize()
- **Scales 初始化**: 从 KNN 改为基于深度的 2D 缩放
  - 公式: `s = log(scaling_scale_ * depth / focal)`
  - 维度: `(N, 3)` → `(N, 2)`
- **天空盒 Scales**: `repeat({1, 3})` → `repeat({1, 2})`

#### 4. gaussian.cpp - extend()
- **渲染调用**: `render()` → `render_2d()`
- **Alpha 获取**: `1 - std::get<1>(render_pkg).squeeze(0)` → `render_pkg.rendered_alpha`
- **新增高斯 Scales**: `repeat({1, 3})` → `repeat({1, 2})`

#### 5. gaussian.cpp - optimize()
- **渲染调用**: `render()` → `render_2d()`
- **渲染图像获取**: `std::get<0>(render_pkg)` → `render_pkg.rendered_image`
- **新增正则化项**:
  ```cpp
  // Distortion Loss (lambda = 0.0005)
  auto dist_loss = lambda_dist * loss_utils::distortion_loss(render_pkg.rendered_distortion);
  
  // Normal Consistency Loss (lambda = 0.0005)
  auto normal_loss = lambda_normal * loss_utils::normal_consistency_loss(
      render_pkg.rendered_normal, render_pkg.rendered_depth, render_pkg.rendered_alpha,
      fx, fy, cx, cy);
  ```
- **可见性掩码**: `std::get<3>(render_pkg)` → `render_pkg.visibility_mask`

#### 6. gaussian.cpp - evaluateVisualQuality()
- **渲染调用**: `render()` → `render_2d()`（训练视图和测试视图循环均已修改）

### 未修改的部分
- **backward.cu**: 由于损失仅使用 render_2d() 已有输出计算，梯度自动流回，无需修改
- **SparseGaussianAdam**: 已原生支持 visibility_mask，无需修改

---

## 需要注意的是：
现在加入 `法线一致性正则化` 与`深度畸变正则化`后，未详细修改权重与参与的训练轮次：
2dgs中：
- lambda_normal (法线一致性正则化): **0.05**
生效条件: **迭代次数 > 7000**
作用: 约束渲染法线（Rendered Normal）与几何表面法线（Surface Normal）的一致性，提高几何重建质量。
- lambda_dist (深度畸变正则化): **0.0(默认为关闭)** 
生效条件: **迭代次数 > 3000**
作用: 减少深度图中的伪影（floater）和畸变。
注意: 虽然代码里写了计算逻辑，但默认参数是 0，意味着必须在运行训练脚本时显式通过命令行参数（如 --lambda_dist 100）来开启它，否则这部分 loss 始终为 0。


### 新增修改1-26

在 2D Gaussian Splatting 中，为了获得更好的几何重建质量，作者引入了Ray Marching（光线步进）的概念来估计深度和法线。
1. 两种深度图
原版 2DGS (gaussian_renderer/__init__.py) 输出了两种深度：
Expected Depth (render_depth_expected): 也就是平时说的渲染深度。它是所有高斯点深度的加权平均（权重是 alpha）。
特点：平滑，但在物体边缘容易产生“鬼影”或过度平滑。
Median Depth (render_depth_median): 累积透射率达到 0.5 时的深度值。
特点：更锐利，更能代表物体表面的“硬”边界。
2. 伪表面深度 (surf_depth)
为了结合两者的优点，原版代码定义了一个混合深度：
# pythonsurf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
pipe.depth_ratio 是一个超参数。
对于有界场景 (Bounded)，通常设为 1.0 (全用 Median Depth)。
对于无界场景 (Unbounded)，通常设为 0.0 (全用 Expected Depth)。
3. 伪表面法线 (surf_normal)
这是关键差异点。原版代码不是直接使用光栅化器输出的法线图 (render_normal)，而是基于上述 surf_depth 计算出来的几何法线。
它调用了 depth_to_normal 函数：
# pythonsurf_normal = depth_to_normal(viewpoint_camera, surf_depth)surf_normal = surf_normal * (render_alpha).detach() # 还要乘以 alpha 掩码
这意味着：法线一致性损失 (normal_loss) 实际上是在约束“光栅化法线”去接近“由深度图推导出的几何法线”。
您的当前 C++ 实现 (renderer_2d.cpp)：
您直接输出了光栅化器产生的 rendered_depth (即 expected depth) 和 rendered_normal。缺失了“混合深度”和“从深度图计算法线”的步骤。
🛠️ 修改方案：在 C++ 中复刻逻辑
我们需要在 renderer_2d.cpp 中添加两个功能：
实现 depth_to_normal (从深度图计算法线)。
在 render_2d 函数中实现深度的混合和法线的计算。

## 1月29日

### 针对 2DGS/3DGS 视锥剔除与光栅化细节的技术对齐

以下是对您提出的四个技术问题的详细分析与解答，基于对 `my_gaussian_lic` 代码库的深入审查。

#### Q1: 2DGS 代码中是否有“-0.15/1.15”的拉回操作？如果缺失，应该加在哪里？

**现状确认：**  
在 `my_gaussian_lic/src/diff_surfel_rasterization_2d/cuda_rasterizer/forward.cu` 中，**没有**发现类似的拉回（Clamp）操作。  
目前的 `compute_transmat` 函数直接执行坐标变换和透视除法，没有对 NDC 坐标或视口范围进行限制。

**修改建议：**  
如果需要复刻 3DGS 的这一行为（为了防止边缘闪烁），建议在 `compute_transmat` 函数中，在透视除法（`world2ndc` 后）和视口变换（`ndc2pix` 前）之间加入限制。

**代码位置：**  
`my_gaussian_lic/src/diff_surfel_rasterization_2d/cuda_rasterizer/forward.cu` 的 `compute_transmat` 函数内部（约 L130 附近）。

**参考代码（逻辑）：**
需引入 `limx_neg, limx_pos` 等参数（需修改函数签名以传入这些值）。
```cpp
// 假设 t 是 View Space 下的坐标，或者 p_ndc 是 NDC 坐标
// 3DGS 是在 View Space 做除法时 clamp 的：
float tx = p_view.x / p_view.z;
float ty = p_view.y / p_view.z;
tx = min(limx_pos, max(limx_neg, tx));
ty = min(limy_pos, max(limy_neg, ty));
p_view.x = tx * p_view.z;
p_view.y = ty * p_view.z;
// 然后继续后续变换...
```

---

#### Q2: `in_frustum` (L226-L229) 的数学原理与具体剔除限制

**代码片段：**
```cpp
if (p_view.z <= 0.2f) return false; // 2DGS forward.cu L202 (auxiliary.h)
```

**数学原理：**
这是一个**近平面裁剪 (Near Plane Clipping)** 操作。
*   **坐标系**：`p_view` 是高斯中心在**相机坐标系 (View Space)** 下的坐标。Z 轴通常垂直于相机平面指向前方。
*   **限制**：`p_view.z <= 0.2f` 意味着任何深度（距离相机的垂直距离）小于或等于 0.2 单位的点都会被剔除。
*   **目的**：
    1.  **防止除零错误**：在透视投影中需要除以 Z，如果 Z 接近 0 或为负（在相机背后），投影会发散或错误。
    2.  **物理合理性**：剔除相机极近处或背后的物体。
*   **注意**：此函数**未**启用常规的视锥宽高剔除（如 `|x| > 1.3 * w`），这意味着只要深度合格，哪怕在屏幕很远之外的高斯点也会进入后续处理（依赖后续的 Tile Culling 或 Scissor Test 进行进一步筛选）。

---

#### Q3: 为什么在 Preprocess (L283-L289) 计算 RGB，而不是只存 SH？

**原因：性能优化与数据流设计**

1.  **视图依赖性 (View Dependency)**：球谐函数 (SH) 的核心作用是根据**观察方向**改变颜色。在每一帧渲染开始时（Preprocess 阶段），相机位置和高斯点位置是确定的，因此**观察方向**也是确定的。
2.  **计算复杂度**：计算 SH 到 RGB 涉及大量的浮点乘加运算（MADs）。
    *   **Preprocess 中计算**：每个高斯点计算 **1 次**。复杂度为 $O(N)$。
    *   **Rasterizer 中计算**：如果在光栅化阶段计算，则每个像素（或每个被该高斯覆盖的采样点）都要计算一次。复杂度为 $O(N \times \text{PixelArea})$。
3.  **结论**：在 Preprocess 阶段一次性算好当前视角的 RGB 颜色，并将其作为静态属性传给 Rasterizer，是图形学中标准的“Per-Vertex”优化策略（类似于 Gouraud Shading 与 Phong Shading 的区别）。对于高斯泼溅这种微小图元，Per-Vertex (Per-Splat) 的精度通常已足够，且性能收益巨大。

---

#### Q4: 3DGS 的 `computeTilebasedCullingTileCount` 原理及 2DGS 适配方案

**3DGS 原理 (`rasterizer/cuda_rasterizer/forward.cu:L488`)：**
该函数用于细粒度的剔除。虽然我们已经知道高斯点的 AABB 覆盖了哪些 Tile，但 AABB 是矩形的，而高斯是椭圆的。
*   **操作**：它遍历 AABB 内的每一个 Tile。
*   **核心逻辑 (`max_contrib_power_rect_gaussian_float`)**：计算高斯函数 $e^{-0.5 (x-\mu)^T \Sigma^{-1} (x-\mu)}$ 在该矩形 Tile 区域内的**最大可能值**。
*   **判断**：如果这个最大不透明度（Opacity * Max_Gaussian_Val）仍低于阈值（1/255），则说明该 Tile 实际上即使在最近点也是透明的，因此可以直接跳过（不计入 `tiles_touched`，也不生成排序键值）。这能有效剔除 AABB 四个角落的空虚区域。

**2DGS 适配方案：**

2DGS 的 Alpha 计算公式为：
$$ \alpha = \min(0.99, \text{opa} \times \exp(-0.5 \times \min(\rho_{3D}, \rho_{2D}))) $$
要剔除一个 Tile，需要证明在该 Tile 内的所有像素上，$\alpha$ 都小于阈值。
这意味着需要证明：$\min(\rho_{3D}, \rho_{2D})$ 在 Tile 内的最小值也非常大。
即：Tile 内的像素既**远离** 3D 投影形状（$\rho_{3D}$ 大），也**远离**屏幕中心点（$\rho_{2D}$ 大）。

**实现步骤：**

1.  **准备参数**：
    在 `preprocessCUDA` 中，你需要利用计算出的变换矩阵 $T$（3x3）来构建 2D 屏幕协方差矩阵。
    $$ \Sigma = T \cdot T^T $$
    $$ \text{Conic}_{3D} = \Sigma^{-1} $$ (只取 2x2 部分)

2.  **构建辅助 Conic**：
    *   **3D 形状 Conic** (`co3d`)：来自上述 $\Sigma^{-1}$。
    *   **2D 滤波 Conic** (`co2d`)：2DGS 的低通滤波项 `FilterInvSquare` 实际上是一个圆形的 Gaussian。
        对应 Conic 为：`{2.0f * FilterInvSquare, 0, 2.0f * FilterInvSquare, opacity}`。

3.  **编写适配函数**：
    可以直接复用 3DGS 的 `max_contrib_power_rect_gaussian_float`（它定义在 `forward.h` 中），但需要调用两次。

```cpp
// 2DGS 适配版 Tile 计数逻辑 (伪代码)
// 加在 preprocessCUDA 的对应位置

// 1. 计算 3D 部分的 Conic
glm::mat3 Sigma = T * glm::transpose(T);
float det = Sigma[0][0] * Sigma[1][1] - Sigma[0][1] * Sigma[1][0];
float det_inv = 1.0f / (det + 1e-6f);
float4 co3d = { 
    Sigma[1][1] * det_inv, 
    -Sigma[0][1] * det_inv, 
    Sigma[0][0] * det_inv, 
    opacities[idx] 
};

// 2. 准备 2D 部分的 Conic (圆形)
float4 co2d = { 
    2.0f * FilterInvSquare, 
    0.0f, 
    2.0f * FilterInvSquare, 
    opacities[idx] 
};

// 3. 在循环中检查
// const int tile_count_init ... (同 3DGS)
// ...
// Inside loop over tiles:
    glm::vec2 max_pos;
    // 检查 3D 投影贡献
    float power3d = max_contrib_power_rect_gaussian_float(co3d, point_image, tile_min, tile_max, max_pos);
    
    // 检查 2D 滤波贡献 (其实就是距离检查)
    float power2d = max_contrib_power_rect_gaussian_float(co2d, point_image, tile_min, tile_max, max_pos);

    // 2DGS 规则：min(rho3d, rho2d)。
    // 如果 min(power3d, power2d) < log_threshold，说明至少有一个机制让点可见。
    // 反之，如果 power3d > threshold AND power2d > threshold，则不可见。
    
    // 注意 max_contrib 返回的是 exponent (正值，越小贡献越大? 需确认函数实现)
    // 查看 max_contrib 实现：它返回的是 "evaluate_opacity_factor"，即 0.5 * d^T M d。这是一个正数。
    // 贡献度 = exp(-power)。
    // 如果 power 很大，贡献度很小。
    // 阈值是 -log(1/255) ≈ 5.54。
    // 如果 power > 5.54，则不可见。
    
    // 2DGS 可见条件：power_effective = min(power3d, power2d) < threshold.
    // 即：power3d < threshold OR power2d < threshold.
    
    if (power3d <= opacity_threshold || power2d <= opacity_threshold) {
        tile_count++;
    }
```
**注意**：2DGS 的 `compute_aabb` 已经相当精确（基于 $T$ 的解析解），通常比 3DGS 的 AABB 更紧凑。因此，引入这个 Tile Culling 的性能收益可能不如 3DGS 明显，但对于非常细长的结构（长条形 splat 斜穿 Tile），这仍然能剔除大量空 Tile。

## 2月2日
>##### 注意⚠️：深度估计网络（SPNet）出来的稠密深度图不参与 Loss 的计算监督，它仅用于初始化。

## 2月3日

### SPNet 深度补全集成到 Gaussian-LIC

#### 一、新增文件

| 文件 | 作用 |
|------|------|
| `src/spnet_wrapper.h/.cpp` | SPNet TorchScript 模型封装，处理 RGB+稀疏深度 → 稠密深度推理 |
| `src/depth_completion.h/.cpp` | 深度补全核心逻辑：稀疏深度图构建、补充点筛选、像素坐标反投影 |

#### 二、修改文件

##### 1. `gaussian.h`
- 添加 `SPNetWrapper` 前向声明
- `extend()` 函数签名新增 `std::shared_ptr<SPNetWrapper> spnet` 参数

##### 2. `gaussian.cpp`

**initialize() 修改**：
- 集成 SPNet 深度补全逻辑
- LiDAR 点 opacity = **0.5**（高可信度）
- 网络补充点 opacity = **0.1**（低可信度）
- 所有点均采用**随机旋转初始化**

**extend() 修改**（核心新增约 70 行）：
```
流程：viewpoint_cam → SPNet 深度补全 → 补充点合并到 dataset->pointcloud_
     → render_2d(pc) 生成 alpha → 联合过滤（depth 去重 + alpha 可见性）→ densify
```
- 补充点直接 push 到 `dataset->pointcloud_/pointcolor_/pointdepth_`
- `render_2d(pc)` 渲染的是**现有高斯模型**，不包含新点（逻辑正确）
- 旋转初始化从**单位四元数**改为**随机旋转**，与 initialize 保持一致

##### 3. `mapping.cpp`
- 添加 `#include "spnet_wrapper.h"`
- 从 YAML 读取 `spnet_model_path` 和 `spnet_max_depth`
- 初始化 `SPNetWrapper` 并传递给 `initialize()` 和 `extend()`

##### 4. `CMakeLists.txt`
- 添加 `spnet_wrapper.cpp` 和 `depth_completion.cpp` 到编译目标

#### 三、SPNet 配置参数

在 YAML 配置文件（如 `config/fastlivo.yaml`）中添加以下参数：

```yaml
# === SPNet 深度补全配置 ===
spnet_model_path: "/path/to/spnet_model.pt"  # TorchScript 模型路径
spnet_max_depth: 100.0                        # 最大深度值 (室外=100m, 室内=20m)
```

**参数说明**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `spnet_model_path` | string | (可选) | TorchScript 模型 `.pt` 文件路径，不填则禁用深度补全 |
| `spnet_max_depth` | float | 100.0 | 深度归一化最大值，影响网络推理精度 |

**注意**：
1. 模型文件需通过 `export_spnet.py` 导出为 TorchScript 格式
2. 输入图片大小由相机内参 `width`/`height` 决定，无需额外配置
3. 若 `spnet_model_path` 未配置或加载失败，系统正常运行但跳过深度补全

#### 四、关键设计决策

1. **深度补全仅用于初始化/扩展**：SPNet 生成的深度值**不参与 Loss 监督**
2. **区分点云可信度**：LiDAR 点 opacity 高于网络点，让梯度优先信任真实传感器数据
3. **Alpha 过滤时序**：先补全深度，后渲染 alpha → 确保只在地图覆盖不足区域添加新点
