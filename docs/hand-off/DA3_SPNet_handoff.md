# LIV-Surfel R&R：DA3 / SPNet 取舍问题交接说明

## 1. 背景

RA-L 本轮 R&R 中，多位 Reviewer 以及 AE 都集中质疑了当前 pipeline 中 **DA3 与 SPNet 同时存在的必要性**。核心问题不是二者在实现上是否承担不同功能，而是论文当前缺少实验去证明：

- 为什么需要同时保留 SPNet 与 DA3；
- SPNet 带来的收益是否足以抵消额外的推理开销和系统复杂度；
- 当前性能提升是否主要来自 DA3，而不是 LIV-Surfel 本身的回归与初始化设计。

当前论文中的职责划分是：

- **SPNet**：沿用 Gaussian-LIC2 的做法，对稀疏 LiDAR 深度进行 depth completion，用于在 LiDAR 覆盖不足 / blind region 中补充少量 3D 点；
- **DA3**：主要利用其更强的单目结构先验产生高质量表面法向，作为 normal-guided surfel rotation 的几何先验。

从设计意图上看，二者并非完全重复：SPNet 更偏向“可靠的 metric candidate position / blind-area completion”，DA3 更偏向“高质量 surface orientation prior”。但这一区分目前主要停留在方法描述层面，缺少直接实验支撑。

## 2. 当前对 DA3 的判断

我们倾向于 **保留 DA3**。

原因是 DA3 与 LIV-Surfel 的核心方法设计结合较紧：normal-guided rotation、Gram-Schmidt surfel frame 等都直接依赖法向先验。即使 DA3 的深度在经过简单 global scale + shift / affine alignment 后仍不足以作为精确 metric geometry，它仍可能提供明显优于 SPNet depth differentiation 的局部表面方向与法向质量。

因此，对 DA3 的合理定位应当是：

> **不把 DA3 当作主要的 metric point-position source，而把它作为高质量的 local surface-normal prior。**

这一点在逻辑上并不矛盾：一个单目深度模型可以在绝对深度 / pixel-wise metric accuracy 上存在明显误差，同时仍然给出较好的局部结构与表面方向。

## 3. 当前对 SPNet 的判断

我们对 **SPNet 是否值得保留** 持开放态度，需要先通过实验决定。

SPNet 是从 Gaussian-LIC2 继承的模块，其主要作用不是生成整张地图的主体几何，而是在 LiDAR 缺失区域补充少量点。当前 LIV-Surfel 后续还存在 candidate screening，因此真正由 SPNet 生成并最终进入 map 的点可能只占全部新增 surfel 的较小比例。

因此存在两种可能：

### 情况 A：SPNet 对最终效果有明确帮助

如果移除 SPNet 后，在 LiDAR blind / under-reconstructed 区域出现明显几何空洞、渲染退化或整体指标下降，则应保留 SPNet，并在 revision 中把它定位为：

> **用于 LiDAR blind-area compensation 的 metric candidate supplementation，与 DA3 的 normal prior 形成互补。**

此时需要用实验说明二者职责不同、不可简单互相替代。

### 情况 B：SPNet 的帮助很小

如果移除 SPNet 后总体结果几乎不变，或者收益远小于其带来的推理开销和系统复杂度，则应认真考虑直接删除 SPNet。

这反而可能有利于本轮 R&R，因为可以同时缓解：

- Reviewer 对 SPNet / DA3 redundancy 的质疑；
- computational overhead；
- pipeline complexity；
- contribution attribution 不清；
- AE 关于 design choice justification 的要求。

因此，不应预设“必须保留 SPNet”，而应让实验决定。

## 4. 本轮需要优先完成的验证

**SPNet 的取舍必须先在不使用 feed-forward regression network 的 naive system 上验证。**

这里的 naive system 指当前采用 **rule-based Gaussian / surfel initialization** 的基础增量式 2DGS 系统，即不启用 LIV-Surfel 的前馈回归器。

这样做的原因是：

1. 我们当前要判断的是 **SPNet 作为 candidate-point generation / blind-area completion 模块本身是否有价值**；
2. 如果直接在完整 LIV-Surfel 系统上做比较，regressor 可能通过更好的属性初始化补偿或放大 candidate 位置差异，从而混淆 SPNet 的真实贡献；
3. naive system 更接近 Gaussian-LIC2 所采用的传统初始化范式，也更适合回答“沿用 baseline 中 SPNet 是否必要”这一问题；
4. 只有先明确 SPNet 对基础系统本身的作用，后续才有意义决定它是否应该进入最终 LIV-Surfel pipeline。

因此，本阶段的核心问题应被定义为：

> **在不使用 feed-forward regressor、其他条件尽量保持一致的 naive incremental 2DGS system 中，移除 SPNet 后，map coverage、rendering quality、geometry quality、candidate / surfel growth 以及运行开销会发生什么变化？**



## 5. Codex 需要完成的工作

请 Codex 基于当前代码仓库自行设计并执行这一验证方案。

需要满足的原则是：

- 实验主体使用 **naive / rule-based initialization system**，不要启用 feed-forward regression network；
- 核心比较围绕 **with SPNet vs. without SPNet** 展开；
- 尽量控制其他变量一致，确保差异可归因于 SPNet；
- 除总体 rendering metric 外， SPNet 实际贡献的 candidate / inserted surfel 数量，但是如何关注 LiDAR blind / under-covered 区域是否被补全是一个值得讨论的问题；
- 同时记录 SPNet 引入的时间、显存或其他系统开销，这部分仅供参考，应采用简单设计，而不是为此大费周章；
- 具体 sequence 选择、评价指标、日志格式、统计方式、可视化方式和自动化脚本，由 Codex 根据仓库现状自行设计。



## 6. 最终决策标准

本阶段不要求证明某个预设结论，而是为 revision 做取舍。

实验完成后，应根据结果在以下两条路线中选择：

### 路线 1：保留 SPNet

前提：SPNet 对 blind-area coverage、几何或渲染结果存在稳定且可解释的收益，并且收益足以支持其额外开销。

Revision 中应强调：

- SPNet 负责可靠的 metric candidate supplementation；
- DA3 负责高质量 normal prior；
- 两者功能互补，而非重复；
- 通过 ablation 证明这一设计选择。



### 路线 2：删除 SPNet

前提：SPNet 的实际贡献很小，或收益不足以抵消额外复杂度和计算成本。

Revision 中应直接简化 pipeline，并把主要几何设计重新聚焦为：

- sparse LiDAR 提供可靠 metric geometry / candidate positions；
- DA3 提供 surface-normal prior；
- LIV-Surfel regressor 负责高质量 surfel attribute initialization。

这会使论文的方法叙事更简洁，也更容易回应审稿人关于 redundancy 和 efficiency 的质疑。

## 7. 当前结论

当前暂定观点如下：

- **DA3：倾向保留。** 它承担的是高质量 normal prior，与核心 normal-guided regression 设计直接相关。
- **SPNet：不预设保留。** 它主要继承自 Gaussian-LIC2，并可能只补充少量 candidate，其实际收益需要重新确认。
- **第一步实验：在完全不使用回归网络的 naive system 中做 with / without SPNet 验证。**
- **实验细节交给 Codex 根据代码仓库自行设计。**

