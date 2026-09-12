# LIV-Surfel 回归器局部交互设计：讨论交接

## 当前认识

现有逐 Gaussian 独立回归的结构可能不足，因为同一帧新增 Gaussian 的属性并不是完全独立的。邻近 Gaussian 之间会在 scale、opacity、颜色等属性上产生覆盖、竞争和补偿关系，因此希望网络能够显式感知同一批新增候选点之间的局部关系。

当前每个关键帧大约新增 **8k 个 Gaussian candidate**。因此不倾向使用全局 self-attention：一方面远距离 Gaussian 之间通常缺乏直接交互意义，另一方面 `N≈8000` 时全局 attention 的 `O(N²)` 计算和显存负担较大。

规则分块 / window attention 虽然高效，但会人为引入块边界：空间上相邻的 Gaussian 可能因为位于不同块而无法交互。相比之下，目前更倾向于研究 **irregular sparse local self-attention**：所有当前帧新增 candidates 作为 token 一起进入网络，但每个 token 只与局部邻居进行 attention，最终仍然一次性输出当前帧全部新增 Gaussian 的属性。

形式上可理解为：

```text
all new candidate tokens
        ↓
local sparse self-attention
        ↓
per-candidate features after interaction
        ↓
shared Gaussian attribute decoder
        ↓
all new Gaussians
```

这里的重点不是“对每个中心点单独取邻域、只回归一个 Gaussian”，而是**整批 candidates 联合前向，只是 attention graph 是局部稀疏的**。

## 关于已有地图 Gaussian

当前不倾向把已有地图中的 old Gaussians 作为额外 token 加入网络。

原因是系统本身只会在当前地图 **alpha 较低、颜色误差较大等欠重建区域**补充新 Gaussian；同时下一版网络计划显式输入当前区域的重建状态，例如 rendered color、color error、rendered alpha、alpha deficit 等信息。

因此已有地图“这里已经重建到什么程度”可以通过候选生成和 reconstruction-state features 编码，没有必要再显式引入大量 old Gaussian token。这样也能避免明显增加训练与推理负担。

当前倾向的结构可以概括为：

```text
new-new local interaction
+
reconstruction-state conditioning
```

而不是：

```text
new-new interaction
+
new-old Gaussian interaction
```

## Sparse graph 的定义仍未确定

目前不预设具体 graph，应保留多个可能方向：

- image-space KNN；
- 3D KNN；
- screen-space radius graph；
- 根据潜在 Gaussian footprint overlap 构图；
- 上述信息的混合形式。

不同方案分别对应不同的“Gaussian 之间何时应该发生交互”的假设，目前还没有足够依据提前确定哪一种最好。

建议后续结合代码实现、candidate 空间分布、实际计算开销以及实验效果来判断，而不是现在人为固定 graph 定义。

## 当前倾向

现阶段最值得继续研究的方向是：

> **将原来的独立 per-Gaussian regression 改为针对约 8k 新增 candidates 的 irregular sparse local self-attention，使局部新增 Gaussian 能够共同协调属性，同时保持整批 candidate 一次前向、一次输出。**

graph 的具体定义暂时保持开放，由后续实现与实验辨析。
