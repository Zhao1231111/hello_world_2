# 回归器上下文设计：Rendered Context 与输入–输出耦合问题

## 讨论范围

本文档仅记录本轮关于 **regressor context 中是否应使用已有 Gaussian map 的渲染结果** 的讨论结论，以及用于判断 rendered context 是否真正有价值的实验思路。暂不涉及监督目标、网络规模、各 Gaussian 参数 head、数据集划分等其他问题。

## 1. 当前问题认识：这不是简单的“因果性”问题

当前 regressor 将已有 Gaussian map 的渲染结果作为输入 context 的一部分。由于当前 map 本身受到此前 regressor 输出的影响，因此存在如下反馈链：

```text
上一时刻 regressor 输出
        ↓
更新后的 Gaussian map
        ↓
当前 map render
        ↓
当前 regressor context
        ↓
新的 regressor 输出
```

因此，regressor 的输入分布不是一个固定的外部数据分布，而会随着 regressor 自身过去的输出和地图优化状态改变。

更准确地说，这是一个 **输入–输出闭环耦合 / state-distribution coupling** 问题，而不是简单要求训练输入或监督必须遵循“因果性”。

需要注意的是，“可能产生很多种甚至连续无限种 render 状态”本身不是核心问题；真正的问题在于：

- 训练时采用哪一种 reconstruction state；
- 推理过程中 regressor 实际会访问到哪些 reconstruction state；
- regressor 是否会依赖某种只在训练数据中稳定出现、但在线 rollout 中并不稳定的 rendered context。

## 2. 两种直接方案

### 方案 A：保留 rendered context，并覆盖不同优化成熟度

可以针对同一类 candidate，使用不同 reconstruction maturity 下的 render 作为输入，例如：

```text
未充分优化的 map render
少量优化后的 map render
较充分优化后的 map render
最终/高质量 map render
```

而监督信号保持不变。

这种训练方式的主要意义是让 regressor 对 map reconstruction state 的变化更鲁棒，降低它只适用于某一种“理想 render 状态”的风险。

但需要认识到：如果不同 render 状态下仍强制使用完全相同的 target，那么训练也可能促使网络逐渐忽略 rendered context。因此，这个方案不仅是候选训练方法，也可以视为一个诊断实验：

> 如果在改变 render maturity 的情况下保持同一监督，性能仍然稳定甚至提升，那么 rendered context 可能主要是 nuisance / 状态扰动，而不是不可替代的信息来源。

### 方案 B：完全去掉 rendered context

另一种思路是让 regressor 不再直接读取已有地图的 rendered RGB / rendered feature，仅依赖：

- 当前及历史 raw-image context；
- 深度、法线等几何先验；
- candidate 自身的局部几何/采样信息。

这样可以显著削弱 regressor 自身输出对下一时刻输入的反馈影响。

不过，这并不意味着整个增量系统与 current map 完全解耦。当前 candidate generation 本身仍然依赖已有地图的 rendering opacity / color error，因此 current map 仍然决定“哪里需要新增 Gaussian”。

换言之，可以将职责拆成：

```text
Current map / rendering state
        ↓
决定 where to insert

Raw multi-view + geometry
        ↓
决定 how to initialize
```

这是一种更干净的任务划分。

## 3. 当前倾向：弱耦合式 context

目前更倾向于一个介于上述两种方案之间的 **弱耦合设计**：

> 去掉高维 rendered-image feature，但保留少量、显式、物理意义清晰的 map-state descriptors。

可能的输入形式为：

```text
raw current/history image features
+ depth / inverse depth
+ normal prior
+ local sampling density / KNN spacing
+ explicit map-state descriptors
```

map-state descriptors 可以考虑：

- 当前位置的 rendered alpha / coverage；
- 当前 raw 与 render 的 color error；
- 局部已有 Gaussian density；
- 其他能够直接描述“当前区域重建程度”的低维量。

核心思想是：regressor 真正需要知道的可能不是“当前 render 图像长什么样”，而是：

> 当前 candidate 所在区域已经被重建到什么程度。

如果这个信息可以由少量显式状态量表达，就没有必要让网络从一整张由历史 regressor 输出生成的 rendered image feature 中自行推断。

该方案仍然保留 current-map conditioning，但反馈路径更弱、输入语义更稳定，也更容易分析每个状态量究竟提供了什么信息。

## 4. 需要通过实验回答的核心问题

当前不能仅凭直觉决定 rendered context 是否应保留。真正需要回答的是：

> **Rendered context 是否包含无法被 raw multi-view、geometry prior 和少量显式 map-state descriptors 替代的有效信息？**

建议围绕这一问题比较以下几类 context 设计。

| 版本 | Regresor 输入 | 主要验证的问题 |
|---|---|---|
| A. Full Render Context | Raw + 当前方案的 rendered-image features + geometry | 当前完整方案的参考 |
| B. Multi-Maturity Render | Raw + 不同优化成熟度的 rendered-image features + geometry；监督保持一致 | regressor 对不同 reconstruction state 是否鲁棒；render 是否只是状态扰动 |
| C. Weak Coupling | Raw + geometry + 少量显式 map-state descriptors | 是否只需要低维 reconstruction-state 信息，而不需要高维 render feature |
| D. No Render Context | Raw + geometry，不输入任何 map render 信息 | rendered context 是否真正必要 |

其中 B 更适合作为对反馈状态敏感性的诊断；C 是当前更倾向的实际设计；D 是判断 render 是否有不可替代价值的重要下界。

## 5. 实验控制与观察重点

为了让结论能够真正归因于 context，需要保证这些版本之间除 context 输入外其余条件尽量一致，包括：

- 相同 candidate generation；
- 相同监督信号；
- 相同 Gaussian 参数预测目标；
- 相同训练数据划分；
- 相同在线优化预算；
- 相同测试 sequence 与评价方式。

除了最终 novel-view PSNR / SSIM / LPIPS，还应特别观察：

- insertion 后 step 0 的 rendering quality；
- 固定少量优化步（如 step 4 / 8 / 16）后的质量；
- 达到同一质量阈值所需的优化步数；
- 在线 rollout 过程中性能是否随时间退化或逐渐偏离训练状态；
- 不同 reconstruction maturity 下，同一个 regressor 的预测是否明显变化。

对于 B，还应专门比较 regressor 在不同 maturity render 输入下的预测稳定性以及最终效果，以判断其是否过度依赖某种特定 map state。

## 6. 对实验结果的解释框架

不同结果可以对应不同结论：

```text
A 明显优于 C / D
→ 高维 rendered context 确实包含不可替代的信息；
  若保留该设计，需要进一步认真处理闭环状态分布问题。

C ≈ A 且 C > D
→ current reconstruction state 有价值，
  但少量显式 descriptors 已足够；
  高维 rendered-image context 没有必要。

D ≈ C ≈ A
→ regressor 基本不需要已有 map 的 render 信息；
  当前 render feedback 很可能可以从 regressor 中删除。

B 比只使用单一 render maturity 的 A 更稳定
→ 当前方法确实存在 reconstruction-state distribution sensitivity；
  对不同 map maturity 做训练覆盖有价值。

B 与 A 差异很小
→ regressor 对 render maturity 不敏感，
  需要进一步判断它是否实际使用了 rendered context。
```

## 7. 当前方法倾向

现阶段更倾向于 **弱耦合式 regressor**：

```text
candidate generation:
current map rendering
    → opacity / color error
    → 决定 where to insert

attribute regression:
raw multi-view observations
+ geometry priors
+ low-dimensional map-state descriptors
    → 决定 how to initialize
```

这一倾向目前仍属于待实验验证的设计判断，并非最终结论。Full rendered context、多成熟度 render context 和完全无 render context 都值得作为对照，以严谨确认 current-map rendering 对 attribute regression 的真实贡献。
