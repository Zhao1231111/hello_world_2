# 回归器监督信号：是否引入渲染误差

## 讨论范围

本轮只讨论回归器的监督信号，核心问题是：在现有“优化后 Gaussian 参数自蒸馏”的基础上，是否应该进一步引入多视图渲染误差作为监督。

## 当前认识

现有参数自蒸馏直接使用当前帧新增 Gaussian 在后续优化后的属性作为 teacher label。它的问题在于 Gaussian 参数本身存在一定非唯一性：不同的 scale、opacity、rotation、SH 等组合可能得到相近的最终渲染，因此直接回归某一次优化得到的参数，可能受到具体优化结果和参数分解方式的影响。

最初考虑用 renderer-space supervision 缓解这一问题：对当前帧新增的 Gaussian，在其对历史/未来视图贡献较大的像素区域上计算多视图渲染误差，并允许监督阶段使用未来视图信息，希望网络学习到更有“预测倾向”的初始化。

进一步讨论后认为，这种局部渲染监督仍存在一个更本质的问题：如果只训练当前帧新增的 Gaussian，就必须先确定“这些 Gaussian 应该负责解释哪些像素/区域”。最自然的做法是根据优化后的 teacher Gaussian 的 alpha / compositing contribution 来选择 view、crop 或 pixel mask。

但这样得到的 responsibility assignment 本身就是优化器最终形成的场景分解结果。也就是说，即使 loss 从参数空间改成了渲染空间，监督仍然默认接受了 teacher optimizer 对“哪些 Gaussian 负责哪些区域”的划分。这个划分并不一定具有唯一性：同一场景可能通过旧 Gaussian、当前新增 Gaussian、未来新增 Gaussian 之间不同的 scale、opacity、overlap 分配获得相近的渲染结果。

因此，基于 teacher contribution 的局部 renderer supervision 虽然可以缓解“参数数值必须完全一致”的问题，但不能从根本上摆脱优化结果带来的非唯一场景分解。更准确地说，它不是在模仿完整的优化轨迹，而是在继承 optimizer-induced Gaussian decomposition / responsibility assignment。

如果希望真正只以最终场景渲染质量作为监督，而不预先规定当前新增 Gaussian 的责任范围，那么需要让与其存在竞争关系的大规模 Gaussian 集合共同参与优化，极端情况下接近整场景联合渲染优化。这样才能由 rendering objective 自己决定不同 Gaussian 之间的解释分工。

但这会显著改变训练问题：训练单位不再是当前帧新增 Gaussian 的局部回归，而会变成长序列、大规模 Gaussian 集合上的联合可微优化，计算量、显存占用和训练组织复杂度都会非常高，与当前增量回归器的目标不匹配。

另外，本轮修正了一点：如果 view/crop 已经通过完整 teacher map 中当前新增 Gaussian 的实际有效贡献进行筛选，那么“单独渲染这些 Gaussian 会因为遮挡消失而产生大量错误监督”并不是主要矛盾。真正的核心问题仍然是 responsibility assignment 本身依赖 teacher optimizer。

## 当前倾向

目前倾向于**不把渲染误差作为回归器的主要监督信号**，继续以优化后 Gaussian 参数的 self-distillation 为主。

原因不是 renderer loss 本身没有意义，而是对于“只回归当前 insertion event 新增 Gaussian”的问题设定，局部 renderer loss 仍需要借助 teacher optimizer 决定监督区域，因此没有真正解决优化结果非唯一的问题；而真正摆脱这种分解所需要的联合场景级 renderer supervision 又过于昂贵。

因此，当前更合理的方向是接受 parameter-space self-distillation 这一监督形式，并进一步关注如何减小其标签歧义，例如识别和规范 Gaussian 参数中的等价表示、耦合关系和不稳定自由度，而不是为了避免参数监督而强行引入局部 rendering loss。
