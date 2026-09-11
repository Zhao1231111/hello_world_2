# LIV-Surfel 开发与实验日志

> 维护方式：按时间从旧到新追加，最新记录放在文档末尾。  
> 当前分支：`RAL-resubmit`  
> 当前源码基线：`149b83c` + E013/E014 未提交修改  
> 最后更新：2026-09-11

本文档是 LIV-Surfel 的增量式项目日志。每条记录把动机、代码改动、实验结果、结论和
Git 落盘放在一起，避免代码、实验目录与结论彼此失去对应关系。论文正式描述仍以
`docs/en/root.tex` 为准。

## 当前状态速览

### 模式约定

| 模式 | `generate_dataset` | `use_Gaussian_regress` | 含义 |
| --- | ---: | ---: | --- |
| baseline | `false` | `false` | 基于规则初始化 Gaussian |
| dataset generation | `true` | `false` | 收集 self-distillation 数据 |
| regression | `false` | `true` | 使用前馈回归器初始化 Gaussian |

当前主要研究 baseline 和 dataset-generation。SPNet 分支已经废弃，旧
`densifyAndPrune` 分支已从运行代码和配置中删除。当前低 opacity Gaussian 的删除使用
通用 `gaussians->prune`，与旧致密化逻辑无关。

### 当前保留与放弃的方向

| 状态 | 方向 | 当前判断 |
| --- | --- | --- |
| 保留 | 修复 warp intrinsic 未定义行为 | 正确性修复 |
| 保留 | x/y 独立 extent | 70 秒实验中的主要加速来源 |
| 保留 | preprocess 剔除 opacity `< 1/255` | 小幅加速，原理明确 |
| 保留 | 定期删除 opacity `< 1/255` 的死亡 Gaussian | 明显减少 Gaussian，未见明显质量损失 |
| 保留 | RGB-only / RGB+alpha / full-geometry kernel | 进一步加速，指标基本不变 |
| 保留 | 原 spatial mask | S1/S2 尚未证明新设计更好 |
| 放弃 | 旧 2D tile-based culling | 有覆盖错误，且诱发更多 Gaussian |
| 放弃 | S1/S2 的 stable-sort voxel 选择 | extend 开销增加，无稳定质量收益 |
| 放弃 | SPNet candidate points | 收益有限且增加计算成本，项目已停用 |
| 暂不采用 | 简单降低 insertion alpha threshold | Gaussian 变少，但几乎不加速且损害质量 |

### 当前运行状态

当前源码为 `149b83c` 加 E013/E014 的未提交修改。2026-09-11 已使用以下命令完成 Release
编译，`devel/lib/gaussian_lic/gs_mapping` 与当前源码一致：

```bash
cd /root/catkin_gaussian
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_BUILD_TYPE=Release -j2 -l2
```

后续如果源码继续变化，仍需在实验前重新编译并记录对应 commit 或 diff。

## 日志追加模板

后续记录直接复制以下模板到文档末尾。即使实验失败，也必须保留条目并如实填写状态。

```markdown
## YYYY-MM-DD / EXXX：简短标题

- 状态：计划中 / 运行成功 / 指标完整但编排失败 / 不完整 / 已放弃
- 模式与数据：
- 基线：
- 动机或假设：
- 代码与配置改动：
- 实验结果：
- 对比与结论：
- 决策：保留 / 回退 / 待复验
- Git 落盘：commit hash + 原始标题；未提交时写明原因及 diff 保存位置
- 产物：运行目录、日志、metrics、effective config
- 备注：失败原因、可比性限制、下一步
```

---

## 日期不详 / E000：早期 HKU Campus 00 baseline 回溯

- 状态：历史结果回溯；配置由日志和保存的 YAML 推测，并非全部严格可复现。
- 模式与数据：HKU Campus 00 完整序列，baseline；各组 pose、SPNet 和初始化参数不完全
  相同。
- 基线：`baseline-hku_campus_seq_00-4-2`。
- 动机或假设：整理对话前已有的 baseline，判断亮度阈值、旧 tile culling 和 insertion
  alpha threshold 对 Gaussian 数量、速度与质量的影响。
- 代码与配置改动：本次仅回溯，不修改源码。
- 实验结果：

  | 实验 | 推测的关键配置 | 时长/s | 最终 GS | Novel PSNR | Novel SSIM | Novel LPIPS |
  | --- | --- | ---: | ---: | ---: | ---: | ---: |
  | `baseline-hku_campus_seq_00-4-2` | pose off, SPNet on, dark threshold on, no tile | 750.04 | 1,703,465 | 24.73 | 0.789 | 0.158 |
  | `noDark` | 关闭亮度相关 alpha 阈值 | 748.12 | 1,710,435 | 24.71 | 0.790 | 0.158 |
  | `noDark-tile-downsample` | 历史 2D tile culling | 767.52 | 1,888,213 | 24.79 | 0.793 | 0.150 |
  | `noDark-tile-standard` | 历史 2D tile culling | 838.32 | 2,383,824 | 24.81 | 0.792 | 0.143 |
  | `4-4-forcomparison` | pose on, SPNet on, alpha 0.99, scale 1, opacity modifier 0.2 | 680.60 | 1,315,981 | 25.44 | 0.818 | 0.157 |
  | `4-11-test` | 与上一行接近，alpha threshold 0.80 | 675.88 | 1,036,860 | 25.17 | 0.808 | 0.172 |

- 对比与结论：
  - 关闭 dark threshold 只增加 6,970 个 Gaussian（约 0.4%），时间和指标几乎不变；
  - alpha threshold 从 0.99 降至 0.80 后，Gaussian 少 279,121（约 21.2%），但时间只
    缩短约 0.7%，Novel LPIPS 恶化 0.015；
  - 旧 tile culling 存在已知覆盖错误。覆盖空洞会触发后续额外插入，导致 Gaussian
    反而更多且运行更慢；其较好的 LPIPS 不能证明剔除算法有效；
  - Gaussian 数量不能单独作为优化目标。
- 决策：dark threshold 不是当前优先方向；不采用简单降低 alpha threshold；旧 tile
  culling 后续删除。
- Git 落盘：本条为历史回溯，没有对应的新提交。
- 产物：主要 baseline 位于
  `/root/autodl-tmp/dataset/gs_rg-10/baseline-hku_campus_seq_00-4-2`；其他目录名见表。
- 备注：只有 bag、配置、frame split、优化预算和评估流程一致时，才可称为严格对照。

## 2026-09-08 / E001：SPNet 消融基础设施与完整序列实验

- 状态：实验完成；该功能随后废弃。
- 模式与数据：HKU Campus 00 完整序列，baseline；分别测试 pose refinement on/off 和
  SPNet on/off。
- 基线：不使用 SPNet 的对应 pose 配置。
- 动机或假设：检查 depth-completed candidate points 是否值得其额外计算成本。
- 代码与配置改动：增加受控 SPNet ablation 的配置、启动与记录基础设施；该提交还包含
  当时的一批项目改动，不能把整个提交都解释为单一算法因素。
- 实验结果：

  | Pose | SPNet | 时长/s | 最终 GS | Novel PSNR | Novel SSIM | Novel LPIPS |
  | --- | --- | ---: | ---: | ---: | ---: | ---: |
  | off | off | 679.31 | 1,401,031 | 24.73 | 0.788 | 0.165 |
  | off | on | 714.36 | 1,421,688 | 24.74 | 0.789 | 0.163 |
  | on | off | 725.32 | 1,379,588 | 25.63 | 0.819 | 0.158 |
  | on | on | 747.34 | 1,399,740 | 25.69 | 0.822 | 0.152 |

- 对比与结论：SPNet 带来小幅质量改善，但增加运行时间；结合项目中的其他实验，新增
  candidate points 对最终结果影响有限。
- 决策：停止继续适配 SPNet，下一条日志正式关闭该路径。
- Git 落盘：`d80be46 spnet ablation exp`。
- 产物：实验输出位于 `/root/autodl-tmp` 下对应 SPNet ablation 运行目录。
- 备注：pose on 的结果属于完整序列 smoke 对比，不应与 pose off 组交叉归因。

## 2026-09-09 / E002：确认关闭 SPNet

- 状态：改动完成并提交。
- 模式与数据：影响 baseline、dataset-generation 及相关启动配置；本条没有新增算法实验。
- 基线：E001 的 SPNet 消融结论。
- 动机或假设：SPNet 的小幅收益不足以抵消计算和维护成本。
- 代码与配置改动：在配置、launch 和脚本中关闭/移除 SPNet 默认路径。之后阅读和修改
  项目时不再考虑该分支。
- 实验结果：沿用 E001 的消融结果作为决策依据。
- 对比与结论：当前主流程不再包含 SPNet depth completion。
- 决策：保留该删除；SPNet 标记为废弃。
- Git 落盘：`70a01d4 delete spnet confimed`。
- 产物：Git 提交本身。
- 备注：提交标题中的 `confimed` 是历史拼写，本文保留原题以便检索。

## 2026-09-09 / E003：删除 2D tile culling、修复 warp UB、建立 70 秒基线

- 状态：改动提交；70 秒 baseline 成功。
- 模式与数据：baseline，HKU Campus 00 前 70 秒；pose refinement off，SPNet off。
- 基线：移除已知错误 tile culling 后的 2D 光栅器。
- 动机或假设：旧 2D tile culling 会造成错误渲染；同时 cooperative tile duplication
  中 warp intrinsic 的 mask 和位移表达式存在未定义行为，需要先恢复正确、稳定的基线。
- 代码与配置改动：
  - 完全删除试验性 2D tile culling 及参数传递；
  - 所有 `__ballot_sync` / `__shfl_sync` 使用相同的 `__activemask()`；
  - 避免 lane 0 上右移 32 位等未定义行为；
  - 新增 `bash_scripts/run_2d_rasterizer_baseline_hku70.sh`。
- 实验结果：

  | 时长/s | 最终 GS | Train LPIPS | Novel PSNR | Novel SSIM | Novel LPIPS |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 142.30 | 626,027 | 0.180 | 24.27 | 0.774 | 0.185 |

- 对比与结论：得到后续累积加速实验 E004--E007 的统一起点。旧 culling 不具备继续修复
  的价值。
- 决策：保留 UB 修复和 culling 删除。
- Git 落盘：`c56316c delete 2D tile pruning`。
- 产物：
  `/root/autodl-tmp/experiments/2d_rasterizer_baseline_hku70/20260909T044438Z`。
- 备注：这是修复正确性之后的 baseline，不应与带错误 culling 的历史结果混用。

## 2026-09-09 / E004：x/y 独立 extent

- 状态：实验成功；与后两项优化一起提交。
- 模式与数据：与 E003 相同，HKU Campus 00 前 70 秒。
- 基线：E003，142.30 s，626,027 GS，Novel LPIPS 0.185。
- 动机或假设：二维 surfel 投影常具有明显各向异性；用最大半径构造方形 bbox 会把大量
  实际不相交的 tile 送入 duplication、排序和 blending。
- 代码与配置改动：preprocess 为 x/y 两个方向分别计算保守解析 extent；为兼容既有
  接口，scalar radius 仍保留。
- 实验结果：116.79 s，629,361 GS；Train LPIPS 0.181；Novel
  PSNR/SSIM/LPIPS 为 24.28/0.775/0.186。
- 对比与结论：相对 E003 缩短约 17.9%，指标基本不变，是本轮最主要的加速来源。
- 决策：保留。
- Git 落盘：随 E006 一起落入
  `8ef4c45 2DGS accelerate: low opacity pruning; xy extent`。
- 产物：
  `/root/autodl-tmp/experiments/2d_rasterizer_accel_hku70/e1_xy_extent_complete2`。
- 备注：E004--E007 是累积实验，每条仅与紧邻的上一条直接比较。

## 2026-09-09 / E005：preprocess 剔除 opacity 小于 1/255 的 Gaussian

- 状态：实验成功；与 E004/E006 一起提交。
- 模式与数据：与 E004 相同，HKU Campus 00 前 70 秒。
- 基线：E004，116.79 s，629,361 GS，Novel LPIPS 0.186。
- 动机或假设：像素 alpha 不会大于 Gaussian 自身 opacity；若 opacity `< 1/255`，该
  Gaussian 在所有像素上都会被 kernel 的 alpha 阈值跳过，因此无需进入 tile pipeline。
- 代码与配置改动：统一 `ALPHA_THRESHOLD = 1/255`，并在 preprocess 直接剔除低
  opacity Gaussian。
- 实验结果：115.06 s，625,718 GS；Train LPIPS 0.183；Novel
  PSNR/SSIM/LPIPS 为 24.16/0.772/0.188。
- 对比与结论：获得小幅附加加速；质量有轻微波动，但不足以说明系统性退化。
- 决策：保留。
- Git 落盘：随 E006 一起落入 `8ef4c45`。
- 产物：
  `/root/autodl-tmp/experiments/2d_rasterizer_accel_hku70/e2_preprocess_opacity`。
- 备注：这一剔除不改变当前渲染定义下的像素贡献。

## 2026-09-09 / E006：删除死亡 Gaussian

- 状态：实验成功；改动提交。
- 模式与数据：与 E005 相同，HKU Campus 00 前 70 秒。
- 基线：E005，115.06 s，625,718 GS，Novel LPIPS 0.188。
- 动机或假设：opacity `< 1/255` 的 Gaussian 不参与任何像素 blending，也无法获得
  photometric、depth 或 normal 渲染梯度，在当前系统中属于死亡 Gaussian。
- 代码与配置改动：每 10 个 keyframe 和最终 evaluation 前，通过通用
  `gaussians->prune` 删除这些 Gaussian；没有启用或恢复 `densifyAndPrune`。
- 实验结果：116.72 s，579,466 GS，删除 45,893 个；Train LPIPS 0.182；Novel
  PSNR/SSIM/LPIPS 为 24.26/0.773/0.187。
- 对比与结论：最终 Gaussian 数量明显下降，质量没有明显受损；本次短序列没有体现出
  稳定的总时长收益。
- 决策：保留，在完整序列中继续观察内存和后期渲染收益。
- Git 落盘：`8ef4c45 2DGS accelerate: low opacity pruning; xy extent`。
- 产物：
  `/root/autodl-tmp/experiments/2d_rasterizer_accel_hku70/e3_dead_gaussian_prune`。
- 备注：该提交同时包含 E004、E005 和 E006，查看单项效果应使用相邻实验结果，而不能
  只比较 Git commit 前后。

## 2026-09-09 / E007：RGB-only / RGB+alpha / full-geometry kernel

- 状态：第二次实验成功并提交；第一次运行失败，已保留失败记录但不用于指标比较。
- 模式与数据：与 E006 相同，HKU Campus 00 前 70 秒。
- 基线：E006，116.72 s，579,466 GS，Novel LPIPS 0.187。
- 动机或假设：许多调用只需要 RGB 或 RGB+alpha，但原 kernel 始终计算并存储完整几何
  channel，可通过编译期专用化消除无用计算和显存流量。
- 代码与配置改动：新增 `RenderMode2D::{RGB_ONLY, RGB_ALPHA, FULL_GEOMETRY}`；调用侧按
  损失和评估需求选择模式；forward/backward 进行编译期特化，只分配需要的辅助 channel。
- 实验结果：111.83 s，579,917 GS，删除 45,815 个；Train LPIPS 0.182；Novel
  PSNR/SSIM/LPIPS 为 24.26/0.774/0.187。
- 对比与结论：相对 E006 再缩短约 4.2%，指标基本不变。
- 决策：保留。
- Git 落盘：`149b83c temp commit: kenel mode`。
- 产物：成功结果位于
  `/root/autodl-tmp/experiments/2d_rasterizer_accel_hku70/e4_kernel_modes_complete`；首次失败
  目录为同一上级目录中的 `e4_kernel_modes`。
- 备注：首次运行在 frame 4 失败且无指标，不能作为实验结果。提交标题中的 `kenel` 是
  历史拼写，本文保留原题以便检索。

## 2026-09-09 / E008：当前优化组合的 HKU Campus 00 完整序列实验

- 状态：实验成功。
- 模式与数据：HKU Campus 00 完整序列，baseline，pose refinement off，SPNet off。
- 基线：最近的 no-pose/no-SPNet 完整实验：679.31 s，1,401,031 GS，Novel LPIPS
  0.165。
- 动机或假设：验证 E003--E007 累积修改在完整序列上的总体速度与质量。
- 代码与配置改动：没有新增源码改动；运行 `149b83c`，包含删除 tile culling、warp UB
  修复、x/y extent、两类低 opacity 优化及三种专用 kernel。
- 实验结果：

  | 时长/s | 最终 GS | Pruned GS | Train PSNR | Train SSIM | Train LPIPS | Novel PSNR | Novel SSIM | Novel LPIPS |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 588.34 | 1,090,761 | 299,097 | 25.76 | 0.829 | 0.160 | 24.72 | 0.788 | 0.168 |

- 对比与结论：相对所选历史基线约快 13.4%，Gaussian 少约 22.1%，Novel LPIPS 差
  0.003。由于两次运行之间包含多项变化，这只是总体状态比较，不是单因素归因。
- 决策：保留当前优化组合，继续针对 Gaussian 插入策略做独立实验。
- Git 落盘：没有新代码提交；运行对应 `149b83c`。
- 产物：
  `/root/autodl-tmp/experiments/hku_campus_00_full-baseline/kernel_modes_20260909T075509Z`。
- 备注：不要用本条单独证明某一项优化的收益；单项结论以 E003--E007 为准。

## 2026-09-10 / E009：审视原 spatial mask 与 `extend()` 语义

- 状态：代码审视完成，没有修改源码。
- 模式与数据：仅讨论 baseline/dataset-generation。
- 基线：`149b83c` 中原有 `extend()`。
- 动机或假设：现有命名和 mask 组合复杂，需要先明确其真实语义再设计实验。
- 代码与配置改动：无。确认当前逻辑为：
  1. `rendering_mask` 选择 alpha 覆盖不足或颜色误差较大的候选；
  2. voxel mask 对当前帧候选进行体素去重；
  3. 高纹理候选绕过 voxel 限制；
  4. `if_full_regress == false` 时使用 `rendering_mask AND spatial_mask`，为 `true` 时只用
     `rendering_mask`。
- 实验结果：本条没有运行实验。
- 对比与结论：原 spatial mask 虽未显式查询地图中的三维邻域，但不能说它完全忽略现有
  地图：`rendering_mask` 已通过当前地图渲染出的 alpha 和颜色误差进行过滤，二者相交后
  保留的主要是覆盖不足或外观误差仍大的位置。现有地图是通过渲染反馈被间接考虑的。
  同时，`if_full_regress` 和 `regress_mask` 是误导性历史命名，它们也影响 baseline 和
  dataset-generation；后者会直接改变训练样本分布。
- 决策：不在缺少对照的情况下直接删除原 mask；设计 S1/S2 进行验证。
- Git 落盘：无代码变更，无提交。
- 产物：审视结论记录于本日志。
- 备注：`densifyAndPrune` 已废弃，不参与本次分析。

## 2026-09-10 / E010：Spatial mask S1——render-first + voxel 内最佳候选

- 状态：Gaussian mapping/evaluation 完整，但外部实验编排失败；代码已回退。
- 模式与数据：HKU Campus 00 前 70 秒，baseline；对比 E007。
- 基线：E007，111.83 s，579,917 GS，Novel LPIPS 0.187。
- 动机或假设：先应用 `rendering_mask`，再在真正需要插入的低纹理候选中做 voxel 去重，
  并保留 voxel 内更有价值的候选，可能比“先任意选 voxel 代表再相交”更合理。
- 代码与配置改动：高纹理候选全部保留；低纹理且需要插入的候选每个 voxel 保留一个；
  选择优先级为 coverage 不足优先，其次 alpha 更低，再其次颜色误差更高。实现使用
  stable sort。
- 实验结果：114.58 s，599,634 GS；总候选 1,277,463，插入 547,684（42.87%），删除
  48,050；Train LPIPS 0.182，Novel LPIPS 0.187；blind coverage 0.7771。
- 对比与结论：相对 E007 多 19,717 个最终 Gaussian、多插入 21,952 个、慢 2.75 s，
  没有 LPIPS 收益，blind 指标也更差。
- 决策：回退 S1，不保留。
- Git 落盘：未提交；源码已恢复。实验目录中的 `git_diff.patch` 可用于重建。
- 产物：`/root/autodl-tmp/experiments/spatial_mask_hku70/s1_render_first_best_voxel`。
- 备注：Gaussian 日志包含 `Gaussian-LIC Done!`，635 帧 render/GT 和 train/test 指标均
  完整；但 Coco-LIC 在 mapping 结束后 abort，脚本正确返回失败。因此只能把 Gaussian
  侧结果记为完整，不能把整个 run 标记为成功。

## 2026-09-10 / E011：Spatial mask S2——限制高纹理 appearance-only 候选

- 状态：两次运行均不完整；代码已回退。
- 模式与数据：HKU Campus 00 前 70 秒，baseline；对比 E007。
- 基线：E007，111.83 s，579,917 GS，Novel LPIPS 0.187。
- 动机或假设：高纹理区域仍需保留 coverage-insufficient 候选，但 alpha 已充足、仅颜色
  误差较大的 appearance-only 候选可能过密，可在每个 voxel 只保留误差最大的一个。
- 代码与配置改动：以 S1 为基础；高纹理 coverage-insufficient 候选全部保留；高纹理
  appearance-only 候选每 voxel 保留颜色误差最大者；继续使用 stable sort。
- 实验结果：

  | 运行 | 时长/s | 最终 GS | 插入数（比例） | Train LPIPS | Novel PSNR | Novel SSIM | Novel LPIPS | Blind coverage |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | S2 | 114.38 | 573,521 | 517,204（40.48%） | 0.181 | 24.25 | 0.774 | 0.186 | 0.7705 |
  | S2 retry | 115.47 | 574,904 | 519,130（40.64%） | 0.182 | 24.23 | 0.773 | 0.187 | 0.7657 |

- 对比与结论：相对 E007，最终少约 5k--6k Gaussian、少插入约 6.6k--8.5k，但慢
  2.55--3.64 s，没有稳定质量收益，blind coverage 更差。主要额外代价来自 stable
  sort：`extending` 约 42--44 s，而 E007 为 26.27 s。
- 决策：回退 S2，继续使用原 spatial mask。若以后重试，优先研究无需全局 stable sort
  的局部/并行选择，并分别统计 coverage-insufficient 与 appearance-only 插入。
- Git 落盘：未提交；源码已恢复。两个实验目录均保存 `git_diff.patch`。
- 产物：
  - `/root/autodl-tmp/experiments/spatial_mask_hku70/s2_limit_appearance_only`
  - `/root/autodl-tmp/experiments/spatial_mask_hku70/s2_limit_appearance_only_retry`
- 备注：两次均处理到最后一帧，第一轮 train/test 指标完整，但 Coco-LIC 在 bag 结束后
  `terminate called without active exception`，并在第二轮保存图像时令 Gaussian 进程被
  终止；日志没有 `Gaussian-LIC Done!`，因此必须标为不完整。脚本没有空转，能够返回并
  传播 child-process failure。

## 2026-09-10 / E012：建立增量式项目日志

- 状态：文档改动完成，尚未提交。
- 模式与数据：不涉及运行模式或数据。
- 基线：原 `docs/project_status/2026-09-10.md` 按主题拆分，不方便持续追加，也使一次
  改动、实验、结论和提交分散在不同章节。
- 动机或假设：用固定结构记录每次迭代，能够减少命名混乱和依赖记忆回溯的问题。
- 代码与配置改动：将状态文档重构为本日志；采用时间正序；每条同时记录动机、基线、
  改动、实验、结论、决策、Git 和产物；增加当前状态速览与追加模板。
- 实验结果：不涉及实验。
- 对比与结论：后续更新只需向文档末尾追加日志项，不再按“代码改动/实验/提交”三个
  主题分别维护。
- 决策：使用 `docs/project_status/experiment_log.md` 作为持续更新入口。
- Git 落盘：尚未提交；提交后应在本条补写 commit hash 和原始标题。
- 产物：`docs/project_status/experiment_log.md`。
- 备注：本文档记录的是已知事实及其可比性限制；不得把 incomplete run 改写为成功。

## 2026-09-11 / E013：删除废弃的 `densifyAndPrune` 逻辑

- 状态：代码清理完成，Release 编译通过，尚未提交。
- 模式与数据：影响共用 mapping/optimization 代码及默认配置；未运行 rosbag 实验。
- 基线：`149b83c`。旧致密化通过 `densify_from_train_times: 200000` 实际关闭，但梯度
  统计、候选帧扫描和状态维护仍会执行。
- 动机或假设：已经废弃且不会触发的致密化分支增加代码复杂度、配置负担和无效运行
  开销，应直接删除而不是继续依赖极大阈值软关闭。
- 代码与配置改动：
  - 删除 `densifyAndPrune/Clone/Split` 及其专属辅助计算；
  - 删除 densify candidate selector、共视门控、P_boost 和历史调度状态；
  - 删除 `addDensificationStats`、梯度/计数/半径统计和新生点梯度 boost；
  - 删除 `mapping.h`、`GaussianModel` 和各 YAML 中全部 densify 参数；
  - 保留 `densificationPostfix()` 的名称和正常 `extend()` 插入功能，仅移除其旧统计扩展；
  - 完整保留 `mapping.cpp::prune_dead_gaussians`、通用 `GaussianModel::prune()` 和独立的
    `if_prune_` scale-ratio prune；
  - 删除两个被 Git 跟踪的 `.ipynb_checkpoints` 旧镜像，并加入忽略规则；
  - 为 `simple_knn.h` 和 `feature_utils.h` 补充其实际使用的 CUDA 类型/API 头文件，消除
    对已移除 allocator 头文件的传递依赖。
- 实验结果：未运行数据实验。`catkin_make -DCMAKE_BUILD_TYPE=Release -j2 -l2` 成功，
  `[100%] Built target gs_mapping`。默认 `-j192` 曾因内存不足令 `cc1plus` 被杀死，不是
  源码错误。
- 对比与结论：运行代码中已无旧 densify 调用、统计或配置；优化视角列表恢复为
  P1→P2→P3。保留的两个 prune 路径和正常 Gaussian 插入接口均保持可用。
- 决策：保留本次清理；后续 baseline 不再出现 densify 配置或软关闭阈值。
- Git 落盘：尚未提交；提交后在本条补写 commit hash 和原始标题。
- 产物：当前工作区 diff；编译产物
  `/root/catkin_gaussian/devel/lib/gaussian_lic/gs_mapping`。
- 备注：本条仅验证编译与调用边界，没有宣称运行时指标通过；下一次实验应使用当前重新
  编译的二进制并保存 effective config。

## 2026-09-11 / E014：删除滞后 loss 分层并实验 P1 + 历史帧均匀随机调度

- 状态：代码修改、Release 编译和完整序列实验均成功；尚未提交。
- 模式与数据：HKU Campus 00 完整序列，baseline；pose refinement off，SPNet off，
  regressor off，`experiment_seed=20260909`。
- 基线：E008 的 full baseline，`149b83c`，588.34 s，1,090,761 GS；Train
  PSNR/SSIM/LPIPS 为 25.7556/0.8290/0.1601，Novel 为
  24.7201/0.7879/0.1676。
- 动机或假设：旧 `keyframe_loss_[idx]` 保存的是该帧上次被访问时、执行参数更新之前的
  loss，既不反映本帧刚完成的 step，也不反映其他视图随后对共享 Gaussian 的更新。
  因此 P2/P3 会固化陈旧的高/低损失身份。作为第一个可解释的替代方案，固定保留最新
  P1 滑窗，再从其余历史训练关键帧中无放回均匀随机抽样，填满 100 个视角预算。
- 代码与配置改动：
  - 删除 `keyframe_loss_`、`hiloss_threshold` 及 loss 回写；
  - P1 数量取 `min(train_camera_num, max(slide_window_size, 0), max_iters)`，先完整加入 P1；
  - 对 `[0, start_idx_p1)` 历史训练帧使用可复现随机种子洗牌，再按剩余预算截断；
  - 当前实验明确只覆盖 baseline，不把 dataset-generation 的 test cameras 放入候选池。
- 实验结果：

  | 实验 | 时长/s | 最终 GS | Train PSNR | Train SSIM | Train LPIPS | Novel PSNR | Novel SSIM | Novel LPIPS | Blind PSNR |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | E008 旧调度 | 588.34 | 1,090,761 | 25.7556 | 0.8290 | 0.1601 | 24.7201 | 0.7879 | 0.1676 | 24.8809 |
  | E014 P1 + 均匀随机 | 604.33 | 1,089,843 | 25.8147 | 0.8291 | 0.1610 | 24.7565 | 0.7876 | 0.1686 | 24.9534 |

- 对比与结论：最终 Gaussian 少 918（0.084%）。Train/Novel PSNR 分别提高
  0.0591/0.0363 dB，Novel SSIM 下降 0.00029，Train/Novel LPIPS 分别变差
  0.00090/0.00105，均属于很小的单次运行波动；没有观察到删除 P2/P3 后的明显质量退化，
  也没有证据表明简单均匀随机抽样能显著改善质量。总 mapping 时间增加 15.99 s（2.7%）；
  由于 E014 与 E008 之间还包含 E013 对废弃统计和调度代码的清理，速度差不能单独归因于
  新 opt-list 策略，但这些 E013 差异不会改变渲染指标的直接对照意义。
- 决策：保留方案 1 作为新的简单调度基线；若继续方案 2，应以 E014 为对照，仅增加一个
  明确定义的陈旧度因素。
- Git 落盘：尚未提交；实验目录保存了运行时的 `git_status.txt` 和完整
  `git_diff.patch`。提交后应补写 commit hash 和原始标题。
- 产物：
  `/root/autodl-tmp/experiments/opt_list_uniform_full/p1_uniform_seed20260909`；包含 effective
  configs、commands、Git 状态/diff、GPU 采样、完整日志和逐帧 metrics。
- 备注：Coco-LIC 在完整读完 bag 后按既有行为以 134 退出；本次编排将其单独记录，并让
  Gaussian-LIC 继续到 `Gaussian-LIC Done!`。最终状态为 completed，全部必需指标存在。
