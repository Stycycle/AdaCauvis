# Layer-Adaptive Cauvis 实验说明

本文档记录我们在 SDGOD 上相对固定低频比例 Cauvis baseline 的本地改进方案。baseline 入口为 `scripts/train_baseline_ep4_bs4.sh`，改进方案入口按计划为 `scripts/train_layer_adaptive_ep12_bs4.sh`。

当前仓库中已存在同一路径的 4 epoch 脚本 `scripts/train_layer_adaptive_ep4_bs4.sh`。如果本地还没有单独的 12 epoch wrapper，可以先用下面的等价方式运行：

```bash
EPOCHS=12 bash scripts/train_layer_adaptive_ep4_bs4.sh
```

## 核心创新

原始 baseline 保留 Cauvis 模块，但低频选择策略是固定的：所有层都使用同一个低频比例 `0.20`。我们的改进目标是让频域分支更符合图像空间结构，并且让不同层可以在稳定范围内自适应选择低频信息。

### 1. 缩短视觉 prompt token 长度

改进方案统一使用：

```text
token_length = 100
```

相比更长的 prompt token 设置，`100` 个 token 更轻量，也更适合当前 640x640 的 SDGOD 实验配置。这个修改降低了 prompt 表达的冗余，避免引入过多不必要的可学习提示参数。

### 2. 从 1D FFT 改为 2D FFT

baseline 的频域路径可以退化为对 flatten 后 token 序列做 1D FFT。我们的改进显式启用：

```text
use_2d_fft = True
```

这样 Fourier 过滤是在二维 patch grid 上完成，而不是把图像 token 当成一条一维序列处理。2D FFT 保留了图像特征的空间拓扑关系，更符合目标检测中特征图的结构。

### 3. 小范围 layer-adaptive 低频比例

baseline 固定为：

```text
low_freq_ratio = 0.20
```

我们的改进学习每一层的低频比例偏移：

```text
low_freq_ratio_l = 0.20 + 0.05 * tanh(delta_l)
```

因此每层的低频比例被限制在一个较小、可解释的范围内，大致为：

```text
[0.15, 0.25]
```

这个设计和直接学习一个无约束比例不同：它允许不同层自适应，但不会让频率选择跑到过宽或不稳定的范围里。换句话说，我们不是让模型任意搜索频率，而是在 `0.20` 附近做受控自适应。

### 4. 自适应参数使用完整学习率

改进方案对新引入的频率偏移参数使用完整学习率：

```text
cauvis.aux_branch.freq_delta: lr_mult = 1.0
```

因为 backbone 通常使用较小学习率，而 `freq_delta` 是新加的轻量参数。如果它也被 backbone 的 `0.1x` 学习率压住，短 epoch 实验中可能学不充分。

## 配置差异

| 项目 | Baseline | Layer-Adaptive 改进方案 |
|---|---|---|
| 训练脚本 | `scripts/train_baseline_ep4_bs4.sh` | `scripts/train_layer_adaptive_ep12_bs4.sh` |
| 主配置 | `configs/cauvis/fixed_020_cauvis_dinov2_dinohead_bs1x4_sdgod.py` | `configs/cauvis/cauvis_dinov2_dinohead_bs1x4_sdgod.py` |
| Prompt token 数 | `100` | `100` |
| FFT 方式 | 默认频域路径 | `use_2d_fft=True` |
| 低频比例 | 固定 `0.20` | 每层自适应 `0.20 +/- 0.05` |
| 可学习频率参数 | 无 | `cauvis.aux_branch.freq_delta` |

## 复现实验

训练 baseline：

```bash
bash scripts/train_baseline_ep4_bs4.sh
```

训练改进方案，12 epoch：

```bash
bash scripts/train_layer_adaptive_ep12_bs4.sh
```

如果当前本地还没有 `train_layer_adaptive_ep12_bs4.sh`，可以用现有脚本覆盖 epoch 数：

```bash
EPOCHS=12 bash scripts/train_layer_adaptive_ep4_bs4.sh
```

当前 4 epoch 对比实验：

```bash
bash scripts/train_layer_adaptive_ep4_bs4.sh
```

## 当前 4 Epoch 结果

结果来自：

```text
work_dir/train_layer_adaptive_640_ep4_bs4/20260613_215253/20260613_215253.log
```

| 方法 | Daytime Clear | Daytime Foggy | Dusk Rainy | Night Rainy | Night-Sunny | Mean mAP |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-ratio baseline | 69.62 | 52.58 | 60.13 | 42.76 | 56.32 | 56.28 |
| Layer-Adaptive Cauvis | 67.69 | 51.43 | 56.58 | 42.38 | 54.59 | 54.53 |
| 差值 | -1.93 | -1.15 | -3.55 | -0.38 | -1.73 | -1.75 |

## 结果分析

当前 4 epoch 结果中，Layer-Adaptive Cauvis 的 mean mAP 为 `54.53`，低于 fixed-ratio baseline 的 `56.28`，差值为 `-1.75`。

分场景看，Layer-Adaptive 在所有场景上都没有超过 fixed 0.20 baseline，其中 `Dusk Rainy` 差距最大，为 `-3.55`。这说明在当前 4 epoch 设置下，小范围 layer-adaptive 低频比例还没有带来收益，固定 `0.20` 反而更稳。

需要注意的是，这个对比里 baseline 和改进方案都使用 `token_length=100`。因此当前差异主要来自：

1. 改进方案启用 2D FFT。
2. 改进方案把固定 `0.20` 改为每层受控自适应 `0.20 +/- 0.05`。
3. 新增频率偏移参数使用完整学习率。

从现有 4 epoch 结果看，固定低频比例仍然是更强的参照。后续如果要证明 layer-adaptive 有效，需要重点看 12 epoch 结果，或者进一步检查 `freq_delta` 的学习幅度是否过小、过大或被短训练周期限制。

## 备注

- 上表是 4 epoch sanity comparison，最终论文或报告结果应以 12 epoch 改进方案为准。
- 当前日志中出现过少量 `grad_norm: nan/inf`，但 loss 没有 NaN，验证正常完成。后续 12 epoch 训练建议继续观察 AMP 下的数值稳定性。
- `work_dir/` 下是运行产物，不建议提交到 git。
