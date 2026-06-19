# AdaCauvis
## Layer-Adaptive Spatial-Frequency Causal Visual Prompts for Single-Source Domain Generalized Object Detection

<div align="center">
    <img src="assets/overview_of_cauvis.png">
</div>

> **TL;DR**: AdaCauvis extends [Cauvis](#acknowledgment) (Causal Visual Prompts) for single-source domain generalized object detection. We replace the fixed, flatten-token frequency path with a **2D spatial-frequency** path on the patch grid, and turn the single fixed low-frequency ratio into a **small, per-layer learnable offset** centered on `0.20`. The goal is a frequency-selection mechanism that respects image structure and can adapt per layer while staying stable under short training schedules.

> This repository is a **fork** of the original Cauvis codebase. The detection backbone, dual-branch adapter, and MMDetection scaffolding are from the upstream work; the layer-adaptive frequency changes described below are the contribution of this fork. See [Acknowledgment](#acknowledgment) for upstream credit.

---

## Table of Contents
- [Method](#method)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Train / Eval](#train--eval)
- [Results](#results)
- [Acknowledgment](#acknowledgment)

## Method

AdaCauvis keeps the Cauvis visual-prompt backbone and dual-branch adapter, and changes how the auxiliary frequency branch selects information. The three key modifications, all configured in `configs/cauvis/cauvis_dinov2_dinohead_bs1x4_sdgod.py`:

| Item | Cauvis baseline | AdaCauvis |
|---|---|---|
| Prompt token length | longer | `token_length = 100` (lighter, suited to 640×640 SDGOD) |
| FFT | 1D over flattened tokens | `use_2d_fft = True` — 2D FFT on the patch grid, preserving spatial topology |
| Low-freq ratio | fixed `0.20` | per-layer `r_l = 0.20 + 0.05·tanh(δ_l)`, bounded to `[0.15, 0.25]` |
| Frequency params | none | `cauvis.aux_branch.freq_delta` trained at full lr (`lr_mult = 1.0`) |

The bounded per-layer offset lets different layers adapt their frequency selection around `0.20` without letting the selection drift into a wide or unstable range. Full method notes, ablation rationale, and the latest experiment results are in [`README_LAYER_ADAPTIVE.md`](README_LAYER_ADAPTIVE.md).

## Installation

### Pretrained DINOv2
DINOv2 weights are primarily derived from the [Rein](https://github.com/w1oves/Rein) repository.
* **Download** pre-trained weights from [facebookresearch](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth).
* **Convert**
  ```bash
  python tools/convert_models/convert_dinov2.py checkpoints/dinov2_vitl14_pretrain.pth checkpoints/dinov2_converted_1024.pth --height 1024 --width 1024
  ```

### Environment
```bash
conda create -n adacauvis -y python=3.10
pip3 install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r ./requirements.txt
pip install albumentations==1.4.4 timm einops
pip install -U openmim
mim install mmengine
mim install mmcv==2.2.0
pip install xformers==0.0.24 # torch 2.2
pip install -v -e .
pip install numpy==1.26.0
```

## Data Preparation
Download the [SDGOD](https://github.com/AmingWu/Single-DGOD) dataset and organize it under `dataset/` as follows:
```
|-- Single-DGOD/
|   |-- Daytime_Sunny/
|   |   |-- daytime_clear/
|   |       |-- VOC2007
|   |           |-- Annotations
|   |           |-- ImageSets/Main
|   |           |-- JPEGImages
|   |-- DaytimeFoggy/
|   |-- Dusk-rainy/
|   |-- Night_rainy/
|   |-- Night-Sunny/
```

## Train / Eval

The main AdaCauvis config is `configs/cauvis/cauvis_dinov2_dinohead_bs1x4_sdgod.py`. The fixed-ratio baseline (`low_freq_ratio = 0.20`, no per-layer offset) is `configs/cauvis/fixed_020_cauvis_dinov2_dinohead_bs1x4_sdgod.py`.

**Single GPU (convenience scripts)**
```bash
# AdaCauvis (layer-adaptive)
bash scripts/train_layer_adaptive_ep4_bs4.sh    # 4-epoch sanity run
bash scripts/train_layer_adaptive_ep12_bs4.sh   # 12-epoch run

# Fixed-ratio baseline
bash scripts/train_baseline_ep4_bs4.sh
```

**Distributed**
```bash
# train
bash tools/dist_train.sh configs/cauvis/cauvis_dinov2_dinohead_bs1x4_sdgod.py 8 --amp \
  --work-dir ./work_dir/adacauvis --find_unused_parameters
# test
bash tools/dist_test.sh configs/cauvis/cauvis_dinov2_dinohead_bs1x4_sdgod.py path/to/your.pth 8 \
  --work-dir ./work_dir/test_adacauvis
```

## Results

4-epoch sanity comparison on SDGOD (mAP per scene). Both rows use `token_length = 100`; the difference is 2D FFT + per-layer adaptive ratio vs. fixed `0.20`.

| Method | Day Clear | Day Foggy | Dusk Rainy | Night Rainy | Night Clear | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-ratio baseline | 69.62 | 52.58 | 60.13 | 42.76 | 56.32 | **56.28** |
| AdaCauvis (layer-adaptive) | 67.69 | 51.43 | 56.58 | 42.38 | 54.59 | 54.53 |

At 4 epochs the fixed-ratio baseline is still the stronger reference. The layer-adaptive variant should be judged primarily on longer (12-epoch) schedules — see [`README_LAYER_ADAPTIVE.md`](README_LAYER_ADAPTIVE.md) for the up-to-date analysis. Run outputs live under `work_dir/` and are not tracked by git.

## Acknowledgment

This work builds directly on **Cauvis** (*Towards Single-Source Domain Generalized Object Detection via Causal Visual Prompts*, Chen Li et al., Huazhong University of Science and Technology). The implementation is based on [MMDetection](https://github.com/open-mmlab/mmdetection) and [Rein](https://github.com/w1oves/Rein). Thanks to their authors.
