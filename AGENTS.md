# Repository Guidelines

## Project Structure & Module Organization

This repository is a Cauvis/MMDetection-style Python project. Core package code lives in `mmdet/`, with model components under `mmdet/models/`, dataset and evaluation code under `mmdet/datasets/` and `mmdet/evaluation/`, and inference APIs under `mmdet/apis/`. Experiment configs are in `configs/`, especially `configs/cauvis/`. Training and evaluation entry points are in `tools/`. Paper figures and README media are in `assets/`; published logs are in `resources/`. Keep large runtime outputs in `work_dir/`, `weights/`, or `checkpoints/` rather than mixing them with source code.

## Build, Test, and Development Commands

- `pip install -r requirements.txt`: install runtime, build, and optional dependencies listed by the project.
- `pip install -v -e .`: install the local `mmdet` package in editable mode.
- `python tools/test.py configs/cauvis/cauvis_dinov2_dinohead_bs1x4_sdgod.py weights/cauvis_dinohead.pth`: run single-GPU evaluation for the SDGOD Cauvis config.
- `bash tools/dist_train.sh configs/cauvis/cauvis_dinov2_dinohead_bs1x4_sdgod.py 8 --amp --work-dir ./work_dir/cauvis --find_unused_parameters`: launch distributed training.
- `bash tools/dist_test.sh <config.py> <checkpoint.pth> 8 --work-dir ./work_dir/test_cauvis`: launch distributed evaluation.

## Coding Style & Naming Conventions

Follow the existing OpenMMLab Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and config filenames that describe model, backbone, head, batch size, and dataset. Prefer registry-based integration patterns already used in `mmdet/models/`. Keep imports grouped standard library, third-party, then local. Test requirements include `flake8`, `isort==4.3.21`, and `yapf`; use them when touching broad areas.

## Testing Guidelines

There is no populated top-level test suite in this checkout, but pytest is listed in `requirements/tests.txt`. Add focused `test_*.py` files near the relevant tool or package area when adding behavior. For model or config changes, validate with the smallest practical command first, then run the relevant training or evaluation command and store generated logs under `work_dir/`.

## Commit & Pull Request Guidelines

Recent history uses short imperative messages such as `update README.md`, `modify logs`, and `submit code`. Keep commits concise but more specific when possible, for example `add cauvis backbone config` or `fix robustness eval path`. Pull requests should include the affected configs/modules, exact commands run, dataset/checkpoint assumptions, and key metrics or log paths. Include screenshots only for documentation or visualization changes.

## Security & Configuration Tips

Do not commit private datasets, credentials, or machine-specific absolute paths. Prefer relative paths in configs and README examples. Keep pretrained weights in `weights/` or `checkpoints/` and document external download locations instead of adding new large binaries.
