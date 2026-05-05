# AT-SPGD: Frequency-Adaptive Spectral Evasion for Split-Learning Inference

This repository contains the reproducibility package for the paper:

**AT-SPGD: Frequency-Adaptive Spectral Evasion for Split-Learning Inference**  
Paper link: **[review manuscript link placeholder](https://example.com)**

The codebase implements a modular split-learning inference pipeline for evaluating frequency-adaptive adversarial evasion against image backbones. It separates reusable backend components in `src/` from notebook-style execution scripts in `scripts/`, so reviewers can inspect the method implementation, reproduce the quantitative results, and regenerate the paper figures from the same code path.

## Framework Overview

The framework trains a split image classifier composed of:

- an `ImageClient` that extracts image embeddings from frozen pretrained backbones;
- a `VFLServer` that performs downstream classification from the client embedding;
- a full-model wrapper used for adversarial evaluation and visualization.

The attack pipeline evaluates standard first-order baselines and the proposed adaptive spectral attack. AT-SPGD perturbs inputs through a frequency-aware optimization path, while preserving a clean interface compatible with the split-learning inference model.

## Reproducibility Guide

We strongly recommend running the experiments on accelerator-backed environments such as Kaggle, Google Colab, or an equivalent CUDA-enabled workstation. The full benchmark uses multiple datasets, several pretrained backbone families, and a diverse set of adversarial attacks, so CPU-only execution is not practical for full reproduction.

Datasets are loaded through the official `torchvision.datasets` APIs with `download=True`. Baseline adversarial attacks are implemented through `torchattacks` where available. The only exception is SSA: `src/attacks/ssa.py` provides a PyTorch wrapper following the official Spectrum Simulation Attack release methodology for integration with this split-learning pipeline.

Recommended execution order:

1. Run `scripts/01_train_vfl.py` to train the split-learning models. This script exports model checkpoints required by the attack stage.
2. Run `scripts/02_evaluate_attacks.py` to evaluate baseline attacks and AT-SPGD. This script exports both metric tables and saved adversarial tensors for downstream analysis.
3. Run `scripts/03_visualize_results.py` to regenerate the paper-style plots and analysis tables from the saved attack artifacts.

The backend modules in `src/` define the canonical model, data, attack, evaluation, and plotting behavior. The frontend scripts call these backend modules directly, so the exported notebooks and scripts produce the same plots and tables used for reporting.

## Exported Notebooks

Reviewer-facing notebooks can be exported from the scripts and placed in the project root. Placeholder names:

- `01_train_vfl.ipynb`
- `02_evaluate_attacks.ipynb`
- `03_visualize_results.ipynb`

These notebooks are intended to mirror the corresponding scripts in `scripts/` while providing an interactive execution record for review.

## Project Structure

```text
src/
  attacks/              # AT-SPGD and SSA attack implementations
  data/                 # Dataset loading and transforms
  engine/               # Training and attack evaluation loops
  models/               # Split-learning model components
  utils/                # Metrics and evaluation utilities
  visualization/        # Publication plotting functions

scripts/
  01_train_vfl.py        # Split-model training and checkpoint export
  02_evaluate_attacks.py # Attack evaluation and artifact export
  03_visualize_results.py# Figure/table regeneration
```

## Notes for Reviewers

Generated outputs such as checkpoints, result tables, adversarial tensors, and figures are intentionally excluded from version control. They are recreated by running the scripts above in order.

Thank you for reviewing this reproducibility package.
