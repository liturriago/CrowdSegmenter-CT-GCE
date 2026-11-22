# CrowdSegmenter-CT-GCE

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
[![CI](https://img.shields.io/github/actions/workflow/status/liturriago/CrowdSegmenter-CT-GCE/ci.yml?style=for-the-badge&logo=github)](https://github.com/liturriago/CrowdSegmenter-CT-GCE/actions/workflows/ci.yml)

Verified implementation and modular pipeline for **Category-Truncated Generalized Cross-Entropy (CT-GCE)** in multi-annotator semantic segmentation.

## Repository Structure

```text
CrowdSegmenter-CT-GCE/
├── .github/workflows/ci.yml              # GitHub Actions CI running pytest
├── pyproject.toml                        # Package metadata and dependencies
├── configs/
│   ├── training/                         # Training YAML configs (ctgce, crowdseg, MV, spatial_staple, supervised)
│   └── optimization/                     # Optuna hyperparameter search configs (ctgce, crowdseg)
├── src/
│   ├── CrowdSegmenter_CT_GCE/            # Verified mathematical core (__provenance__)
│   ├── CrowdSegmenter_CT_GCE_Benchmark/    # Pilot metrics and zero-GT NLC evaluation utilities
│   └── crowdsegmenter/                   # MLOps package for data loading, training, evaluation, and visualization
│       ├── data/
│       ├── losses/
│       ├── models/
│       ├── optimization/
│       ├── training/
│       ├── utils/
│       └── visualization/
├── scripts/                              # Training, Optuna tuning, benchmarking, ablation, and visualization CLIs
│   └── pilots/                           # Auxiliary proof-of-concept scripts
├── notebooks/
│   ├── pilots/                           # Verification and pilot-study notebooks
│   └── experiments/                      # Colab notebooks (ctgce, crowdseg, baselines, supervised, benchmark, visualization)
├── results/
│   ├── benchmark/                        # CSV/JSON summary tables and per-sample metrics (Oxford-IIIT Pet, RIGA, NAIP Well Pad)
│   ├── optuna/                           # Optuna study databases, summaries, and diagnostic plots
│   └── visualizations/                   # 300-DPI sample prediction panels and Λ / confusion-matrix maps
└── tests/                                # Multi-level test suite (smoke, invariants, synthetic, audit, configs, scripts)
```

## Datasets and Methods

- **Datasets**: `oxfordiiitpet`, `riga`, `naip_wellpad`.
- **Methods**: **CT-GCE** (proposed), **CrowdSeg**, **Majority Voting (MV)**, **Spatial-STAPLE**, and **Supervised-GT** *(oracle upper bound on datasets with ground truth)*.

## Main CLI Entrypoints

- **Training**: `scripts/train_ctgce.py`, `scripts/train_crowdseg.py`, `scripts/train_supervised.py`
- **Classical Baselines**: `scripts/evaluate_baselines.py`
- **Hyperparameter Optimization (Optuna)**: `scripts/optimize_ct_gce.py`, `scripts/optimize_noisy_label.py`
- **Statistical Benchmark**: `scripts/evaluate_benchmark.py` (flexible over provided `--*_weights`, paired significance tests, and `--eval-nlc`)
- **CT-GCE Component Ablation**: `scripts/evaluate_ablation.py`
- **Visual Prediction Export (300 DPI)**: `scripts/visualize_predictions.py`

## Installation and Testing

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
