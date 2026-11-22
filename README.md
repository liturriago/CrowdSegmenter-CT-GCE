# CrowdSegmenter-CT-GCE

Implementación verificada y pipeline modular para **Category-Truncated Generalized Cross-Entropy (CT-GCE)** en segmentación semántica con múltiples anotadores.

## Estructura del repositorio

```text
CrowdSegmenter-CT-GCE/
├── .github/workflows/ci.yml              # CI con pytest
├── pyproject.toml                        # Empaquetado src/ y dependencias
├── configs/
│   ├── training/                         # Configuraciones YAML (ctgce, crowdseg, MV, spatial_staple, supervised)
│   └── optimization/                     # Espacios de búsqueda Optuna (ctgce, crowdseg)
├── src/
│   ├── CrowdSegmenter_CT_GCE/            # Núcleo matemático verificado (__provenance__)
│   ├── CrowdSegmenter_CT_GCE_Benchmark/    # Harness de métricas piloto y evaluación NLC
│   └── crowdsegmenter/                   # Paquete MLOps de entrenamiento, evaluación y visualización
│       ├── data/
│       ├── losses/
│       ├── models/
│       ├── optimization/
│       ├── training/
│       ├── utils/
│       └── visualization/
├── scripts/                              # Entrenamiento, Optuna, benchmark, ablación y visualización
│   └── pilots/                           # Scripts auxiliares del PoC
├── notebooks/
│   ├── pilots/                           # Notebooks de verificación y estudios piloto
│   └── experiments/                      # Notebooks Colab (ctgce, crowdseg, baselines, supervised, benchmark, visualization)
├── results/
│   ├── benchmark/                        # Tablas CSV/JSON y métricas por muestra (Oxford-IIIT Pet, RIGA, NAIP Well Pad)
│   ├── optuna/                           # Estudios y gráficas de búsqueda de hiperparámetros
│   └── visualizations/                   # Paneles y mapas Λ/CM a 300 DPI organizados por muestra
└── tests/                                # Suite de pruebas (smoke, invariants, synthetic, audit, configs, scripts)
```

## Métodos y datasets incluidos

- **Datasets**: `oxfordiiitpet`, `riga`, `naip_wellpad`.
- **Métodos evaluados**: **CT-GCE** (propuesto), **CrowdSeg**, **Majority Voting (MV)**, **Spatial-STAPLE** y **Supervised-GT** *(oráculo cuando hay Ground Truth)*.

## Scripts principales

- **Entrenamiento**: `scripts/train_ctgce.py`, `scripts/train_crowdseg.py`, `scripts/train_supervised.py`
- **Baselines clásicos**: `scripts/evaluate_baselines.py`
- **Optimización de hiperparámetros (Optuna)**: `scripts/optimize_ct_gce.py`, `scripts/optimize_noisy_label.py`
- **Benchmark estadístico**: `scripts/evaluate_benchmark.py` (flexible en los `--*_weights` proporcionados, con pruebas pareadas e integración `--eval-nlc`)
- **Ablación CT-GCE**: `scripts/evaluate_ablation.py`
- **Visualización de predicciones (300 DPI)**: `scripts/visualize_predictions.py`

## Instalación y pruebas

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
