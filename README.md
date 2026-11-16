# CrowdSegmenter-CT-GCE

Implementación verificada y pipeline modular (MLOps) para **Category-Truncated Generalized Cross-Entropy (CT-GCE)** en segmentación semántica con múltiples anotadores.

## Estructura del repositorio

```text
CrowdSegmenter-CT-GCE/
├── .github/workflows/ci.yml        # CI con pytest
├── pyproject.toml                  # Empaquetado src/ y dependencias
├── BENCHMARK.md                    # Declaración formal del protocolo de benchmark
├── configs/
│   └── training/                   # Configuraciones YAML por método/dataset
├── src/
│   ├── CrowdSegmenter_CT_GCE/      # Núcleo matemático verificado (__provenance__)
│   ├── CrowdSegmenter_CT_GCE_Benchmark/  # Harness de benchmark y métricas piloto
│   └── crowdsegmenter/             # Paquete MLOps de entrenamiento y experimentación
│       ├── data/
│       ├── losses/
│       ├── models/
│       ├── optimization/
│       ├── training/
│       └── utils/
├── scripts/
│   └── pilots/                     # Scripts auxiliares del PoC
├── notebooks/
│   ├── pilots/                     # Notebooks de verificación y estudios piloto
│   └── experiments/                # Notebooks de experimentación
├── tests/                          # Suite por niveles (smoke, invariants, synthetic, audit)
├── data/                           # Datasets locales (ignorado en git)
├── models/                         # Checkpoints (ignorado en git)
└── results/                        # Registros y salidas de experimentos (ignorado en git)
```

> Nota: El subdirectorio `CrowdSegmenter-CT-GCE/` interno conserva los registros históricos del PoC y metadatos de trazabilidad (`AGREED.md`, `Results/`, `diagrams/`).

## Instalación y pruebas

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
