"""CLI script for Bayesian hyperparameter optimization of ``CTGCELoss``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from crowdsegmenter.optimization.ct_gce_tuner import CTGCETuner
from scripts._cli_utils import run_optimization_cli


def run_ct_gce_optimization(
    config_path: str,
    n_trials: Optional[int] = None,
    n_epochs_per_trial: Optional[int] = None,
    output_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Optuna optimization for ``CTGCEModel`` + ``CTGCELoss``."""
    return run_optimization_cli(
        tuner_cls=CTGCETuner,
        config_path=config_path,
        n_trials=n_trials,
        n_epochs_per_trial=n_epochs_per_trial,
        output_dir=output_dir,
        label="CT-GCE",
        data_dir=data_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT-GCE Hyperparameter Optimization")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/optimization/ctgce/oxfordiiitpet.yaml",
        help="Path to the optimization YAML config.",
    )
    parser.add_argument("--data", type=str, default=None, help="Override data_dir.")
    parser.add_argument("--n_trials", type=int, default=None, help="Override n_trials.")
    parser.add_argument(
        "--n_epochs_per_trial",
        type=int,
        default=None,
        help="Override n_epochs_per_trial.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Override output_dir.")
    args = parser.parse_args()

    run_ct_gce_optimization(
        config_path=args.config,
        n_trials=args.n_trials,
        n_epochs_per_trial=args.n_epochs_per_trial,
        output_dir=args.output_dir,
        data_dir=args.data,
    )
