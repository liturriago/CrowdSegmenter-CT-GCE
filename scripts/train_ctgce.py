"""CLI script for training and test-set evaluation of ``CTGCEModel`` (r10)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from crowdsegmenter.losses.ct_gce import CTGCELoss
from crowdsegmenter.models.ctgce import CTGCEModel
from crowdsegmenter.training.trainer import CTGCETrainer
from scripts._cli_utils import run_training_experiment


def run_ctgce_experiment(
    config_path: str,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    eval_nlc: Optional[bool] = None,
) -> Dict[str, Any]:
    """Train CT-GCE on Train/Valid and evaluate on Test."""
    return run_training_experiment(
        config_path=config_path,
        data_mode="CT-GCE",
        model_builder=lambda cfg: CTGCEModel(cfg.model),
        loss_builder=lambda cfg: CTGCELoss(
            annotators=cfg.model.num_annotators,
            classes=cfg.model.num_classes,
            ignored_value=cfg.data.ignored_value,
            q=cfg.training.ctgce_q,
        ),
        trainer_cls=CTGCETrainer,
        weights_filename="ct_gce_final.pth",
        label="CT-GCE",
        extra_summary_fn=lambda tr: {
            "rho_hat": tr.rho_hat,
            "cutoffs": tr.cutoffs,
        },
        data_dir=data_dir,
        output_dir=output_dir,
        eval_nlc=eval_nlc,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT-GCE Training & Test Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/ctgce/oxfordiiitpet.yaml",
        help="Path to the YAML experiment configuration file.",
    )
    parser.add_argument("--data", type=str, default=None, help="Override data_dir.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output_dir.")
    parser.add_argument(
        "--eval-nlc",
        "--eval_nlc",
        dest="eval_nlc",
        action="store_true",
        default=None,
        help="Compute zero-GT NLC consistency on Test.",
    )
    args = parser.parse_args()
    run_ctgce_experiment(
        args.config,
        data_dir=args.data,
        output_dir=args.output_dir,
        eval_nlc=args.eval_nlc,
    )
