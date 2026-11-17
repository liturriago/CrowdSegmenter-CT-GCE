"""CLI script for training and test-set evaluation of the Supervised GT Oracle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from crowdsegmenter.losses.supervised import SupervisedCELoss
from crowdsegmenter.models.supervised import SupervisedResUNet
from crowdsegmenter.training.trainer import SupervisedTrainer
from scripts._cli_utils import run_training_experiment


def run_supervised_experiment(
    config_path: str,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    eval_nlc: Optional[bool] = None,
) -> Dict[str, Any]:
    """Train Supervised-GT ResUNet oracle on Ground Truth and evaluate on Test."""
    return run_training_experiment(
        config_path=config_path,
        data_mode="CT-GCE",
        model_builder=lambda cfg: SupervisedResUNet(cfg.model),
        loss_builder=lambda cfg: SupervisedCELoss(
            ignored_value=cfg.data.ignored_value,
            smooth=cfg.training.smooth,
        ),
        trainer_cls=SupervisedTrainer,
        weights_filename="supervised_final.pth",
        label="Supervised-GT",
        data_dir=data_dir,
        output_dir=output_dir,
        eval_nlc=eval_nlc,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised GT Oracle Training & Test Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/supervised/oxfordiiitpet.yaml",
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
    run_supervised_experiment(
        args.config,
        data_dir=args.data,
        output_dir=args.output_dir,
        eval_nlc=args.eval_nlc,
    )
