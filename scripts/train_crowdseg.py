"""CLI script for training and test-set evaluation of ``CrowdSeg``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from crowdsegmenter.losses.noisy_label import NoisyLabelLoss
from crowdsegmenter.models.crowdseg import CrowdSeg
from crowdsegmenter.training.trainer import Trainer
from scripts._cli_utils import run_training_experiment


def run_crowdseg_experiment(
    config_path: str,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    eval_nlc: Optional[bool] = None,
) -> Dict[str, Any]:
    """Train CrowdSeg on Train/Valid and evaluate on Test."""
    return run_training_experiment(
        config_path=config_path,
        data_mode="CrowdSeg",
        model_builder=lambda cfg: CrowdSeg(cfg.model),
        loss_builder=lambda cfg: NoisyLabelLoss(
            annotators=cfg.model.num_annotators,
            classes=cfg.model.num_classes,
            ignored_value=cfg.data.ignored_value,
            min_trace=cfg.training.min_trace,
            alpha=cfg.training.alpha,
            smooth=cfg.training.smooth,
        ),
        trainer_cls=Trainer,
        weights_filename="crowdseg_final.pth",
        label="CrowdSeg",
        data_dir=data_dir,
        output_dir=output_dir,
        eval_nlc=eval_nlc,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrowdSeg Training & Test Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/crowdseg/oxfordiiitpet.yaml",
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
    run_crowdseg_experiment(
        args.config,
        data_dir=args.data,
        output_dir=args.output_dir,
        eval_nlc=args.eval_nlc,
    )
