"""CLI script for training and test-set evaluation of ``AnnotHarmony``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from crowdsegmenter.losses.tgce_ssps import TGCE_SSPS
from crowdsegmenter.models.annot_harmony import AnnotHarmony
from crowdsegmenter.training.trainer import Trainer
from scripts._cli_utils import run_training_experiment


def run_annotharmony_experiment(
    config_path: str,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Train AnnotHarmony on Train/Valid and evaluate on Test."""
    return run_training_experiment(
        config_path=config_path,
        data_mode="Annot-Harmony",
        model_builder=lambda cfg: AnnotHarmony(cfg.model),
        loss_builder=lambda cfg: TGCE_SSPS(
            annotators=cfg.model.num_annotators,
            classes=cfg.model.num_classes,
            ignored_value=cfg.data.ignored_value,
            q=cfg.training.tgce_q,
            lambda_factor=cfg.training.tgce_lambda,
            smooth=cfg.training.smooth,
        ),
        trainer_cls=Trainer,
        weights_filename="annot_harmony_final.pth",
        label="AnnotHarmony",
        data_dir=data_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnnotHarmony Training & Test Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/annotharmony/oxfordiiitpet.yaml",
        help="Path to the YAML experiment configuration file.",
    )
    parser.add_argument("--data", type=str, default=None, help="Override data_dir.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output_dir.")
    args = parser.parse_args()
    run_annotharmony_experiment(args.config, data_dir=args.data, output_dir=args.output_dir)
