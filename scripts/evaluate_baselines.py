"""CLI script for evaluating non-parametric baselines (MV & Spatial-STAPLE) on Test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from crowdsegmenter.config import ExperimentConfig
from crowdsegmenter.data.loader import CrowdSegmenterDataLoader
from crowdsegmenter.models.baselines import MajorityVoting, SpatialSTAPLE
from crowdsegmenter.utils.reproducibility import set_seed
from scripts._cli_utils import evaluate_on_test


def run_baseline_evaluation(
    config_path: str,
    model_name: str,
    output_dir: Optional[str] = None,
    beta: Optional[float] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate ``MV`` or ``Spatial-STAPLE`` on the Test partition."""
    cfg = ExperimentConfig.from_yaml(config_path)
    if data_dir is not None:
        cfg.data.data_dir = str(data_dir)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    set_seed(cfg.training.seed)

    out_path = Path(output_dir if output_dir is not None else cfg.experiment.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'':=<60}")
    print(f"  Baseline Experiment : {cfg.experiment.name}")
    print(f"  Model               : {model_name}")
    print(f"  Device              : {device}")
    print(f"  Output Directory    : {out_path}")
    print(f"  GT eval             : {cfg.data.load_ground_truth}")
    print(f"{'':=<60}\n")

    data_manager = CrowdSegmenterDataLoader(cfg.data, mode="Annot-Harmony")
    _, _, test_loader = data_manager.get_split_loaders()

    if model_name == "MV":
        model: torch.nn.Module = MajorityVoting(
            num_annotators=cfg.model.num_annotators,
            num_classes=cfg.model.num_classes,
            ignored_value=cfg.data.ignored_value,
        )
    elif model_name == "Spatial-STAPLE":
        staple_beta = beta if beta is not None else cfg.model.staple_beta
        model = SpatialSTAPLE(
            num_annotators=cfg.model.num_annotators,
            num_classes=cfg.model.num_classes,
            ignored_value=cfg.data.ignored_value,
            beta=staple_beta,
            max_iter=cfg.model.staple_max_iter,
            tol=cfg.model.staple_tol,
        )
    else:
        raise ValueError(f"Unknown baseline model: {model_name!r}")

    model = model.to(device)
    test_summary = evaluate_on_test(
        model=model,
        test_loader=test_loader,
        cfg=cfg,
        device=device,
        output_path=out_path,
        label=model_name,
        prefix="test",
    )

    results: Dict[str, Any] = {
        "experiment": cfg.experiment.name,
        "model": model_name,
        **test_summary,
    }
    save_file = out_path / f"baseline_{model_name.lower()}_results.json"
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nEvaluation complete. Results saved to: {save_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline Evaluation Script (Majority Voting & Spatial-STAPLE)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML experiment configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["MV", "Spatial-STAPLE"],
        required=True,
        help="Baseline model to evaluate.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional override for data_dir (e.g. kagglehub download path).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional override for output directory.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Optional Potts MRF beta override for Spatial-STAPLE (0.0 => classical STAPLE).",
    )
    args = parser.parse_args()
    run_baseline_evaluation(
        config_path=args.config,
        model_name=args.model,
        output_dir=args.output_dir,
        beta=args.beta,
        data_dir=args.data,
    )
