"""Ablation runner and paired statistical hypothesis testing on Test results.

Supports two modes:
    1. ``--compare name=dir ...``: loads ``test_<protocol>_per_sample.npz`` from
       each experiment directory, runs paired Wilcoxon signed-rank and paired
       t-tests with Holm-Bonferroni correction against ``--reference``, and
       exports JSON + CSV tables.
    2. ``--run_ct_gce_ablation --config <yaml>``: trains and evaluates the five
       CT-GCE ablation arms on the same splits/seed:
           - ``plain_gce``: no truncation, no reliability branch
           - ``s1_hard_gate``: hard truncation gate ``1{P > C_k}`` (r10 arm S1)
           - ``s2_gate_plus_lambda``: ``1{P > C_k} * Lambda`` (r10 arm S2)
           - ``s3_ct_gce_no_warmup``: full CT-GCE without warm-up
           - ``s3_ct_gce_full``: full CT-GCE (warm-up + Eq. 3/4/7/1, r10 arm S3)
       and runs paired statistical tests against ``s3_ct_gce_full``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from crowdsegmenter.config import ExperimentConfig
from crowdsegmenter.data.loader import CrowdSegmenterDataLoader
from crowdsegmenter.losses.ct_gce import CTGCELoss
from crowdsegmenter.models.ctgce import CTGCEModel
from crowdsegmenter.training.trainer import CTGCETrainer
from crowdsegmenter.utils.ablation import (
    compare_per_sample_metrics,
    load_per_sample_npz,
    save_ablation_report,
)
from crowdsegmenter.utils.reproducibility import set_seed
from scripts._cli_utils import evaluate_on_test


@dataclass(frozen=True)
class AblationArmSpec:
    name: str
    ablation_arm: Literal["S1", "S2", "S3"]
    no_warmup: bool = False
    plain_gce_only: bool = False


CT_GCE_ABLATION_ARMS: tuple[AblationArmSpec, ...] = (
    AblationArmSpec(name="plain_gce", ablation_arm="S3", plain_gce_only=True),
    AblationArmSpec(name="s1_hard_gate", ablation_arm="S1"),
    AblationArmSpec(name="s2_gate_plus_lambda", ablation_arm="S2"),
    AblationArmSpec(name="s3_ct_gce_no_warmup", ablation_arm="S3", no_warmup=True),
    AblationArmSpec(name="s3_ct_gce_full", ablation_arm="S3"),
)


def _print_ablation_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    print(f"\n{' ABLATION & STATISTICAL COMPARISON ':=^98}")
    print(
        f"{'Method':<22} | {'Metric':<11} | {'Mean ± Std':<17} | "
        f"{'Delta':<9} | {'Wilcoxon p_adj':<14} | {'Sig (0.05)'}"
    )
    print("-" * 98)
    for r in rows:
        mean_std = f"{r['mean']:.4f} ± {r['std']:.4f}"
        print(
            f"{r['method']:<22} | {r['metric']:<11} | {mean_std:<17} | "
            f"{r['delta_vs_ref']:+8.4f} | {r['wilcoxon_p_adj']:<14.4g} | {str(r['significant'])}"
        )
    print("=" * 98)


def compare_saved_results(
    entries: Dict[str, str | Path],
    protocol: Literal["probabilistic", "gt"],
    reference: str,
    output_dir: str | Path,
    prefix: str = "test",
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """Load saved per-sample ``.npz`` files and run paired ablation tests."""
    method_arrays = {
        name: load_per_sample_npz(Path(directory) / f"{prefix}_{protocol}_per_sample.npz")
        for name, directory in entries.items()
    }
    rows = compare_per_sample_metrics(method_arrays, reference=reference, alpha=alpha)
    _print_ablation_table(rows)
    save_ablation_report(rows, output_dir=output_dir, stem=f"ablation_{protocol}")
    return rows


def run_ct_gce_ablation_suite(
    config_path: str | Path,
    output_dir: Optional[str | Path] = None,
    alpha: float = 0.05,
    data_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Train and evaluate all CT-GCE ablation arms on the same splits & seed."""
    cfg = ExperimentConfig.from_yaml(config_path)
    if data_dir is not None:
        cfg.data.data_dir = str(data_dir)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    out_root = Path(output_dir) if output_dir is not None else Path(cfg.experiment.output_dir) / "ablation"
    out_root.mkdir(parents=True, exist_ok=True)

    data_manager = CrowdSegmenterDataLoader(cfg.data, mode="CT-GCE")
    train_loader, val_loader, test_loader = data_manager.get_split_loaders()

    arm_dirs: Dict[str, Path] = {}
    arm_summaries: Dict[str, Any] = {}

    for spec in CT_GCE_ABLATION_ARMS:
        set_seed(cfg.training.seed)
        arm_cfg = cfg.model_copy(deep=True)
        arm_cfg.experiment.save_results = True
        arm_dir = out_root / spec.name
        arm_dir.mkdir(parents=True, exist_ok=True)
        arm_cfg.experiment.output_dir = arm_dir

        if spec.plain_gce_only:
            arm_cfg.training.warmup_epochs = arm_cfg.training.epochs
        elif spec.no_warmup:
            arm_cfg.training.warmup_epochs = 0

        print(f"\n>>> Running CT-GCE ablation arm: {spec.name} (mode={spec.ablation_arm})")
        model = CTGCEModel(arm_cfg.model).to(device)
        criterion = CTGCELoss(
            annotators=arm_cfg.model.num_annotators,
            classes=arm_cfg.model.num_classes,
            ignored_value=arm_cfg.data.ignored_value,
            q=arm_cfg.training.ctgce_q,
            ablation_arm=spec.ablation_arm,
        ).to(device)

        trainer = CTGCETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            config=arm_cfg.training,
        )
        trained_model, _ = trainer.fit()

        test_summary = evaluate_on_test(
            model=trained_model,
            test_loader=test_loader,
            cfg=arm_cfg,
            device=device,
            output_path=arm_dir,
            label=spec.name,
            prefix="test",
        )
        arm_dirs[spec.name] = arm_dir
        arm_summaries[spec.name] = test_summary

    protocols: List[Literal["probabilistic", "gt"]] = ["probabilistic"]
    if cfg.data.load_ground_truth:
        protocols.append("gt")

    comparisons: Dict[str, List[Dict[str, Any]]] = {}
    for proto in protocols:
        comparisons[proto] = compare_saved_results(
            entries=arm_dirs,
            protocol=proto,
            reference="s3_ct_gce_full",
            output_dir=out_root,
            alpha=alpha,
        )

    payload = {"arms": arm_summaries, "comparisons": comparisons}
    with open(out_root / "ablation_suite_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    return payload


def _parse_compare_pairs(pairs: List[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Expected NAME=DIR pair in --compare, got {item!r}")
        name, path = item.split("=", 1)
        parsed[name.strip()] = path.strip()
    return parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CT-GCE Ablation Suite & Paired Hypothesis Testing"
    )
    parser.add_argument(
        "--run_ct_gce_ablation",
        action="store_true",
        help="Run the 5 CT-GCE ablation arms on a YAML config and compare on Test.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/ctgce/oxfordiiitpet.yaml",
        help="YAML config used with --run_ct_gce_ablation.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional override for data_dir when running --run_ct_gce_ablation.",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        help="Compare saved per-sample .npz results as NAME=OUTPUT_DIR pairs.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["probabilistic", "gt"],
        default="probabilistic",
        help="Evaluation protocol when using --compare.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="CT-GCE",
        help="Reference method name when using --compare.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ablation",
        help="Directory to save JSON/CSV comparison reports.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level after Holm-Bonferroni correction.",
    )
    args = parser.parse_args()

    if args.run_ct_gce_ablation:
        run_ct_gce_ablation_suite(
            config_path=args.config,
            output_dir=args.output_dir,
            alpha=args.alpha,
            data_dir=args.data,
        )
    elif args.compare:
        entries = _parse_compare_pairs(args.compare)
        compare_saved_results(
            entries=entries,
            protocol=args.protocol,
            reference=args.reference,
            output_dir=args.output_dir,
            alpha=args.alpha,
        )
    else:
        parser.error("Specify either --run_ct_gce_ablation or --compare NAME=DIR ...")
