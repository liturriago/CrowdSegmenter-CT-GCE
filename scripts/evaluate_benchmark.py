"""Flexible Test benchmark & paired statistical comparison across selected methods.

Evaluates any subset of trained segmentation models (one or more of ``CT-GCE``,
``Annot-Harmony``, ``CrowdSeg``, and optional ``Supervised-GT``) loaded from
saved ``.pth`` checkpoints alongside optional non-parametric baselines
(``MV``, ``Spatial-STAPLE``) on the same Test split, then computes paired
t-tests and paired Wilcoxon signed-rank tests with Holm-Bonferroni correction
against ``--reference`` (auto-selecting ``CT-GCE`` when present, otherwise the
first active model).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from crowdsegmenter.config import ExperimentConfig
from crowdsegmenter.data.loader import CrowdSegmenterDataLoader
from crowdsegmenter.utils.ablation import (
    compare_per_sample_metrics,
    load_per_sample_npz,
    save_ablation_report,
)
from crowdsegmenter.utils.metrics import (
    compute_loader_panel_nlc_metrics,
    extract_prepared_dense_features,
)
from crowdsegmenter.utils.reproducibility import set_seed
from scripts._cli_utils import (
    DEFAULT_DATASET_CONFIGS,
    _enable_realtime_logs,
    evaluate_on_test,
    load_checkpoint,
    resolve_method_selection,
    resolve_reference_method,
    serialize_tensors,
)


def _print_benchmark_table(
    rows: List[Dict[str, Any]],
    protocol: str,
    alpha: float,
) -> None:
    if not rows:
        return
    title = f" BENCHMARK & STATISTICAL TESTS ({protocol.upper()}) "
    print(f"\n{title:=^124}")
    print(
        f"{'Method':<16} | {'Metric':<11} | {'Mean ± Std':<17} | "
        f"{'Delta':<9} | {'t-test p':<11} | {'t-test p_adj':<12} | "
        f"{'Wilcoxon p_adj':<14} | {'Sig (t/W @ ' + str(alpha) + ')'}"
    )
    print("-" * 124)
    for r in rows:
        mean_std = f"{r['mean']:.4f} ± {r['std']:.4f}"
        sig_t = r["ttest_p_adj"] < alpha
        sig_w = r["wilcoxon_p_adj"] < alpha
        print(
            f"{r['method']:<16} | {r['metric']:<11} | {mean_std:<17} | "
            f"{r['delta_vs_ref']:+8.4f} | {r['ttest_p']:<11.4g} | {r['ttest_p_adj']:<12.4g} | "
            f"{r['wilcoxon_p_adj']:<14.4g} | t={sig_t} / W={sig_w}"
        )
    print("=" * 124)


def _print_nlc_table(
    nlc_rows: List[Dict[str, Any]],
    panel_nlc: Optional[Dict[str, Any]] = None,
) -> None:
    if not nlc_rows:
        return
    cls_keys = [k for k in nlc_rows[0] if k.startswith("class_")]
    header_cls = " | ".join(f"{k:<10}" for k in cls_keys)
    width = 62 + len(cls_keys) * 13
    print(f"\n{' BENCHMARK ZERO-GT NLC CONSISTENCY ':=^{width}}")
    print(
        f"{'Method':<22} | {'Macro Avg':<10} | {'Delta':<9} | "
        + (header_cls if cls_keys else "")
    )
    print("-" * width)
    for r in nlc_rows:
        cls_vals = " | ".join(f"{r[k]:<10.4f}" for k in cls_keys)
        print(
            f"{r['method']:<22} | {r['macro_avg']:<10.4f} | {r['delta_vs_ref']:+8.4f} | "
            + cls_vals
        )
    if panel_nlc is not None:
        p_cls = " | ".join(
            f"{panel_nlc['per_class'][k]['panel_mean']:<10.4f}"
            for k in cls_keys
            if k in panel_nlc.get("per_class", {})
        )
        print("-" * width)
        print(
            f"{'Annotator Panel (mean)':<22} | {panel_nlc['macro_panel_mean']:<10.4f} | {'—':<9} | "
            + p_cls
        )
    print("=" * width)


def run_benchmark_evaluation(
    ctgce_weights: Optional[str | Path] = None,
    annotharmony_weights: Optional[str | Path] = None,
    crowdseg_weights: Optional[str | Path] = None,
    supervised_weights: Optional[str | Path] = None,
    results_dir: Optional[str | Path] = None,
    include_mv: bool = True,
    include_staple: bool = True,
    methods: Optional[Sequence[str]] = None,
    dataset: Optional[str] = "oxfordiiitpet",
    config_overrides: Optional[Mapping[str, str | Path]] = None,
    data_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    reference: Optional[str] = None,
    alpha: float = 0.05,
    beta: Optional[float] = None,
    eval_nlc: Optional[bool] = None,
    nlc_pretrained: bool = True,
) -> Dict[str, Any]:
    """Evaluate any selected subset of deep models and classical baselines on Test."""
    _enable_realtime_logs()

    method_specs, weights_map, resolved_configs = resolve_method_selection(
        ctgce_weights=ctgce_weights,
        annotharmony_weights=annotharmony_weights,
        crowdseg_weights=crowdseg_weights,
        supervised_weights=supervised_weights,
        results_dir=results_dir,
        include_mv=include_mv,
        include_staple=include_staple,
        methods=methods,
        dataset=dataset,
        config_overrides=config_overrides,
    )
    ref_name = resolve_reference_method(method_specs, reference=reference)

    configs: Dict[str, ExperimentConfig] = {}
    for name, cfg_path in resolved_configs.items():
        cfg = ExperimentConfig.from_yaml(cfg_path)
        if data_dir is not None:
            cfg.data.data_dir = str(data_dir)
        cfg.experiment.save_results = True
        configs[name] = cfg

    ref_cfg = configs[ref_name]
    device = torch.device(ref_cfg.training.device if torch.cuda.is_available() else "cpu")
    set_seed(ref_cfg.training.seed)

    out_root = Path(
        output_dir
        if output_dir is not None
        else f"results/benchmark/{dataset or 'custom'}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    use_nlc = ref_cfg.training.eval_nlc if eval_nlc is None else eval_nlc

    print(f"\n{'':=<72}")
    print(f"  Benchmark Evaluation ({len(method_specs)} methods, {dataset or 'custom'})")
    print(f"  Methods   : {', '.join(s.name for s in method_specs)}")
    print(f"  Device    : {device}")
    print(f"  Reference : {ref_name}")
    print(f"  Output    : {out_root}")
    print(f"  GT eval   : {ref_cfg.data.load_ground_truth}")
    print(f"  Zero-GT   : {use_nlc}")
    print(f"{'':=<72}\n")

    # Use all-annotator mode so probabilistic consensus targets are identical across methods
    data_manager = CrowdSegmenterDataLoader(ref_cfg.data, mode="CT-GCE")
    _, _, test_loader = data_manager.get_split_loaders()

    nlc_features = (
        extract_prepared_dense_features(
            loader=test_loader,
            device=device,
            pretrained=nlc_pretrained,
        )
        if use_nlc
        else None
    )

    method_dirs: Dict[str, Path] = {}
    method_summaries: Dict[str, Any] = {}

    for spec in method_specs:
        cfg = configs[spec.name]
        model_dir = out_root / spec.slug
        model_dir.mkdir(parents=True, exist_ok=True)

        model = spec.builder(cfg, beta).to(device)
        if spec.weights_key is not None:
            load_checkpoint(model, weights_map[spec.weights_key], device=device)
        else:
            model.eval()

        test_summary = evaluate_on_test(
            model=model,
            test_loader=test_loader,
            cfg=cfg,
            device=device,
            output_path=model_dir,
            label=spec.name,
            prefix="test",
            eval_nlc=use_nlc,
            nlc_features=nlc_features,
            nlc_pretrained=nlc_pretrained,
        )
        method_dirs[spec.name] = model_dir
        method_summaries[spec.name] = test_summary

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    protocols: List[Literal["probabilistic", "gt"]] = ["probabilistic"]
    if ref_cfg.data.load_ground_truth:
        protocols.append("gt")

    comparisons: Dict[str, List[Dict[str, Any]]] = {}
    for proto in protocols:
        arrays = {
            name: load_per_sample_npz(m_dir / f"test_{proto}_per_sample.npz")
            for name, m_dir in method_dirs.items()
        }
        rows = compare_per_sample_metrics(arrays, reference=ref_name, alpha=alpha)
        _print_benchmark_table(rows, protocol=proto, alpha=alpha)
        save_ablation_report(rows, output_dir=out_root, stem=f"benchmark_{proto}")
        comparisons[proto] = rows

    nlc_rows: List[Dict[str, Any]] = []
    panel_nlc: Optional[Dict[str, Any]] = None
    if use_nlc and nlc_features is not None:
        panel_nlc = compute_loader_panel_nlc_metrics(
            features=nlc_features,
            loader=test_loader,
            num_classes=ref_cfg.model.num_classes,
            num_annotators=ref_cfg.model.num_annotators,
            ignored_value=ref_cfg.data.ignored_value,
            prepare=False,
        )
        ref_macro = float(method_summaries[ref_name]["nlc"]["macro_avg"])
        for spec in method_specs:
            nlc_info = method_summaries[spec.name]["nlc"]
            row: Dict[str, Any] = {
                "method": spec.name,
                "reference": ref_name,
                "macro_avg": float(nlc_info["macro_avg"]),
                "delta_vs_ref": float(ref_macro - nlc_info["macro_avg"]),
            }
            for cls_k, cls_val in nlc_info["per_class"].items():
                row[cls_k] = float(cls_val)
            nlc_rows.append(row)
        _print_nlc_table(nlc_rows, panel_nlc=panel_nlc)
        save_ablation_report(nlc_rows, output_dir=out_root, stem="benchmark_nlc")

    summary_payload: Dict[str, Any] = {
        "dataset": dataset,
        "reference": ref_name,
        "alpha": alpha,
        "eval_nlc": use_nlc,
        "weights": {k: str(v) for k, v in weights_map.items()},
        "configs": {k: str(v) for k, v in resolved_configs.items()},
        "methods": serialize_tensors(method_summaries),
        "comparisons": serialize_tensors(comparisons),
    }
    if use_nlc:
        summary_payload["nlc_comparison"] = serialize_tensors(nlc_rows)
        summary_payload["panel_nlc"] = serialize_tensors(panel_nlc)
    summary_path = out_root / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=4)

    print(f"\nBenchmark complete. Summary & paired tests saved in: {out_root}")
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flexible Test evaluation & paired t-test/Wilcoxon benchmark across selected methods"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=sorted(DEFAULT_DATASET_CONFIGS.keys()),
        default="oxfordiiitpet",
        help="Dataset preset selecting default training YAML configs.",
    )
    parser.add_argument(
        "--ctgce_weights",
        type=str,
        default=None,
        help="Optional path to trained CT-GCE checkpoint (.pth).",
    )
    parser.add_argument(
        "--tgce_ssps_weights",
        "--annotharmony_weights",
        dest="tgce_ssps_weights",
        type=str,
        default=None,
        help="Optional path to trained TGCE-SSPS (Annot-Harmony) checkpoint (.pth).",
    )
    parser.add_argument(
        "--crowdseg_weights",
        type=str,
        default=None,
        help="Optional path to trained CrowdSeg checkpoint (.pth).",
    )
    parser.add_argument(
        "--supervised_weights",
        type=str,
        default=None,
        help="Optional path to trained Supervised-GT oracle checkpoint (.pth).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help=(
            "Optional root results folder to auto-discover trained checkpoints "
            "(e.g. results/{ctgce,tgce_ssps,crowdseg,supervised}/{dataset}/*_final.pth)."
        ),
    )
    parser.add_argument(
        "--no-mv",
        "--no_mv",
        dest="include_mv",
        action="store_false",
        default=True,
        help="Exclude Majority Voting baseline when --methods is not specified.",
    )
    parser.add_argument(
        "--no-staple",
        "--no_staple",
        dest="include_staple",
        action="store_false",
        default=True,
        help="Exclude Spatial-STAPLE baseline when --methods is not specified.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "Explicit subset of methods to run (e.g. CT-GCE CrowdSeg MV Spatial-STAPLE). "
            "When omitted, runs all DL models whose weights are provided plus MV/Spatial-STAPLE."
        ),
    )
    parser.add_argument("--ctgce_config", type=str, default=None, help="Optional CT-GCE YAML override.")
    parser.add_argument(
        "--tgce_ssps_config",
        "--annotharmony_config",
        dest="tgce_ssps_config",
        type=str,
        default=None,
        help="Optional TGCE-SSPS (Annot-Harmony) YAML override.",
    )
    parser.add_argument("--crowdseg_config", type=str, default=None, help="Optional CrowdSeg YAML override.")
    parser.add_argument("--mv_config", type=str, default=None, help="Optional MV YAML override.")
    parser.add_argument("--staple_config", type=str, default=None, help="Optional Spatial-STAPLE YAML override.")
    parser.add_argument("--supervised_config", type=str, default=None, help="Optional Supervised-GT YAML override.")
    parser.add_argument("--data", type=str, default=None, help="Optional override for data_dir.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional override for output_dir.")
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Reference method for paired tests (defaults to CT-GCE if active, else first active model).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level after Holm-Bonferroni correction.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Optional Potts MRF beta override for Spatial-STAPLE.",
    )
    parser.add_argument(
        "--eval-nlc",
        "--eval_nlc",
        dest="eval_nlc",
        action="store_true",
        default=None,
        help="Compute zero-GT NLC consistency across active methods and annotator panel.",
    )
    args = parser.parse_args()

    overrides: Dict[str, str] = {}
    for key, val in (
        ("CT-GCE", args.ctgce_config),
        ("Annot-Harmony", args.tgce_ssps_config),
        ("CrowdSeg", args.crowdseg_config),
        ("MV", args.mv_config),
        ("Spatial-STAPLE", args.staple_config),
        ("Supervised-GT", args.supervised_config),
    ):
        if val is not None:
            overrides[key] = val

    run_benchmark_evaluation(
        ctgce_weights=args.ctgce_weights,
        annotharmony_weights=args.tgce_ssps_weights,
        crowdseg_weights=args.crowdseg_weights,
        supervised_weights=args.supervised_weights,
        results_dir=args.results_dir,
        include_mv=args.include_mv,
        include_staple=args.include_staple,
        methods=args.methods,
        dataset=args.dataset,
        config_overrides=overrides,
        data_dir=args.data,
        output_dir=args.output_dir,
        reference=args.reference,
        alpha=args.alpha,
        beta=args.beta,
        eval_nlc=args.eval_nlc,
    )


if __name__ == "__main__":
    main()
