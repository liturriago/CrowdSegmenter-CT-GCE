"""Shared CLI helpers for training, test evaluation, and Optuna tuning scripts."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _enable_realtime_logs() -> None:
    """Ensure stdout/stderr flush line-by-line in Colab and non-TTY runners."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True)
            except Exception:
                pass

from crowdsegmenter.config import ExperimentConfig
from crowdsegmenter.data.loader import CrowdSegmenterDataLoader
from crowdsegmenter.models.annot_harmony import AnnotHarmony
from crowdsegmenter.models.baselines import MajorityVoting, SpatialSTAPLE
from crowdsegmenter.models.crowdseg import CrowdSeg
from crowdsegmenter.models.ctgce import CTGCEModel
from crowdsegmenter.models.supervised import SupervisedResUNet
from crowdsegmenter.utils.metrics import MetricTracker
from crowdsegmenter.utils.reproducibility import set_seed


DEFAULT_DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    "oxfordiiitpet": {
        "CT-GCE": "configs/training/ctgce/oxfordiiitpet.yaml",
        "Annot-Harmony": "configs/training/tgce_ssps/oxfordiiitpet.yaml",
        "CrowdSeg": "configs/training/crowdseg/oxfordiiitpet.yaml",
        "MV": "configs/training/MV/oxfordiiitpet.yaml",
        "Spatial-STAPLE": "configs/training/spatial_staple/oxfordiiitpet.yaml",
        "Supervised-GT": "configs/training/supervised/oxfordiiitpet.yaml",
    },
    "riga": {
        "CT-GCE": "configs/training/ctgce/riga.yaml",
        "Annot-Harmony": "configs/training/tgce_ssps/riga.yaml",
        "CrowdSeg": "configs/training/crowdseg/riga.yaml",
        "MV": "configs/training/MV/riga.yaml",
        "Spatial-STAPLE": "configs/training/spatial_staple/riga.yaml",
    },
    "naip_wellpad": {
        "CT-GCE": "configs/training/ctgce/naip_wellpad.yaml",
        "Annot-Harmony": "configs/training/tgce_ssps/naip_wellpad.yaml",
        "CrowdSeg": "configs/training/crowdseg/naip_wellpad.yaml",
        "MV": "configs/training/MV/naip_wellpad.yaml",
        "Spatial-STAPLE": "configs/training/spatial_staple/naip_wellpad.yaml",
        "Supervised-GT": "configs/training/supervised/naip_wellpad.yaml",
    },
}

_METHOD_ALIASES: Dict[str, str] = {
    "ct-gce": "CT-GCE",
    "ctgce": "CT-GCE",
    "ct_gce": "CT-GCE",
    "annot-harmony": "Annot-Harmony",
    "annotharmony": "Annot-Harmony",
    "annot_harmony": "Annot-Harmony",
    "tgce-ssps": "Annot-Harmony",
    "tgce_ssps": "Annot-Harmony",
    "crowdseg": "CrowdSeg",
    "mv": "MV",
    "majority-voting": "MV",
    "majority_voting": "MV",
    "spatial-staple": "Spatial-STAPLE",
    "spatial_staple": "Spatial-STAPLE",
    "staple": "Spatial-STAPLE",
    "supervised-gt": "Supervised-GT",
    "supervised_gt": "Supervised-GT",
    "supervised": "Supervised-GT",
}


@dataclass(frozen=True)
class MethodSpec:
    """Descriptor pairing a benchmarked method name with its builder and optional weights."""

    name: str
    slug: str
    builder: Callable[[ExperimentConfig, Optional[float]], nn.Module]
    weights_key: Optional[str] = None


def _build_neural_model(
    cls: type[nn.Module],
    cfg: ExperimentConfig,
) -> nn.Module:
    # Avoid downloading ImageNet weights before overwriting parameters from checkpoint
    infer_model_cfg = cfg.model.model_copy(update={"pretrained": False})
    return cls(infer_model_cfg)


def load_checkpoint(
    model: nn.Module,
    weights_path: str | Path,
    device: torch.device,
) -> nn.Module:
    """Load state_dict from ``weights_path`` into ``model`` and switch to eval mode."""
    ckpt_path = Path(weights_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(payload, dict):
        if "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        elif "state_dict" in payload:
            state_dict = payload["state_dict"]
        else:
            state_dict = payload
    else:
        state_dict = payload

    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_all_method_specs() -> Tuple[MethodSpec, ...]:
    """Return canonical descriptors for all supported neural and classical methods."""
    return (
        MethodSpec(
            name="CT-GCE",
            slug="ct_gce",
            weights_key="CT-GCE",
            builder=lambda cfg, _: _build_neural_model(CTGCEModel, cfg),
        ),
        MethodSpec(
            name="Annot-Harmony",
            slug="annot_harmony",
            weights_key="Annot-Harmony",
            builder=lambda cfg, _: _build_neural_model(AnnotHarmony, cfg),
        ),
        MethodSpec(
            name="CrowdSeg",
            slug="crowdseg",
            weights_key="CrowdSeg",
            builder=lambda cfg, _: _build_neural_model(CrowdSeg, cfg),
        ),
        MethodSpec(
            name="MV",
            slug="mv",
            builder=lambda cfg, _: MajorityVoting(
                num_annotators=cfg.model.num_annotators,
                num_classes=cfg.model.num_classes,
                ignored_value=cfg.data.ignored_value,
            ),
        ),
        MethodSpec(
            name="Spatial-STAPLE",
            slug="spatial_staple",
            builder=lambda cfg, beta: SpatialSTAPLE(
                num_annotators=cfg.model.num_annotators,
                num_classes=cfg.model.num_classes,
                ignored_value=cfg.data.ignored_value,
                beta=beta if beta is not None else cfg.model.staple_beta,
                max_iter=cfg.model.staple_max_iter,
                tol=cfg.model.staple_tol,
            ),
        ),
        MethodSpec(
            name="Supervised-GT",
            slug="supervised",
            weights_key="Supervised-GT",
            builder=lambda cfg, _: _build_neural_model(SupervisedResUNet, cfg),
        ),
    )


def normalize_method_name(name: str) -> str:
    """Normalize method names or CLI slugs to canonical display names."""
    key = name.strip().lower()
    if key not in _METHOD_ALIASES:
        valid = sorted({spec.name for spec in build_all_method_specs()})
        raise ValueError(f"Unknown method {name!r}. Expected one of {valid}.")
    return _METHOD_ALIASES[key]


def resolve_method_selection(
    ctgce_weights: Optional[str | Path] = None,
    annotharmony_weights: Optional[str | Path] = None,
    crowdseg_weights: Optional[str | Path] = None,
    supervised_weights: Optional[str | Path] = None,
    include_mv: bool = True,
    include_staple: bool = True,
    methods: Optional[Sequence[str]] = None,
    dataset: Optional[str] = "oxfordiiitpet",
    config_overrides: Optional[Mapping[str, str | Path]] = None,
) -> Tuple[Tuple[MethodSpec, ...], Dict[str, Path], Dict[str, Path]]:
    """Select active methods, validate available checkpoints, and resolve YAML configs.

    Only deep-learning models whose weights path is provided are auto-selected,
    making single-model or partial-subset benchmarks & visual runs seamless.
    """
    all_specs = {spec.name: spec for spec in build_all_method_specs()}
    defaults = DEFAULT_DATASET_CONFIGS.get(dataset or "", {})

    weights_map: Dict[str, Path] = {}
    for key, val in (
        ("CT-GCE", ctgce_weights),
        ("Annot-Harmony", annotharmony_weights),
        ("CrowdSeg", crowdseg_weights),
        ("Supervised-GT", supervised_weights),
    ):
        if val is not None:
            weights_map[key] = Path(val)

    selected_specs: List[MethodSpec] = []
    if methods is not None:
        seen: set[str] = set()
        for raw_name in methods:
            canonical = normalize_method_name(raw_name)
            if canonical in seen:
                continue
            seen.add(canonical)
            spec = all_specs[canonical]
            if spec.weights_key is not None and spec.weights_key not in weights_map:
                raise ValueError(
                    f"Method {canonical!r} requires a checkpoint path, but none was provided."
                )
            selected_specs.append(spec)
    else:
        for spec in build_all_method_specs():
            if spec.weights_key is not None:
                if spec.weights_key in weights_map:
                    selected_specs.append(spec)
            elif spec.name == "MV" and include_mv:
                if (config_overrides or {}).get("MV") is not None or "MV" in defaults:
                    selected_specs.append(spec)
            elif spec.name == "Spatial-STAPLE" and include_staple:
                if (
                    (config_overrides or {}).get("Spatial-STAPLE") is not None
                    or "Spatial-STAPLE" in defaults
                ):
                    selected_specs.append(spec)

    if not selected_specs:
        raise ValueError(
            "No methods selected. Provide at least one model checkpoint or enable MV / Spatial-STAPLE."
        )

    active_weights = {
        spec.weights_key: weights_map[spec.weights_key]
        for spec in selected_specs
        if spec.weights_key is not None
    }
    for name, wpath in active_weights.items():
        if not wpath.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {name}: {wpath}")

    resolved_configs: Dict[str, Path] = {}
    for spec in selected_specs:
        override = (config_overrides or {}).get(spec.name)
        cfg_val = override if override is not None else defaults.get(spec.name)
        if cfg_val is None:
            raise ValueError(
                f"No YAML config resolved for {spec.name!r} (dataset={dataset!r})."
            )
        resolved_configs[spec.name] = Path(cfg_val)

    return tuple(selected_specs), active_weights, resolved_configs


def resolve_reference_method(
    selected_specs: Sequence[MethodSpec],
    reference: Optional[str] = None,
) -> str:
    """Resolve the statistical reference method from the active selection."""
    selected_names = [spec.name for spec in selected_specs]
    if reference is not None:
        canonical = normalize_method_name(reference)
        if canonical not in selected_names:
            raise ValueError(
                f"Reference method {canonical!r} is not among active methods {selected_names}."
            )
        return canonical

    if "CT-GCE" in selected_names:
        return "CT-GCE"
    for spec in selected_specs:
        if spec.weights_key is not None:
            return spec.name
    return selected_names[0]


def serialize_tensors(obj: Any) -> Any:
    """Recursively convert tensors into JSON-serializable Python lists/scalars."""
    if isinstance(obj, dict):
        return {k: serialize_tensors(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_tensors(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def evaluate_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
    output_path: Path,
    label: str,
    prefix: str = "test",
    eval_nlc: Optional[bool] = None,
    nlc_features: Optional[Sequence[np.ndarray]] = None,
    nlc_pretrained: bool = True,
) -> Dict[str, Any]:
    """Run probabilistic, optional GT, and optional zero-GT NLC evaluation on ``test_loader``."""
    class_names = [f"Class {i}" for i in range(cfg.model.num_classes)]
    tracker = MetricTracker(cfg.training)

    prob_metrics = tracker.evaluation(
        model,
        test_loader,
        device,
        save_results=cfg.experiment.save_results,
        output_dir=output_path,
        prefix=prefix,
    )
    tracker.print_full_report(f"{label} Test — Probabilistic", prob_metrics, class_names)

    summary: Dict[str, Any] = {
        "probabilistic": serialize_tensors(prob_metrics),
    }

    if cfg.data.load_ground_truth:
        gt_metrics = tracker.evaluation_gt(
            model,
            test_loader,
            device,
            save_results=cfg.experiment.save_results,
            output_dir=output_path,
            prefix=prefix,
        )
        tracker.print_full_report(f"{label} Test — Ground Truth", gt_metrics, class_names)
        summary["ground_truth"] = serialize_tensors(gt_metrics)

    use_nlc = cfg.training.eval_nlc if eval_nlc is None else eval_nlc
    if use_nlc:
        nlc_metrics = tracker.evaluation_nlc(
            model=model,
            loader=test_loader,
            device=device,
            features=nlc_features,
            nlc_pretrained=nlc_pretrained,
        )
        cls_str = " | ".join(
            f"{k}: {v:.4f}" for k, v in nlc_metrics["per_class"].items()
        )
        print(
            f"\n[{label} Test — Zero-GT NLC] Macro Avg: {nlc_metrics['macro_avg']:.4f}"
            + (f" | {cls_str}" if cls_str else "")
        )
        summary["nlc"] = serialize_tensors(nlc_metrics)

    return summary


def run_training_experiment(
    config_path: str | Path,
    data_mode: str,
    model_builder: Callable[[ExperimentConfig], nn.Module],
    loss_builder: Callable[[ExperimentConfig], nn.Module],
    trainer_cls: Type[Any],
    weights_filename: str,
    label: str,
    extra_summary_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
    data_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    eval_nlc: Optional[bool] = None,
    nlc_pretrained: bool = True,
) -> Dict[str, Any]:
    """Execute training, test-set evaluation, and artifact serialization."""
    _enable_realtime_logs()
    cfg = ExperimentConfig.from_yaml(config_path)
    if data_dir is not None:
        cfg.data.data_dir = str(data_dir)
    if output_dir is not None:
        cfg.experiment.output_dir = Path(output_dir)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    set_seed(cfg.training.seed)

    output_path = Path(cfg.experiment.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'':=<60}")
    print(f"  Experiment : {cfg.experiment.name} ({label})")
    print(f"  Device     : {device}")
    print(f"  Output     : {output_path}")
    print(f"  GT eval    : {cfg.data.load_ground_truth}")
    print(f"{'':=<60}\n")

    data_manager = CrowdSegmenterDataLoader(cfg.data, mode=data_mode)  # type: ignore[arg-type]
    train_loader, val_loader, test_loader = data_manager.get_split_loaders()

    model = model_builder(cfg).to(device)
    criterion = loss_builder(cfg).to(device)

    trainer = trainer_cls(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        config=cfg.training,
    )
    trained_model, history = trainer.fit()

    test_summary = evaluate_on_test(
        model=trained_model,
        test_loader=test_loader,
        cfg=cfg,
        device=device,
        output_path=output_path,
        label=label,
        prefix="test",
        eval_nlc=eval_nlc,
        nlc_pretrained=nlc_pretrained,
    )

    weights_path = output_path / weights_filename
    torch.save(trained_model.state_dict(), weights_path)

    full_summary: Dict[str, Any] = {
        "experiment": cfg.experiment.name,
        "model": label,
        "best_val_dice": trainer.best_val_dice,
        "history": serialize_tensors(history),
        "test": test_summary,
        "weights": str(weights_path),
    }
    if extra_summary_fn is not None:
        full_summary.update(serialize_tensors(extra_summary_fn(trainer)))

    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=4)

    print(f"\nExperiment complete. Weights and summary saved in: {output_path}")
    return full_summary


def run_optimization_cli(
    tuner_cls: Type[Any],
    config_path: str | Path,
    n_trials: Optional[int],
    n_epochs_per_trial: Optional[int],
    output_dir: Optional[str],
    label: str,
    data_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run an Optuna tuner from CLI args and optionally export matplotlib plots."""
    _enable_realtime_logs()
    print(f"\nLoading dataset & initializing {label} tuner...", flush=True)
    tuner = tuner_cls(
        config_path=config_path,
        n_trials=n_trials,
        n_epochs_per_trial=n_epochs_per_trial,
        output_dir=output_dir,
        data_dir=data_dir,
    )
    print(
        f"\nStarting {label} Optimization "
        f"(Trials: {tuner.n_trials}, Epochs/trial: {tuner.n_epochs_per_trial})"
    )
    results = tuner.run()
    print(f"\nOptimization complete. Best params: {results['best_params']}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import optuna

        vis = optuna.visualization.matplotlib
        plot_specs = [
            ("optimization_history.png", vis.plot_optimization_history),
            ("param_slice.png", vis.plot_slice),
            ("intermediate_values.png", vis.plot_intermediate_values),
            ("param_importances.png", vis.plot_param_importances),
        ]
        saved_plots = []
        for filename, plot_fn in plot_specs:
            try:
                ax = plot_fn(tuner.study)
                fig = ax.ravel()[0].figure if hasattr(ax, "ravel") else ax.figure
                fig.savefig(tuner.output_dir / filename, dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved_plots.append(filename)
            except Exception as exc:
                print(f"Skipping {filename}: {exc}")
        if saved_plots:
            print(f"Saved optimization plots to {tuner.output_dir}: {', '.join(saved_plots)}")
    except ImportError:
        print("matplotlib or optuna visualization backend not available; skipping plots.")

    return results
