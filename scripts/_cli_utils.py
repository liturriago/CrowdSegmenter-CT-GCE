"""Shared CLI helpers for training, test evaluation, and Optuna tuning scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crowdsegmenter.config import ExperimentConfig
from crowdsegmenter.data.loader import CrowdSegmenterDataLoader
from crowdsegmenter.utils.metrics import MetricTracker
from crowdsegmenter.utils.reproducibility import set_seed


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
) -> Dict[str, Any]:
    """Run probabilistic and optional GT evaluation on ``test_loader``."""
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
) -> Dict[str, Any]:
    """Execute training, test-set evaluation, and artifact serialization."""
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
