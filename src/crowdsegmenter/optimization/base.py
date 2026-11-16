"""Shared Optuna hyperparameter optimization base classes and pruning trainers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import optuna
except ImportError:  # pragma: no cover - handled gracefully when tuning extra is absent
    optuna = None  # type: ignore[assignment]

from crowdsegmenter.config import ExperimentConfig, TrainConfig
from crowdsegmenter.data.loader import CrowdSegmenterDataLoader
from crowdsegmenter.training.trainer import CTGCETrainer, Trainer
from crowdsegmenter.utils.metrics import MetricTracker
from crowdsegmenter.utils.reproducibility import set_seed


def require_optuna() -> Any:
    """Return the ``optuna`` module or raise a clear ``ImportError``."""
    if optuna is None:
        raise ImportError(
            "optuna is required for hyperparameter tuning. "
            "Install optional extras with: pip install -e '.[tuning]'"
        )
    return optuna


class _OptunaPruningMixin:
    """Mixin forwarding validation Dice to Optuna and raising ``TrialPruned`` when triggered."""

    trial: Any
    current_epoch: int
    _use_gt: bool

    def _report_to_optuna(self, score: float) -> None:
        opt = require_optuna()
        self.trial.report(score, self.current_epoch)
        if self.trial.should_prune():
            raise opt.TrialPruned()
        self.current_epoch += 1

    def evaluate(self, loader: DataLoader, prefix: str = "Val") -> float:
        val_dice_prob = super().evaluate(loader, prefix)  # type: ignore[misc]
        if not self._use_gt:
            self._report_to_optuna(val_dice_prob)
        return val_dice_prob

    def evaluate_gt(self, loader: DataLoader, prefix: str = "Val (GT)") -> float:
        val_dice_gt = super().evaluate_gt(loader, prefix)  # type: ignore[misc]
        if self._use_gt:
            self._report_to_optuna(val_dice_gt)
        return val_dice_gt


class OptunaTrainer(_OptunaPruningMixin, Trainer):
    """Curriculum ``Trainer`` with Optuna intermediate reporting and pruning."""

    def __init__(self, trial: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trial = trial
        self.current_epoch = 0


class OptunaCTGCETrainer(_OptunaPruningMixin, CTGCETrainer):
    """Decoupled ``CTGCETrainer`` with Optuna intermediate reporting and pruning."""

    def __init__(self, trial: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trial = trial
        self.current_epoch = 0


class BaseOptunaTuner(ABC):
    """Shared Bayesian hyperparameter tuner backed by Optuna."""

    data_mode: Literal["Annot-Harmony", "CT-GCE", "CrowdSeg"]
    primary_param_name: str
    default_study_name: str = "optuna_study"
    default_output_dir: str = "results/optuna"

    def __init__(
        self,
        config_path: str | Path,
        n_trials: Optional[int] = None,
        n_epochs_per_trial: Optional[int] = None,
        study_name: Optional[str] = None,
        output_dir: Optional[str | Path] = None,
        seed: Optional[int] = None,
        data_dir: Optional[str | Path] = None,
    ) -> None:
        opt = require_optuna()

        self.config_path = str(config_path)
        self.config = ExperimentConfig.from_yaml(config_path)
        if data_dir is not None:
            self.config.data.data_dir = str(data_dir)
        opt_cfg = self.config.optimization

        self.n_trials = (
            n_trials
            if n_trials is not None
            else (opt_cfg.n_trials if opt_cfg is not None else 50)
        )
        self.n_epochs_per_trial = (
            n_epochs_per_trial
            if n_epochs_per_trial is not None
            else (
                opt_cfg.n_epochs_per_trial
                if opt_cfg is not None
                else self.config.training.n_epochs_per_trial
            )
        )
        self.study_name = (
            study_name
            if study_name is not None
            else (opt_cfg.study_name if opt_cfg is not None else self.default_study_name)
        )
        self.output_dir = Path(
            output_dir
            if output_dir is not None
            else (
                self.config.experiment.output_dir
                if opt_cfg is not None
                else self.default_output_dir
            )
        )
        self.seed = seed if seed is not None else self.config.training.seed
        self.device = torch.device(
            self.config.training.device if torch.cuda.is_available() else "cpu"
        )

        data_manager = CrowdSegmenterDataLoader(self.config.data, mode=self.data_mode)
        self.train_loader, self.val_loader, _ = data_manager.get_split_loaders()

        self.study = opt.create_study(
            study_name=self.study_name,
            direction="maximize",
            sampler=opt.samplers.TPESampler(seed=self.seed),
            pruner=opt.pruners.MedianPruner(),
        )

        self.results: List[Dict[str, Any]] = []
        self.best_model_weights: Optional[Dict[str, torch.Tensor]] = None
        self.overall_best_dice: float = -1.0

    @abstractmethod
    def _suggest_params(self, trial: Any, trial_config: TrainConfig) -> Dict[str, Any]:
        """Sample hyperparameters from ``trial`` and update ``trial_config`` if needed."""

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Instantiate a fresh model on ``self.device``."""

    @abstractmethod
    def _build_criterion(self, params: Dict[str, Any]) -> nn.Module:
        """Instantiate the loss criterion for the sampled ``params``."""

    def _build_trainer(
        self,
        trial: Any,
        model: nn.Module,
        criterion: nn.Module,
        trial_config: TrainConfig,
    ) -> Trainer | CTGCETrainer:
        return OptunaTrainer(
            trial=trial,
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            criterion=criterion,
            device=self.device,
            config=trial_config,
        )

    def _search_space(self) -> Any:
        """Return ``OptimizationConfig`` when configured, else ``TrainConfig`` fallback."""
        return self.config.optimization if self.config.optimization is not None else self.config.training

    def _prepare_trial_config(self) -> TrainConfig:
        trial_config = self.config.training.model_copy()
        trial_config.epochs = self.n_epochs_per_trial
        if trial_config.epochs_phases is not None:
            trial_config.epochs_phases = [
                p for p in trial_config.epochs_phases if p < self.n_epochs_per_trial
            ]
        return trial_config

    def objective(self, trial: Any) -> float:
        """Execute a single Optuna trial and return the best validation Dice."""
        opt = require_optuna()
        model: Optional[nn.Module] = None
        criterion: Optional[nn.Module] = None
        trainer: Optional[Any] = None

        try:
            set_seed(self.config.training.seed + trial.number)
            trial_config = self._prepare_trial_config()
            params = self._suggest_params(trial, trial_config)
            print(
                f"\n[Trial {trial.number + 1}/{self.n_trials}] params={params}",
                flush=True,
            )

            model = self._build_model().to(self.device)
            criterion = self._build_criterion(params).to(self.device)
            trainer = self._build_trainer(trial, model, criterion, trial_config)

            _, history = trainer.fit()
            metric_key = "val_dice_gt" if trial_config.ground_truth else "val_dice"
            best_val_dice = max(history[metric_key]) if history.get(metric_key) else 0.0

            if best_val_dice > self.overall_best_dice:
                self.overall_best_dice = best_val_dice
                self.best_model_weights = trainer.best_model_weights

            return float(best_val_dice)

        except opt.TrialPruned:
            raise
        except Exception as exc:
            print(f"[Trial {trial.number}] Failed with exception: {exc}")
            return 0.0
        finally:
            del model, criterion, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run(self) -> Dict[str, Any]:
        """Run the full optimization study, evaluate the best checkpoint, and persist JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.study.optimize(self.objective, n_trials=self.n_trials)

        sorted_trials = sorted(
            self.study.trials,
            key=lambda t: t.value if t.value is not None else 0.0,
            reverse=True,
        )

        trials_data: List[Dict[str, Any]] = []
        for t in sorted_trials:
            entry: Dict[str, Any] = {
                "number": t.number,
                "params": dict(t.params),
                "value": t.value,
                "state": t.state.name,
            }
            if self.primary_param_name in t.params:
                entry[self.primary_param_name] = t.params[self.primary_param_name]
            trials_data.append(entry)

        best_trial = self.study.best_trial
        best_params = dict(best_trial.params)
        best_val_dice = float(best_trial.value) if best_trial.value is not None else 0.0

        if self.best_model_weights is not None:
            best_model = self._build_model().to(self.device)
            best_model.load_state_dict(self.best_model_weights)

            tracker = MetricTracker(self.config.training)
            class_names = [f"Class {i}" for i in range(self.config.model.num_classes)]

            prob_metrics = tracker.evaluation(best_model, self.val_loader, str(self.device))
            tracker.print_full_report("Best Trial Val — Probabilistic", prob_metrics, class_names)

            if self.config.data.load_ground_truth:
                gt_metrics = tracker.evaluation_gt(best_model, self.val_loader, str(self.device))
                tracker.print_full_report("Best Trial Val — Ground Truth", gt_metrics, class_names)

        results_dict: Dict[str, Any] = {
            "best_params": best_params,
            "best_val_dice": best_val_dice,
            "trials": trials_data,
        }
        if self.primary_param_name in best_params:
            results_dict[f"best_{self.primary_param_name}"] = best_params[self.primary_param_name]

        results_path = self.output_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4)

        return results_dict
