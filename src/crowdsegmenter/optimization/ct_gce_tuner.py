"""Bayesian hyperparameter tuner for ``CTGCELoss`` with ``CTGCEModel``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from crowdsegmenter.config import TrainConfig
from crowdsegmenter.losses.ct_gce import CTGCELoss
from crowdsegmenter.models.ctgce import CTGCEModel
from crowdsegmenter.optimization.base import BaseOptunaTuner, OptunaCTGCETrainer


class CTGCETuner(BaseOptunaTuner):
    """Optuna tuner for ``CTGCEModel`` + ``CTGCELoss`` (r10 decoupled training).

    Searches the truncation exponent ``q`` in ``[q_search_low, q_search_high]``.
    When ``tune_schedule`` is ``True`` (in ``optimization`` or ``training``),
    also searches ``warmup_epochs`` and ``calibration_interval``.
    """

    data_mode = "CT-GCE"
    primary_param_name = "q"
    default_study_name = "ct_gce_optimization"
    default_output_dir = "results/optuna/ct_gce"

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
        super().__init__(
            config_path=config_path,
            n_trials=n_trials,
            n_epochs_per_trial=n_epochs_per_trial,
            study_name=study_name,
            output_dir=output_dir,
            seed=seed,
            data_dir=data_dir,
        )
        self._cached_rho_hat: Optional[torch.Tensor] = None

    def _suggest_params(self, trial: Any, trial_config: TrainConfig) -> Dict[str, Any]:
        space = self._search_space()
        q = trial.suggest_float("q", space.q_search_low, space.q_search_high)
        params: Dict[str, Any] = {"q": q}

        max_warmup = max(0, self.n_epochs_per_trial - 1)
        if space.tune_schedule:
            low = min(space.warmup_search_low, max_warmup)
            high = min(space.warmup_search_high, max_warmup)
            warmup = trial.suggest_int("warmup_epochs", low, high)
            calib = trial.suggest_categorical(
                "calibration_interval",
                list(space.calib_interval_choices),
            )
            trial_config.warmup_epochs = warmup
            trial_config.calibration_interval = int(calib)
            params["warmup_epochs"] = warmup
            params["calibration_interval"] = int(calib)
        elif trial_config.warmup_epochs >= self.n_epochs_per_trial:
            # Ensure at least one CT-GCE truncated epoch executes within each trial
            trial_config.warmup_epochs = max_warmup

        return params

    def _build_model(self) -> nn.Module:
        return CTGCEModel(self.config.model)

    def _build_criterion(self, params: Dict[str, Any]) -> nn.Module:
        return CTGCELoss(
            annotators=self.config.model.num_annotators,
            classes=self.config.model.num_classes,
            ignored_value=self.config.data.ignored_value,
            q=params["q"],
        )

    def _build_trainer(
        self,
        trial: Any,
        model: nn.Module,
        criterion: nn.Module,
        trial_config: TrainConfig,
    ) -> OptunaCTGCETrainer:
        assert isinstance(criterion, CTGCELoss)
        trainer = OptunaCTGCETrainer(
            trial=trial,
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            criterion=criterion,
            device=self.device,
            config=trial_config,
            precomputed_rho_hat=self._cached_rho_hat,
        )
        if self._cached_rho_hat is None:
            self._cached_rho_hat = trainer._estimate_dataset_noise_rates()
            trainer.rho_hat = self._cached_rho_hat.clone()
        return trainer
