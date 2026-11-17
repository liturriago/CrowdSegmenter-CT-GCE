"""Bayesian hyperparameter tuner for ``NoisyLabelLoss`` with ``CrowdSeg``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch.nn as nn

from crowdsegmenter.config import TrainConfig
from crowdsegmenter.losses.noisy_label import NoisyLabelLoss
from crowdsegmenter.models.crowdseg import CrowdSeg
from crowdsegmenter.optimization.base import BaseOptunaTuner


class NoisyLabelTuner(BaseOptunaTuner):
    """Optuna tuner searching trace weight ``alpha`` for ``CrowdSeg`` + ``NoisyLabelLoss``."""

    data_mode = "CrowdSeg"
    primary_param_name = "alpha"
    default_study_name = "noisy_label_alpha_optimization"
    default_output_dir = "results/optuna/noisy_label"

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

    def _suggest_params(self, trial: Any, trial_config: TrainConfig) -> Dict[str, Any]:
        space = self._search_space()
        # Sample log-uniformly so small scales (0.01-0.1) are explored alongside 0.1-2.0.
        alpha = trial.suggest_float(
            "alpha", space.alpha_search_low, space.alpha_search_high, log=True
        )
        return {"alpha": alpha}

    def _build_model(self) -> nn.Module:
        return CrowdSeg(self.config.model)

    def _build_criterion(self, params: Dict[str, Any]) -> nn.Module:
        return NoisyLabelLoss(
            annotators=self.config.model.num_annotators,
            classes=self.config.model.num_classes,
            ignored_value=self.config.data.ignored_value,
            min_trace=False,
            alpha=params["alpha"],
            smooth=self.config.training.smooth,
        )
