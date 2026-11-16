"""Bayesian hyperparameter tuner for ``TGCE_SSPS`` with ``AnnotHarmony``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch.nn as nn

from crowdsegmenter.config import TrainConfig
from crowdsegmenter.losses.tgce_ssps import TGCE_SSPS
from crowdsegmenter.models.annot_harmony import AnnotHarmony
from crowdsegmenter.optimization.base import BaseOptunaTuner


class TGCESSPSTuner(BaseOptunaTuner):
    """Optuna tuner searching ``q`` for ``AnnotHarmony`` + ``TGCE_SSPS``."""

    data_mode = "Annot-Harmony"
    primary_param_name = "q"
    default_study_name = "tgce_ssps_q_optimization"
    default_output_dir = "results/optuna/tgce_ssps"

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
        q = trial.suggest_float("q", space.q_search_low, space.q_search_high)
        return {"q": q}

    def _build_model(self) -> nn.Module:
        return AnnotHarmony(self.config.model)

    def _build_criterion(self, params: Dict[str, Any]) -> nn.Module:
        return TGCE_SSPS(
            annotators=self.config.model.num_annotators,
            classes=self.config.model.num_classes,
            ignored_value=self.config.data.ignored_value,
            q=params["q"],
            lambda_factor=self.config.training.tgce_lambda,
            smooth=self.config.training.smooth,
        )
