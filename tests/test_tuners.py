"""Unit tests for Optuna hyperparameter tuners (TGCE-SSPS, NoisyLabel, CT-GCE)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from crowdsegmenter.optimization import base as opt_base
from crowdsegmenter.optimization.ct_gce_tuner import CTGCETuner
from crowdsegmenter.optimization.noisy_label_tuner import NoisyLabelTuner
from crowdsegmenter.optimization.tgce_ssps_tuner import TGCESSPSTuner


class _FakeTrialPruned(Exception):
    pass


@dataclass
class _FakeTrial:
    number: int
    params: Dict[str, Any] = field(default_factory=dict)
    reports: List[tuple[float, int]] = field(default_factory=list)
    value: float | None = None
    state: Any = field(default_factory=lambda: SimpleNamespace(name="COMPLETE"))
    prune_on_report: bool = False

    def suggest_float(self, name: str, low: float, high: float) -> float:
        val = 0.5 * (low + high)
        self.params[name] = val
        return val

    def suggest_int(self, name: str, low: int, high: int) -> int:
        val = low
        self.params[name] = val
        return val

    def suggest_categorical(self, name: str, choices: List[Any]) -> Any:
        val = choices[0]
        self.params[name] = val
        return val

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        return self.prune_on_report


class _FakeStudy:
    def __init__(self, initial_trials: List[_FakeTrial] | None = None) -> None:
        self.trials: List[_FakeTrial] = list(initial_trials or [])

    def ask(self) -> _FakeTrial:
        t = _FakeTrial(number=len(self.trials))
        self.trials.append(t)
        return t

    def optimize(self, objective, n_trials: int, callbacks=None) -> None:
        start = len(self.trials)
        for i in range(start, start + n_trials):
            t = _FakeTrial(number=i)
            try:
                t.value = objective(t)
                t.state = SimpleNamespace(name="COMPLETE")
            except _FakeTrialPruned:
                t.state = SimpleNamespace(name="PRUNED")
            self.trials.append(t)
            for cb in callbacks or ():
                cb(self, t)

    @property
    def best_trial(self) -> _FakeTrial:
        completed = [t for t in self.trials if t.value is not None]
        if not completed:
            raise ValueError("No trials are completed yet.")
        return max(completed, key=lambda t: t.value or 0.0)


def _make_fake_optuna(studies: Dict[str, _FakeStudy] | None = None):
    store = studies if studies is not None else {}

    def _create_study(study_name="study", load_if_exists=False, **kwargs):
        if load_if_exists and study_name in store:
            return store[study_name]
        st = _FakeStudy()
        store[study_name] = st
        return st

    return SimpleNamespace(
        Trial=_FakeTrial,
        TrialPruned=_FakeTrialPruned,
        create_study=_create_study,
        samplers=SimpleNamespace(TPESampler=lambda seed=42: seed),
        pruners=SimpleNamespace(MedianPruner=lambda **kwargs: kwargs),
    )


@pytest.fixture
def tiny_config_path(tmp_path: Path) -> str:
    config = {
        "data": {
            "data_dir": str(tmp_path),
            "partitions": ["Train", "Valid", "Test"],
            "num_classes": 2,
            "num_annotators": 3,
            "image_size": [32, 32],
            "batch_size": 2,
            "ignored_value": 0.6,
        },
        "model": {
            "num_classes": 2,
            "num_annotators": 3,
            "in_channels": 3,
            "image_size": 32,
            "pretrained": False,
            "decoder_channels": [32, 16, 8, 8],
            "seg_head_activation": "softmax",
            "annotator_activation": "sigmoid",
        },
        "training": {
            "epochs": 2,
            "lr": 1e-3,
            "transfer_lr": 1e-3,
            "threshold": 0.5,
            "probabilistic_thresholds": [0.5],
            "num_classes": 2,
            "num_annotators": 3,
            "ignored_value": 0.6,
            "smooth": 1e-8,
            "epochs_phases": [],
            "gamma": 1.0,
            "warmup_epochs": 5,
            "calibration_interval": 1,
            "n_epochs_per_trial": 2,
            "q_search_low": 0.2,
            "q_search_high": 0.8,
            "alpha_search_low": 0.05,
            "alpha_search_high": 0.5,
            "seed": 42,
            "device": "cpu",
        },
        "experiment": {
            "name": "tuner_test",
            "version": 1,
            "output_dir": str(tmp_path / "outputs"),
            "save_results": False,
        },
    }
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return str(cfg_path)


@pytest.fixture
def dummy_loaders():
    torch.manual_seed(0)
    images = torch.randn(2, 3, 32, 32)
    labels = torch.randint(0, 2, (2, 3, 32, 32))
    masks = torch.nn.functional.one_hot(labels, num_classes=2).permute(0, 4, 1, 2, 3).float()
    masks = masks.reshape(2, 6, 32, 32)
    anns_ids = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    loader = DataLoader(TensorDataset(images, masks, anns_ids), batch_size=2)
    with patch("crowdsegmenter.optimization.base.CrowdSegmenterDataLoader") as mock_dm:
        mock_dm.return_value.get_split_loaders.return_value = (loader, loader, loader)
        yield


def test_require_optuna_raises_clear_error_when_missing() -> None:
    with patch.object(opt_base, "optuna", None):
        with pytest.raises(ImportError, match="optuna is required"):
            opt_base.require_optuna()


@pytest.mark.parametrize(
    "tuner_cls,param_key,expected_min,expected_max",
    [
        (TGCESSPSTuner, "q", 0.2, 0.8),
        (NoisyLabelTuner, "alpha", 0.05, 0.5),
        (CTGCETuner, "q", 0.2, 0.8),
    ],
)
def test_all_three_tuners_run_and_persist_results(
    tuner_cls,
    param_key: str,
    expected_min: float,
    expected_max: float,
    tiny_config_path: str,
    dummy_loaders,
    tmp_path: Path,
) -> None:
    fake_optuna = _make_fake_optuna()
    out_dir = tmp_path / tuner_cls.__name__
    with patch.object(opt_base, "optuna", fake_optuna):
        tuner = tuner_cls(
            config_path=tiny_config_path,
            n_trials=2,
            n_epochs_per_trial=2,
            output_dir=str(out_dir),
        )
        results = tuner.run()

    assert f"best_{param_key}" in results
    assert expected_min <= results[f"best_{param_key}"] <= expected_max
    assert 0.0 <= results["best_val_dice"] <= 1.0
    assert len(results["trials"]) == 2
    assert (out_dir / "results.json").is_file()
    with open(out_dir / "results.json", "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["best_params"][param_key] == results[f"best_{param_key}"]


def test_ctgce_tuner_schedule_search_and_pruning(
    tiny_config_path: str,
    dummy_loaders,
    tmp_path: Path,
) -> None:
    with open(tiny_config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["training"]["tune_schedule"] = True
    raw["training"]["warmup_search_low"] = 0
    raw["training"]["warmup_search_high"] = 2
    raw["training"]["calib_interval_choices"] = [1, 2]
    sched_cfg = tmp_path / "sched_config.yaml"
    with open(sched_cfg, "w", encoding="utf-8") as f:
        yaml.dump(raw, f)

    fake_optuna = _make_fake_optuna()
    with patch.object(opt_base, "optuna", fake_optuna):
        tuner = CTGCETuner(
            config_path=str(sched_cfg),
            n_trials=1,
            n_epochs_per_trial=2,
            output_dir=str(tmp_path / "sched_out"),
        )
        results = tuner.run()
        assert "warmup_epochs" in results["best_params"]
        assert "calibration_interval" in results["best_params"]
        assert results["best_params"]["warmup_epochs"] <= 1

        # Verify pruning fires cleanly via _OptunaPruningMixin
        pruned_trial = _FakeTrial(number=99, prune_on_report=True)
        with pytest.raises(_FakeTrialPruned):
            tuner.objective(pruned_trial)


def test_tuner_reads_optimization_config_section(dummy_loaders, tmp_path: Path) -> None:
    opt_cfg_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "optimization"
        / "ctgce"
        / "oxfordiiitpet.yaml"
    )
    fake_optuna = _make_fake_optuna()
    with patch.object(opt_base, "optuna", fake_optuna):
        tuner = CTGCETuner(config_path=opt_cfg_path, output_dir=str(tmp_path / "opt_out"))
        assert tuner.n_trials == 30
        assert tuner.n_epochs_per_trial == 8
        assert tuner.study_name == "ct_gce_oxford_iiitpet"


def test_incremental_persistence_per_trial_and_resume(
    tiny_config_path: str,
    dummy_loaders,
    tmp_path: Path,
) -> None:
    store: Dict[str, _FakeStudy] = {}
    fake_optuna = _make_fake_optuna(store)
    out_dir = tmp_path / "resume_out"

    with patch.object(opt_base, "optuna", fake_optuna):
        # Run 1 trial first
        t1 = CTGCETuner(
            config_path=tiny_config_path,
            n_trials=1,
            n_epochs_per_trial=1,
            study_name="resume_study",
            output_dir=str(out_dir),
        )
        t1.run()
        assert (out_dir / "results.json").is_file()
        assert (out_dir / "best_trial_weights.pth").is_file()

        # Resume up to n_trials=3 -> only 2 additional trials should execute
        t2 = CTGCETuner(
            config_path=tiny_config_path,
            n_trials=3,
            n_epochs_per_trial=1,
            study_name="resume_study",
            output_dir=str(out_dir),
        )
        res = t2.run()
        assert len(res["trials"]) == 3
