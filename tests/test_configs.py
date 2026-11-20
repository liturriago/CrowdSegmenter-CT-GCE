"""Validation tests for all YAML training/optimization configs and Pydantic rules."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from crowdsegmenter.config import (
    DataConfig,
    ExperimentConfig,
    ExperimentMetadata,
    ModelConfig,
    OptimizationConfig,
    TrainConfig,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
TRAIN_CONFIG_ROOT = CONFIG_DIR / "training"
OPT_CONFIG_ROOT = CONFIG_DIR / "optimization"

ALL_TRAIN_CONFIG_PATHS = sorted(TRAIN_CONFIG_ROOT.glob("*/*.yaml"))
ALL_OPT_CONFIG_PATHS = sorted(OPT_CONFIG_ROOT.glob("*/*.yaml"))


def test_expected_training_configs_exist() -> None:
    rel = {p.relative_to(TRAIN_CONFIG_ROOT).as_posix() for p in ALL_TRAIN_CONFIG_PATHS}
    expected = {
        "MV/oxfordiiitpet.yaml",
        "MV/riga.yaml",
        "MV/naip_wellpad.yaml",
        "spatial_staple/oxfordiiitpet.yaml",
        "spatial_staple/riga.yaml",
        "spatial_staple/naip_wellpad.yaml",
        "crowdseg/oxfordiiitpet.yaml",
        "crowdseg/riga.yaml",
        "crowdseg/naip_wellpad.yaml",
        "tgce_ssps/oxfordiiitpet.yaml",
        "tgce_ssps/riga.yaml",
        "tgce_ssps/naip_wellpad.yaml",
        "ctgce/oxfordiiitpet.yaml",
        "ctgce/riga.yaml",
        "ctgce/naip_wellpad.yaml",
        "supervised/oxfordiiitpet.yaml",
        "supervised/naip_wellpad.yaml",
    }
    assert expected.issubset(rel)


def test_expected_optimization_configs_exist() -> None:
    rel = {p.relative_to(OPT_CONFIG_ROOT).as_posix() for p in ALL_OPT_CONFIG_PATHS}
    expected = {
        "crowdseg/oxfordiiitpet.yaml",
        "crowdseg/riga.yaml",
        "crowdseg/naip_wellpad.yaml",
        "tgce_ssps/oxfordiiitpet.yaml",
        "tgce_ssps/riga.yaml",
        "tgce_ssps/naip_wellpad.yaml",
        "ctgce/oxfordiiitpet.yaml",
        "ctgce/riga.yaml",
        "ctgce/naip_wellpad.yaml",
    }
    assert expected.issubset(rel)


@pytest.mark.parametrize(
    "cfg_path",
    ALL_TRAIN_CONFIG_PATHS,
    ids=lambda p: p.relative_to(TRAIN_CONFIG_ROOT).as_posix(),
)
def test_every_training_yaml_validates_with_pydantic(cfg_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(cfg_path)
    assert cfg.data.load_ground_truth == cfg.training.ground_truth
    assert cfg.data.num_classes == cfg.model.num_classes == cfg.training.num_classes
    assert cfg.data.num_annotators == cfg.model.num_annotators == cfg.training.num_annotators


@pytest.mark.parametrize(
    "cfg_path",
    ALL_OPT_CONFIG_PATHS,
    ids=lambda p: p.relative_to(OPT_CONFIG_ROOT).as_posix(),
)
def test_every_optimization_yaml_validates_with_pydantic(cfg_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(cfg_path)
    assert cfg.optimization is not None
    assert cfg.optimization.n_trials > 0
    assert cfg.optimization.n_epochs_per_trial > 0
    assert cfg.data.load_ground_truth == cfg.training.ground_truth
    assert cfg.data.num_classes == cfg.model.num_classes == cfg.training.num_classes
    assert cfg.data.num_annotators == cfg.model.num_annotators == cfg.training.num_annotators


def test_cross_section_mismatch_and_invalid_opt_bounds_fail_validation(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="num_classes"):
        ExperimentConfig(
            data=DataConfig(data_dir=str(tmp_path), num_classes=2),
            model=ModelConfig(num_classes=3),
            training=TrainConfig(num_classes=2),
            experiment=ExperimentMetadata(),
        )

    with pytest.raises(ValidationError, match="CT-GCE requires"):
        ExperimentConfig(
            data=DataConfig(data_dir=str(tmp_path)),
            model=ModelConfig(seg_head_activation="sigmoid"),
            training=TrainConfig(model_name="CT-GCE"),
            experiment=ExperimentMetadata(),
        )

    with pytest.raises(ValidationError, match="q_search_low"):
        OptimizationConfig(q_search_low=0.9, q_search_high=0.2)
