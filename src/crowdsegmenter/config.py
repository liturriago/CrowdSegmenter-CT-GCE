"""Typed experiment configuration schemas backed by Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    """Configuration for multi-annotator data loading and preprocessing."""

    data_dir: str = Field(..., description="Path to the root data directory")
    partitions: List[str] = Field(
        default=["Train", "Valid", "Test"],
        description="Partitions to load",
    )
    images_folder: str = Field(default="patches", description="Folder containing images")
    masks_folder: str = Field(default="masks", description="Folder containing annotator masks")
    ground_truth_folder: str = Field(
        default="ground_truth",
        description="Folder containing ground-truth masks",
    )

    load_annotators: bool = Field(default=True, description="Whether to load annotator masks")
    load_ground_truth: bool = Field(
        default=False,
        description="Whether to load ground-truth masks",
    )

    num_classes: int = Field(default=2, description="Number of segmentation classes (K)", gt=0)
    num_annotators: int = Field(default=3, description="Number of annotators (R)", gt=0)

    image_size: List[int] = Field(
        default=[256, 256],
        description="Image resizing dimensions (H, W)",
    )
    enh_aug: bool = Field(
        default=False,
        description="Enable enhanced Train augmentations (RandomCrop, Rotate90, HSV, GaussianBlur)",
    )
    enh_crop_size: Optional[List[int]] = Field(
        default=[512, 512],
        description="RandomCrop [H, W] applied on native Train samples before resizing when enh_aug=True",
    )
    ignored_value: float = Field(default=0.6, description="Sentinel value for missing annotations")
    normalize: bool = Field(default=True, description="Whether to scale inputs to [0, 1]")
    transforms: bool = Field(default=True, description="Whether to apply data augmentation/norm")

    mean: Optional[List[float]] = Field(
        default=[0.485, 0.456, 0.406],
        description="Channel-wise mean for image normalization",
    )
    std: Optional[List[float]] = Field(
        default=[0.229, 0.224, 0.225],
        description="Channel-wise standard deviation for image normalization",
    )

    batch_size: int = Field(default=16, description="Batch size", gt=0)
    num_workers: int = Field(default=4, description="Number of DataLoader workers", ge=0)
    pin_memory: bool = Field(default=True, description="Whether to pin memory in DataLoader")
    cache_val: bool = Field(
        default=False,
        description="Cache deterministic Valid samples in memory after first pass (for Optuna speedups)",
    )
    prefetch_factor: Optional[int] = Field(
        default=2,
        description="Prefetch factor for DataLoader when num_workers > 0",
        ge=1,
    )


class ModelConfig(BaseModel):
    """Configuration for segmentation models and baselines."""

    num_annotators: int = Field(default=3, description="Number of annotators (R)", gt=0)
    in_channels: int = Field(default=3, description="Number of input image channels", gt=0)
    num_classes: int = Field(default=2, description="Number of classes (K)", gt=0)
    image_size: int = Field(
        default=256,
        description="Spatial dimension of the input image (assumed square)",
        gt=0,
    )
    pretrained: bool = Field(default=True, description="Whether to use ImageNet-pretrained weights")
    decoder_channels: List[int] = Field(
        default=[256, 128, 64, 64],
        description="Decoder channel dimensions",
    )
    use_residual: bool = Field(default=True, description="Whether to use residual decoder blocks")
    seg_head_activation: Optional[Literal["sigmoid", "softmax", "tanh", "relu"]] = Field(
        default="softmax",
        description="Segmentation head activation",
    )
    annotator_activation: Optional[Literal["sigmoid", "softmax", "tanh", "relu"]] = Field(
        default="sigmoid",
        description="Annotator reliability head activation",
    )

    # Spatial STAPLE parameters
    staple_beta: float = Field(
        default=1.0,
        description="Potts MRF spatial weight (0.0 reduces to classical STAPLE)",
        ge=0.0,
    )
    staple_max_iter: int = Field(default=20, description="Max EM iterations for Spatial STAPLE", gt=0)
    staple_tol: float = Field(default=1e-4, description="EM convergence tolerance", gt=0.0)


class TrainConfig(BaseModel):
    """Configuration for training, evaluation, and loss hyperparameters."""

    model_name: Optional[
        Literal["MV", "Spatial-STAPLE", "Annot-Harmony", "CrowdSeg", "CT-GCE", "Supervised-GT"]
    ] = Field(default=None, description="Model / method name")
    epochs: int = Field(default=40, description="Total number of training epochs", gt=0)
    lr: float = Field(default=1e-3, description="Primary learning rate", gt=0.0)
    transfer_lr: float = Field(default=1e-4, description="Encoder/decoder fine-tuning LR", gt=0.0)
    branch_lr: float = Field(
        default=1e-3,
        description="Learning rate for CT-GCE reliability branch phi",
        gt=0.0,
    )
    device: str = Field(default="cuda", description="Target device ('cuda' or 'cpu')")

    threshold: float = Field(default=0.5, description="Binarization threshold", ge=0.0, le=1.0)
    probabilistic_thresholds: List[float] = Field(
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        description="Thresholds for probabilistic evaluation",
    )
    num_classes: int = Field(default=2, description="Number of classes (K)", gt=0)
    num_annotators: int = Field(default=3, description="Number of annotators (R)", gt=0)
    ignored_value: float = Field(default=0.6, description="Sentinel value for missing annotations")
    smooth: float = Field(default=1e-8, description="Numerical smoothing constant", ge=0.0)
    ground_truth: bool = Field(
        default=False,
        description="Whether to run ground-truth evaluation alongside consensus metrics",
    )
    eval_nlc: bool = Field(
        default=False,
        description="Whether to compute zero-GT Neighborhood Latent Consistency (NLC) on Test",
    )
    epochs_phases: Optional[List[int]] = Field(
        default=[0, 5, 10, 15],
        description="Epochs at which to advance curriculum unfreezing phases",
    )
    gamma: float = Field(default=0.94, description="Exponential LR decay factor", gt=0.0, le=1.0)

    # TGCE-SSPS (Annot-Harmony)
    tgce_q: float = Field(default=0.6704, description="Truncation q for TGCE-SSPS", gt=0.0, le=1.0)
    tgce_lambda: float = Field(default=1.0, description="Reliability scaling factor", gt=0.0)

    # CT-GCE (r10)
    ctgce_q: float = Field(default=0.7, description="Truncation q in (0, 1] for CT-GCE", gt=0.0, le=1.0)
    warmup_epochs: int = Field(
        default=5,
        description="Plain crowd GCE warm-up epochs before enabling CT-GCE truncation",
        ge=0,
    )
    calibration_interval: int = Field(
        default=1,
        description="Epochs between Eq. (4) cutoff recalibrations after warm-up",
        gt=0,
    )
    calibration_max_batches: Optional[int] = Field(
        default=None,
        description="Optional cap on train batches scanned per Dawid-Skene / cutoff calibration pass",
        gt=0,
    )

    # CrowdSeg (NoisyLabelLoss)
    min_trace: bool = Field(
        default=True,
        description="Whether to add (+alpha*trace) or subtract (-alpha*trace)",
    )
    alpha: float = Field(default=0.1, description="Trace regularization weight")

    # Optuna search ranges
    n_epochs_per_trial: int = Field(default=5, description="Epochs per Optuna trial", gt=0)
    q_search_low: float = Field(default=0.1, gt=0.0, le=1.0)
    q_search_high: float = Field(default=0.99, gt=0.0, le=1.0)
    alpha_search_low: float = Field(default=0.01, gt=0.0)
    alpha_search_high: float = Field(default=2.0, gt=0.0)
    tune_schedule: bool = Field(
        default=False,
        description="Whether CTGCETuner also searches warmup_epochs and calibration_interval",
    )
    warmup_search_low: int = Field(default=1, ge=0)
    warmup_search_high: int = Field(default=10, ge=0)
    calib_interval_choices: List[int] = Field(
        default=[1, 2, 5],
        description="Candidate calibration intervals when tune_schedule=True",
    )

    seed: int = Field(default=42, description="Random seed for reproducibility")

    @model_validator(mode="after")
    def _validate_search_bounds(self) -> "TrainConfig":
        if self.q_search_low > self.q_search_high:
            raise ValueError("q_search_low must be <= q_search_high")
        if self.alpha_search_low > self.alpha_search_high:
            raise ValueError("alpha_search_low must be <= alpha_search_high")
        if self.warmup_search_low > self.warmup_search_high:
            raise ValueError("warmup_search_low must be <= warmup_search_high")
        return self


class OptimizationConfig(BaseModel):
    """Configuration for Optuna hyperparameter optimization studies."""

    n_trials: int = Field(default=30, description="Number of Optuna trials", gt=0)
    n_epochs_per_trial: int = Field(default=5, description="Epochs per Optuna trial", gt=0)
    study_name: str = Field(default="optuna_study", description="Optuna study name")
    pruner_warmup_steps: int = Field(
        default=3,
        ge=0,
        description="MedianPruner warm-up steps before pruning is allowed",
    )
    q_search_low: float = Field(default=0.1, gt=0.0, le=1.0)
    q_search_high: float = Field(default=0.99, gt=0.0, le=1.0)
    alpha_search_low: float = Field(default=0.01, gt=0.0)
    alpha_search_high: float = Field(default=2.0, gt=0.0)
    tune_schedule: bool = Field(
        default=False,
        description="Whether CTGCETuner also searches warmup_epochs and calibration_interval",
    )
    warmup_search_low: int = Field(default=1, ge=0)
    warmup_search_high: int = Field(default=10, ge=0)
    calib_interval_choices: List[int] = Field(
        default=[1, 2, 5],
        description="Candidate calibration intervals when tune_schedule=True",
    )

    @model_validator(mode="after")
    def _validate_search_bounds(self) -> "OptimizationConfig":
        if self.q_search_low > self.q_search_high:
            raise ValueError("q_search_low must be <= q_search_high")
        if self.alpha_search_low > self.alpha_search_high:
            raise ValueError("alpha_search_low must be <= alpha_search_high")
        if self.warmup_search_low > self.warmup_search_high:
            raise ValueError("warmup_search_low must be <= warmup_search_high")
        if not self.calib_interval_choices or any(c <= 0 for c in self.calib_interval_choices):
            raise ValueError("calib_interval_choices must contain positive integers")
        return self


class ExperimentMetadata(BaseModel):
    """Metadata for experiment tracking and output serialization."""

    name: str = "base_experiment"
    version: int = Field(default=1, ge=0)
    output_dir: Path = Path("results/experiment_1")
    save_results: bool = False


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    data: DataConfig
    model: ModelConfig
    training: TrainConfig
    experiment: ExperimentMetadata
    optimization: Optional[OptimizationConfig] = Field(
        default=None,
        description="Optional Optuna hyperparameter optimization configuration",
    )

    @model_validator(mode="after")
    def _validate_cross_section_consistency(self) -> "ExperimentConfig":
        if not (self.data.num_classes == self.model.num_classes == self.training.num_classes):
            raise ValueError(
                "num_classes must match across data, model, and training: "
                f"got ({self.data.num_classes}, {self.model.num_classes}, {self.training.num_classes})"
            )
        if not (
            self.data.num_annotators == self.model.num_annotators == self.training.num_annotators
        ):
            raise ValueError(
                "num_annotators must match across data, model, and training: "
                f"got ({self.data.num_annotators}, {self.model.num_annotators}, "
                f"{self.training.num_annotators})"
            )
        if self.data.ignored_value != self.training.ignored_value:
            raise ValueError(
                "ignored_value must match between data and training: "
                f"got {self.data.ignored_value} vs {self.training.ignored_value}"
            )
        if len(self.data.image_size) != 2 or any(d <= 0 for d in self.data.image_size):
            raise ValueError(f"data.image_size must be [H, W] > 0; got {self.data.image_size}")
        if self.data.enh_crop_size is not None and (
            len(self.data.enh_crop_size) != 2 or any(d <= 0 for d in self.data.enh_crop_size)
        ):
            raise ValueError(
                f"data.enh_crop_size must be [H, W] > 0 or None; got {self.data.enh_crop_size}"
            )
        if self.model.image_size != self.data.image_size[0]:
            raise ValueError(
                f"model.image_size ({self.model.image_size}) must match "
                f"data.image_size[0] ({self.data.image_size[0]})"
            )
        if self.training.model_name == "CT-GCE" and self.model.seg_head_activation != "softmax":
            raise ValueError("CT-GCE requires model.seg_head_activation='softmax'")
        return self

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExperimentConfig":
        """Load and validate an experiment configuration from a YAML file."""
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        return cls(**raw_config)
