"""Segmentation architectures and non-parametric crowd baselines."""

from crowdsegmenter.models.annot_harmony import AnnotHarmony, AttentionAnnRel
from crowdsegmenter.models.backbone import (
    ConvBlock,
    Decoder,
    ResNet34Encoder,
    ResidualBlock,
    SegmentationHead,
    get_activation,
    kernel_initializer,
)
from crowdsegmenter.models.baselines import (
    BaselineModel,
    MajorityVoting,
    SpatialSTAPLE,
    build_potts_kernel,
    compute_entropy,
    e_step,
    initialize_parameters,
    m_step,
    run_spatial_staple,
)
from crowdsegmenter.models.crowdseg import AnnRelCM, ConvBlockAnnRelCM, CrowdSeg
from crowdsegmenter.models.ctgce import CTGCEModel, CategoryReliabilityHead
from crowdsegmenter.models.supervised import SupervisedResUNet

__all__ = [
    "AnnotHarmony",
    "AttentionAnnRel",
    "AnnRelCM",
    "BaselineModel",
    "CTGCEModel",
    "CategoryReliabilityHead",
    "ConvBlock",
    "ConvBlockAnnRelCM",
    "CrowdSeg",
    "Decoder",
    "MajorityVoting",
    "ResNet34Encoder",
    "ResidualBlock",
    "SegmentationHead",
    "SpatialSTAPLE",
    "SupervisedResUNet",
    "build_potts_kernel",
    "compute_entropy",
    "e_step",
    "get_activation",
    "initialize_parameters",
    "kernel_initializer",
    "m_step",
    "run_spatial_staple",
]
