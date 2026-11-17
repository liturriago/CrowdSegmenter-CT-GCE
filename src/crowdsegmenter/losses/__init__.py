"""Loss functions for multi-annotator segmentation."""

from crowdsegmenter.losses.ct_gce import CTGCELoss
from crowdsegmenter.losses.noisy_label import NoisyLabelLoss
from crowdsegmenter.losses.supervised import SupervisedCELoss
from crowdsegmenter.losses.tgce_ssps import TGCE_SSPS

__all__ = [
    "CTGCELoss",
    "NoisyLabelLoss",
    "SupervisedCELoss",
    "TGCE_SSPS",
]
