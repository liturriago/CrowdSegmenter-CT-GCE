"""Unit and integration tests for the ``crowdsegmenter`` MLOps modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from crowdsegmenter.config import DataConfig, ExperimentConfig, ExperimentMetadata, ModelConfig, TrainConfig
from crowdsegmenter.data.loader import CrowdSegmenterDataLoader, CrowdSegmenterDataset
from crowdsegmenter.losses import CTGCELoss, NoisyLabelLoss, TGCE_SSPS
from crowdsegmenter.models import (
    AnnotHarmony,
    CTGCEModel,
    CrowdSeg,
    MajorityVoting,
    SpatialSTAPLE,
)
from crowdsegmenter.training import CTGCETrainer, Trainer
from crowdsegmenter.utils import MetricTracker, compute_nlc_panel_metrics, set_seed


def _make_masks_batch(b: int = 2, r: int = 3, k: int = 2, h: int = 16, w: int = 16) -> torch.Tensor:
    """Create valid one-hot masks in layout ``[B, R*K, H, W]`` (channel ``k*R + r``)."""
    torch.manual_seed(0)
    labels = torch.randint(0, k, (b, r, h, w))
    onehot = torch.nn.functional.one_hot(labels, num_classes=k).permute(0, 4, 1, 2, 3).float()
    return onehot.reshape(b, k * r, h, w)


def test_majority_voting_and_spatial_staple_shapes_and_ignored_pixels() -> None:
    masks = _make_masks_batch(b=2, r=3, k=2, h=16, w=16)
    # Mark annotator 0 as missing on top-left corner
    masks[:, 0::3, :4, :4] = 0.6
    anns_ids = torch.ones(2, 3)

    mv = MajorityVoting(num_annotators=3, num_classes=2, ignored_value=0.6)
    mv_out, mv_aux = mv(masks, anns_ids)
    assert mv_aux is None
    assert mv_out.shape == (2, 2, 16, 16)
    assert torch.allclose(mv_out.sum(dim=1), torch.ones(2, 16, 16), atol=1e-5)

    staple = SpatialSTAPLE(num_annotators=3, num_classes=2, ignored_value=0.6, beta=1.0, max_iter=5)
    st_out, st_aux = staple(masks, anns_ids)
    assert st_aux is None
    assert st_out.shape == (2, 2, 16, 16)
    assert torch.all(torch.isfinite(st_out))
    assert torch.allclose(st_out.sum(dim=1), torch.ones(2, 16, 16), atol=1e-5)


def test_neural_models_forward_contracts() -> None:
    cfg = ModelConfig(
        num_annotators=3,
        in_channels=3,
        num_classes=2,
        image_size=32,
        pretrained=False,
        decoder_channels=[32, 16, 8, 8],
        seg_head_activation="softmax",
        annotator_activation="sigmoid",
    )
    x = torch.randn(2, 3, 32, 32)
    multihot = torch.ones(2, 3)
    onehot = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    ah = AnnotHarmony(cfg)
    ah_seg, ah_ann = ah(x, multihot)
    assert ah_seg.shape == (2, 2, 32, 32)
    assert ah_ann is not None and ah_ann.shape == (2, 3, 32, 32)

    cs = CrowdSeg(cfg)
    cs_seg, cs_cm = cs(x, onehot)
    assert cs_seg.shape == (2, 2, 32, 32)
    assert cs_cm is not None and cs_cm.shape == (2, 4, 32, 32)

    ct = CTGCEModel(cfg)
    ct_seg, ct_lam = ct(x, multihot)
    assert ct_seg.shape == (2, 2, 32, 32)
    assert torch.allclose(ct_seg.sum(dim=1), torch.ones(2, 32, 32), atol=1e-5)
    assert ct_lam is not None and ct_lam.shape == (2, 3, 2, 32, 32)
    assert bool((ct_lam >= 0.0).all()) and bool((ct_lam <= 1.0).all())


def test_losses_and_ctgce_decoupled_gradients() -> None:
    masks = _make_masks_batch(b=2, r=3, k=2, h=16, w=16)
    masks[:, 0::3, :2, :2] = 0.6

    seg_logits = torch.randn(2, 2, 16, 16, requires_grad=True)
    seg_pred = torch.softmax(seg_logits, dim=1)

    # 1. NoisyLabelLoss
    cms = torch.rand(2, 4, 16, 16, requires_grad=True)
    onehot = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    nl = NoisyLabelLoss(annotators=3, classes=2, ignored_value=0.6)
    nl_val = nl(seg_pred, cms, masks, onehot)
    assert torch.isfinite(nl_val)

    # 2. TGCE_SSPS
    ann_pred = torch.rand(2, 3, 16, 16, requires_grad=True)
    tgce = TGCE_SSPS(annotators=3, classes=2, ignored_value=0.6)
    tgce_val = tgce(seg_pred, ann_pred, masks)
    assert torch.isfinite(tgce_val)

    # 3. CTGCELoss: Dawid-Skene + calibration + decoupled gradients
    ct_loss = CTGCELoss(annotators=3, classes=2, ignored_value=0.6, q=0.7)
    rho = ct_loss.estimate_noise_rates(masks, max_iter=20)
    assert rho.shape == (2,)
    assert bool(torch.isfinite(rho).all())

    calib = ct_loss.calibrate(seg_pred, masks, rho)
    assert calib.cutoffs.shape == (2,)
    assert bool((calib.cutoffs >= 0.5).all())

    lam_raw = torch.randn(2, 3, 2, 16, 16, requires_grad=True)
    lam = torch.sigmoid(lam_raw)

    # Branch loss must NOT backprop into seg_logits
    b_loss = ct_loss.reliability_loss(seg_pred, lam, masks)
    b_loss.backward(retain_graph=True)
    assert seg_logits.grad is None
    assert lam_raw.grad is not None and torch.any(lam_raw.grad != 0)

    # Segmentation loss must NOT backprop into lam_raw
    lam_raw.grad.zero_()
    s_loss = ct_loss.segmentation_loss(seg_pred, lam, masks, calib.cutoffs)
    s_loss.backward()
    assert seg_logits.grad is not None and torch.any(seg_logits.grad != 0)
    assert torch.all(lam_raw.grad == 0)


def test_dataset_dataloader_and_metrics(tmp_path: Path) -> None:
    set_seed(123)
    for split in ("Train", "Valid", "Test"):
        (tmp_path / split / "patches").mkdir(parents=True)
        for folder in ("annotator_0", "annotator_1", "annotator_2", "ground_truth"):
            for cls in ("class_0", "class_1"):
                (tmp_path / split / "masks" / folder / cls).mkdir(parents=True)

        for idx in range(2):
            name = f"img_{idx}.png"
            img = np.random.randint(0, 255, (24, 24, 3), dtype=np.uint8)
            Image.fromarray(img).save(tmp_path / split / "patches" / name)

            gt_c1 = np.zeros((24, 24), dtype=np.uint8)
            gt_c1[6:18, 6:18] = 255
            gt_c0 = 255 - gt_c1
            Image.fromarray(gt_c0).save(tmp_path / split / "masks" / "ground_truth" / "class_0" / name)
            Image.fromarray(gt_c1).save(tmp_path / split / "masks" / "ground_truth" / "class_1" / name)

            for r in range(3):
                Image.fromarray(gt_c0).save(tmp_path / split / "masks" / f"annotator_{r}" / "class_0" / name)
                Image.fromarray(gt_c1).save(tmp_path / split / "masks" / f"annotator_{r}" / "class_1" / name)

    d_cfg = DataConfig(
        data_dir=str(tmp_path),
        partitions=["Train", "Valid", "Test"],
        load_annotators=True,
        load_ground_truth=True,
        num_classes=2,
        num_annotators=3,
        image_size=[32, 32],
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )
    dm = CrowdSegmenterDataLoader(d_cfg, mode="CT-GCE")
    train_loader, val_loader, test_loader = dm.get_split_loaders()
    batch = next(iter(train_loader))
    assert len(batch) == 4
    assert batch[0].shape == (2, 3, 32, 32)
    assert batch[1].shape == (2, 6, 32, 32)
    assert batch[2].shape == (2, 3)
    assert batch[3].shape == (2, 2, 32, 32)

    # Check CrowdSeg mode: Train samples 1 annotator (one-hot), Valid/Test load all annotators
    cs_dm = CrowdSegmenterDataLoader(d_cfg, mode="CrowdSeg")
    cs_train, cs_val, cs_test = cs_dm.get_split_loaders()
    _, _, cs_tr_ids, _ = next(iter(cs_train))
    _, _, cs_va_ids, _ = next(iter(cs_val))
    _, _, cs_te_ids, _ = next(iter(cs_test))
    assert torch.allclose(cs_tr_ids.sum(dim=1), torch.ones(cs_tr_ids.shape[0]))
    assert torch.allclose(cs_va_ids.sum(dim=1), torch.full((cs_va_ids.shape[0],), 3.0))
    assert torch.allclose(cs_te_ids.sum(dim=1), torch.full((cs_te_ids.shape[0],), 3.0))

    # Check MetricTracker probability mask respects k*A + r layout
    t_cfg = TrainConfig(num_classes=2, num_annotators=3, ignored_value=0.6)
    tracker = MetricTracker(t_cfg)
    prob_mask = tracker.compute_probability_mask(batch[1])
    assert prob_mask.shape == (2, 2, 32, 32)
    assert torch.allclose(prob_mask, batch[3], atol=1e-4)

    # Check NLC zero-GT panel metric bridge
    rng = np.random.default_rng(7)
    feats = [rng.normal(size=(8, 16, 16)).astype(np.float32) for _ in range(3)]
    m_good = np.zeros((16, 16), dtype=np.float32)
    m_good[4:12, 4:12] = 1.0
    nlc = compute_nlc_panel_metrics(feats, [[m_good] * 3, [m_good] * 3])
    assert set(nlc.keys()) == {"per_annotator", "panel_mean", "panel_q10", "panel_median"}
    assert len(nlc["per_annotator"]) == 2


def test_trainers_smoke_fit_cpu() -> None:
    images = torch.randn(4, 3, 32, 32)
    masks = _make_masks_batch(b=4, r=3, k=2, h=32, w=32)
    anns_ids = torch.ones(4, 3)
    loader = DataLoader(TensorDataset(images, masks, anns_ids), batch_size=2)

    m_cfg = ModelConfig(
        num_annotators=3,
        in_channels=3,
        num_classes=2,
        image_size=32,
        pretrained=False,
        decoder_channels=[32, 16, 8, 8],
        seg_head_activation="softmax",
        annotator_activation="sigmoid",
    )
    t_cfg = TrainConfig(
        model_name="CT-GCE",
        epochs=2,
        warmup_epochs=1,
        calibration_interval=1,
        epochs_phases=[],
        device="cpu",
        num_classes=2,
        num_annotators=3,
        probabilistic_thresholds=[0.5],
    )

    # CT-GCE Engine
    ct_model = CTGCEModel(m_cfg)
    ct_crit = CTGCELoss(annotators=3, classes=2, ignored_value=0.6, q=0.7)
    ct_trainer = CTGCETrainer(ct_model, loader, loader, ct_crit, torch.device("cpu"), t_cfg)
    _, ct_hist = ct_trainer.fit()
    assert len(ct_hist["train_loss"]) == 2
    assert len(ct_hist["branch_loss"]) == 2
    assert ct_trainer.cutoffs is not None

    # AnnotHarmony Engine
    ah_model = AnnotHarmony(m_cfg)
    ah_crit = TGCE_SSPS(annotators=3, classes=2, ignored_value=0.6)
    ah_trainer = Trainer(ah_model, loader, loader, ah_crit, torch.device("cpu"), t_cfg)
    _, ah_hist = ah_trainer.fit()
    assert len(ah_hist["val_dice"]) == 2


def test_experiment_config_roundtrip(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        data=DataConfig(data_dir=str(tmp_path)),
        model=ModelConfig(pretrained=False),
        training=TrainConfig(model_name="CT-GCE"),
        experiment=ExperimentMetadata(name="smoke_ctgce"),
    )
    assert cfg.training.model_name == "CT-GCE"
