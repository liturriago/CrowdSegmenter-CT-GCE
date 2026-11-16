"""Tests for training/evaluation scripts, baselines, and ablation/hypothesis testing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from crowdsegmenter.losses.ct_gce import CTGCELoss
from crowdsegmenter.utils.ablation import (
    compare_per_sample_metrics,
    holm_bonferroni,
    save_ablation_report,
)
from scripts.evaluate_ablation import compare_saved_results, run_ct_gce_ablation_suite
from scripts.evaluate_baselines import run_baseline_evaluation
from scripts.train_annotharmony import run_annotharmony_experiment
from scripts.train_crowdseg import run_crowdseg_experiment
from scripts.train_ctgce import run_ctgce_experiment


def _build_synthetic_dataset(root: Path, n_per_split: int = 3) -> Path:
    rng = np.random.default_rng(7)
    for split in ("Train", "Valid", "Test"):
        (root / split / "patches").mkdir(parents=True, exist_ok=True)
        for folder in ("annotator_0", "annotator_1", "annotator_2", "ground_truth"):
            for cls in ("class_0", "class_1"):
                (root / split / "masks" / folder / cls).mkdir(parents=True, exist_ok=True)

        for idx in range(n_per_split):
            name = f"sample_{idx}.png"
            img = rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)
            Image.fromarray(img).save(root / split / "patches" / name)

            gt_c1 = np.zeros((24, 24), dtype=np.uint8)
            gt_c1[6:18, 6:18] = 255
            gt_c0 = 255 - gt_c1
            Image.fromarray(gt_c0).save(root / split / "masks" / "ground_truth" / "class_0" / name)
            Image.fromarray(gt_c1).save(root / split / "masks" / "ground_truth" / "class_1" / name)

            for r in range(3):
                c1 = gt_c1.copy()
                if r == 2:
                    c1[0:8, 0:8] = 255
                c0 = 255 - c1
                Image.fromarray(c0).save(
                    root / split / "masks" / f"annotator_{r}" / "class_0" / name
                )
                Image.fromarray(c1).save(
                    root / split / "masks" / f"annotator_{r}" / "class_1" / name
                )
    return root


def _write_yaml(path: Path, data_dir: Path, out_dir: Path, model_name: str) -> str:
    cfg = {
        "data": {
            "data_dir": str(data_dir),
            "partitions": ["Train", "Valid", "Test"],
            "images_folder": "patches",
            "masks_folder": "masks",
            "ground_truth_folder": "ground_truth",
            "load_annotators": True,
            "load_ground_truth": True,
            "num_classes": 2,
            "num_annotators": 3,
            "image_size": [32, 32],
            "ignored_value": 0.6,
            "batch_size": 3,
            "num_workers": 0,
            "pin_memory": False,
        },
        "model": {
            "num_annotators": 3,
            "in_channels": 3,
            "num_classes": 2,
            "image_size": 32,
            "pretrained": False,
            "decoder_channels": [32, 16, 8, 8],
            "use_residual": True,
            "seg_head_activation": "softmax",
            "annotator_activation": "sigmoid",
            "staple_beta": 1.0,
            "staple_max_iter": 5,
        },
        "training": {
            "model_name": model_name,
            "epochs": 2,
            "lr": 1e-3,
            "transfer_lr": 1e-3,
            "branch_lr": 1e-3,
            "threshold": 0.5,
            "probabilistic_thresholds": [0.5],
            "ground_truth": True,
            "num_classes": 2,
            "num_annotators": 3,
            "ignored_value": 0.6,
            "smooth": 1e-8,
            "epochs_phases": [],
            "gamma": 1.0,
            "ctgce_q": 0.7,
            "warmup_epochs": 1,
            "calibration_interval": 1,
            "tgce_q": 0.7,
            "alpha": 0.1,
            "seed": 42,
            "device": "cpu",
        },
        "experiment": {
            "name": f"test_{model_name.lower()}",
            "version": 1,
            "output_dir": str(out_dir),
            "save_results": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    return str(path)


def test_ctgce_loss_ablation_arms_s1_s2_s3() -> None:
    torch.manual_seed(0)
    seg_pred = torch.softmax(torch.randn(2, 2, 16, 16), dim=1)
    lam = torch.rand(2, 3, 2, 16, 16)
    labels = torch.randint(0, 2, (2, 3, 16, 16))
    masks = (
        torch.nn.functional.one_hot(labels, num_classes=2)
        .permute(0, 4, 1, 2, 3)
        .float()
        .reshape(2, 6, 16, 16)
    )
    cutoffs = torch.tensor([0.55, 0.60])

    losses = []
    for arm in ("S1", "S2", "S3"):
        crit = CTGCELoss(annotators=3, classes=2, q=0.7, ablation_arm=arm)
        val = crit.segmentation_loss(seg_pred, lam, masks, cutoffs)
        assert torch.isfinite(val)
        losses.append(float(val.item()))
    assert len(set(losses)) > 1


def test_holm_bonferroni_and_ablation_comparison(tmp_path: Path) -> None:
    adj = holm_bonferroni([0.01, 0.04, 0.03])
    assert pytest.approx(adj) == [0.03, 0.06, 0.06]

    arrays = {
        "CT-GCE": {
            "dice": np.array([0.85, 0.87, 0.86, 0.88, 0.84, 0.89, 0.86, 0.88]),
            "jaccard": np.array([0.75, 0.77, 0.76, 0.78, 0.74, 0.79, 0.76, 0.78]),
        },
        "MV": {
            "dice": np.array([0.70, 0.72, 0.71, 0.73, 0.69, 0.74, 0.71, 0.73]),
            "jaccard": np.array([0.55, 0.57, 0.56, 0.58, 0.54, 0.59, 0.56, 0.58]),
        },
        "S1": {
            "dice": np.array([0.80, 0.82, 0.81, 0.83, 0.79, 0.84, 0.81, 0.83]),
            "jaccard": np.array([0.68, 0.70, 0.69, 0.71, 0.67, 0.72, 0.69, 0.71]),
        },
    }
    rows = compare_per_sample_metrics(arrays, reference="CT-GCE", metrics=("dice", "jaccard"))
    assert len(rows) == 6
    mv_dice = [r for r in rows if r["method"] == "MV" and r["metric"] == "dice"][0]
    assert mv_dice["delta_vs_ref"] > 0.1
    assert mv_dice["wilcoxon_p_adj"] < 0.05
    assert mv_dice["significant"] is True

    j_path, c_path = save_ablation_report(rows, tmp_path)
    assert j_path.is_file() and c_path.is_file()


def test_train_eval_baselines_and_ablation_end_to_end(tmp_path: Path) -> None:
    data_dir = _build_synthetic_dataset(tmp_path / "data", n_per_split=3)

    # 1. Baselines: MV & Spatial-STAPLE
    mv_cfg = _write_yaml(tmp_path / "mv.yaml", data_dir, tmp_path / "out_mv", "MV")
    st_cfg = _write_yaml(tmp_path / "st.yaml", data_dir, tmp_path / "out_st", "Spatial-STAPLE")
    run_baseline_evaluation(mv_cfg, model_name="MV")
    run_baseline_evaluation(st_cfg, model_name="Spatial-STAPLE", beta=1.0)
    assert (tmp_path / "out_mv" / "test_probabilistic_per_sample.npz").is_file()
    assert (tmp_path / "out_mv" / "test_gt_per_sample.npz").is_file()
    assert (tmp_path / "out_st" / "baseline_spatial-staple_results.json").is_file()

    # 2. Deep models: AnnotHarmony, CrowdSeg, CT-GCE (train + test eval)
    ah_cfg = _write_yaml(tmp_path / "ah.yaml", data_dir, tmp_path / "out_ah", "Annot-Harmony")
    cs_cfg = _write_yaml(tmp_path / "cs.yaml", data_dir, tmp_path / "out_cs", "CrowdSeg")
    ct_cfg = _write_yaml(tmp_path / "ct.yaml", data_dir, tmp_path / "out_ct", "CT-GCE")

    run_annotharmony_experiment(ah_cfg)
    run_crowdseg_experiment(cs_cfg)
    ct_summary = run_ctgce_experiment(ct_cfg)

    assert (tmp_path / "out_ah" / "annot_harmony_final.pth").is_file()
    assert (tmp_path / "out_cs" / "crowdseg_final.pth").is_file()
    assert (tmp_path / "out_ct" / "ct_gce_final.pth").is_file()
    assert "rho_hat" in ct_summary and "cutoffs" in ct_summary
    assert "probabilistic" in ct_summary["test"] and "ground_truth" in ct_summary["test"]

    # 3. Cross-model comparison from saved .npz
    comp_rows = compare_saved_results(
        entries={
            "CT-GCE": tmp_path / "out_ct",
            "AnnotHarmony": tmp_path / "out_ah",
            "CrowdSeg": tmp_path / "out_cs",
            "Spatial-STAPLE": tmp_path / "out_st",
            "MV": tmp_path / "out_mv",
        },
        protocol="gt",
        reference="CT-GCE",
        output_dir=tmp_path / "cmp_out",
    )
    assert len(comp_rows) == 20  # 5 methods * 4 metrics
    assert (tmp_path / "cmp_out" / "ablation_gt.csv").is_file()

    # 4. CT-GCE component ablation suite
    ablation_payload = run_ct_gce_ablation_suite(ct_cfg, output_dir=tmp_path / "ablation_suite")
    assert set(ablation_payload["arms"].keys()) == {
        "plain_gce",
        "s1_hard_gate",
        "s2_gate_plus_lambda",
        "s3_ct_gce_no_warmup",
        "s3_ct_gce_full",
    }
    assert (tmp_path / "ablation_suite" / "ablation_suite_summary.json").is_file()
    assert (tmp_path / "ablation_suite" / "ablation_gt.csv").is_file()


def test_scripts_cli_help_entrypoints() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for script in (
        "scripts/train_ctgce.py",
        "scripts/train_annotharmony.py",
        "scripts/train_crowdseg.py",
        "scripts/evaluate_baselines.py",
        "scripts/evaluate_ablation.py",
        "scripts/optimize_ct_gce.py",
        "scripts/optimize_tgce_ssps.py",
        "scripts/optimize_noisy_label.py",
    ):
        proc = subprocess.run(
            [sys.executable, str(repo_root / script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"{script} --help failed: {proc.stderr}"
        assert "--data" in proc.stdout


def test_cli_data_and_output_dir_overrides(tmp_path: Path) -> None:
    data_dir = _build_synthetic_dataset(tmp_path / "kaggle_dl", n_per_split=2)
    # Point YAML at a non-existent Kaggle path so only --data override can succeed
    cfg_path = _write_yaml(
        tmp_path / "kaggle_cfg.yaml",
        Path("/kaggle/input/nonexistent-dataset"),
        tmp_path / "unused_out",
        "CT-GCE",
    )
    real_out = tmp_path / "drive_out"
    summary = run_ctgce_experiment(cfg_path, data_dir=str(data_dir), output_dir=str(real_out))
    assert (real_out / "ct_gce_final.pth").is_file()
    assert (real_out / "summary.json").is_file()
    assert summary["weights"].startswith(str(real_out))


def test_experiment_notebooks_are_valid_and_reference_real_files() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    nb_root = repo_root / "notebooks" / "experiments"
    nb_paths = sorted(nb_root.rglob("*.ipynb"))
    rel = {p.relative_to(nb_root).as_posix() for p in nb_paths}
    expected = {
        "baselines/oxfordiiitpet.ipynb",
        "baselines/riga.ipynb",
        "crowdseg/optimize_oxfordiiitpet_crowdseg.ipynb",
        "crowdseg/optimize_riga_crowdseg.ipynb",
        "crowdseg/train_oxfordiiitpet_crowdseg.ipynb",
        "crowdseg/train_riga_crowdseg.ipynb",
        "ctgce/optimize_oxfordiiitpet_ctgce.ipynb",
        "ctgce/optimize_riga_ctgce.ipynb",
        "ctgce/train_oxfordiiitpet_ctgce.ipynb",
        "ctgce/train_riga_ctgce.ipynb",
        "tgce_ssps/optimize_oxfordiiitpet_tgce_spss.ipynb",
        "tgce_ssps/optimize_riga_tgce_spss.ipynb",
        "tgce_ssps/train_oxfordiiitpet_tgce_spss.ipynb",
        "tgce_ssps/train_riga_tgce_spss.ipynb",
    }
    assert rel == expected
    for nb_path in nb_paths:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        joined = "\n".join("".join(c["source"]) for c in nb["cells"])
        assert "kagglehub.dataset_download" in joined
        assert "--data" in joined
        assert "drive.mount" in joined
        for token in joined.split():
            if token.startswith("scripts/") and token.endswith(".py"):
                assert (repo_root / token).is_file(), f"Missing {token} in {nb_path}"
            if token.startswith("configs/") and token.endswith(".yaml"):
                assert (repo_root / token).is_file(), f"Missing {token} in {nb_path}"
