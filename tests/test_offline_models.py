from pathlib import Path

import pytest
import torch

from single_photon_mei.stage2 import validate_stage2_config
from single_photon_mei.utils.models import adapt_activation, load_backbone


def test_cnn_activation_is_flattened_without_spatial_pooling():
    activation = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 3, 5, 7)
    features = adapt_activation(activation, family="resnet50", l2_normalize=False)
    assert features.shape == (2, 3 * 5 * 7)
    assert torch.equal(features, activation.flatten(1))


def test_vit_tokens_are_flattened_without_spatial_pooling():
    activation = torch.arange(2 * 21 * 3, dtype=torch.float32).reshape(2, 21, 3)
    features = adapt_activation(
        activation,
        family="dinov2_vit",
        l2_normalize=False,
        num_prefix_tokens=5,
    )
    expected = activation[:, 5:, :].transpose(1, 2).reshape(2, -1)
    assert features.shape == (2, 3 * 4 * 4)
    assert torch.equal(features, expected)


def test_missing_checkpoint_has_no_fallback(tmp_path):
    spec = {"family": "alexnet", "architecture": "alexnet", "checkpoint": "missing.pth", "preprocess": {"mean": [0, 0, 0], "std": [1, 1, 1]}}
    with pytest.raises(FileNotFoundError):
        load_backbone(spec, Path(tmp_path), "cpu", amp=False)


def test_checkpoint_hash_mismatch_is_fatal(tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"not a checkpoint")
    spec = {"family": "alexnet", "architecture": "alexnet", "checkpoint": "model.pth", "sha256": "0" * 64, "preprocess": {"mean": [0, 0, 0], "std": [1, 1, 1]}}
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_backbone(spec, Path(tmp_path), "cpu", amp=False)


def test_unvalidated_generation_branches_are_rejected():
    config = {
        "schema_version": "single-photon-mei-stage2-1",
        "selection": {"n_targets": 1, "permutation_count": 1, "selection_strategy": "top_encoding", "fdr_method": "bh", "tuning_metric": "signed_pearson", "permutation_two_sided": True},
        "readout": {"alpha_mode": "per_target", "pca_mode": "leakage_free", "response_normalization": "zscore_per_target", "fit_intercept": True},
        "generation": {"prompt": "cat", "classifier_free_guidance_scale": 0, "sag_scale": 0, "scheduler": "DPMSolverMultistepScheduler", "objective": "maximize_target_z", "batch_size": 1, "target_resize": "bilinear", "target_preprocess_size": 224, "dtype": "float16", "image_size": 512, "regimes": [{"name": "x"}], "seeds": [1], "expected_images_per_target": 1},
    }
    with pytest.raises(ValueError, match="prompt"):
        validate_stage2_config(config)


def test_source_has_no_legacy_imports():
    source_root = Path(__file__).resolve().parents[1] / "src"
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_root.rglob("*.py"))
    assert "NeuroPredictor" not in text
    assert "analysis_utils" not in text
