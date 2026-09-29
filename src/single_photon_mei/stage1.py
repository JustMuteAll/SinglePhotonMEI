from __future__ import annotations

import importlib.metadata
import json
import time
from functools import cmp_to_key
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .utils.encoding import nested_ridge_cv, zscore_columns
from .utils.io import (
    atomic_csv,
    atomic_json,
    atomic_npy,
    atomic_npz,
    project_paths,
    read_responses,
    sha256_file,
    validate_neural_data,
)
from .utils.models import adapt_activation, load_backbone


class StimulusDataset(Dataset):
    def __init__(self, paths, mean, std, image_size):
        self.paths = list(paths)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            values = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(values).permute(2, 0, 1).contiguous()
        return (tensor - self.mean) / self.std


def _output(config: dict) -> Path:
    return project_paths(config)["output"] / "stage1"


def _safe_layer(layer: str) -> str:
    return layer.replace(".", "_")


def feature_path(config: dict, model_key: str, layer: str) -> Path:
    return _output(config) / "features" / model_key / f"{_safe_layer(layer)}.npy"


def result_paths(config: dict, model_key: str, layer: str) -> tuple[Path, Path, Path]:
    root = _output(config) / "encoding" / model_key
    safe = _safe_layer(layer)
    return root / f"{safe}_scores.npz", root / f"{safe}_oof.npy", root / f"{safe}_result.json"


def validate_stage1_config(config: dict) -> None:
    if config.get("schema_version") != "single-photon-mei-stage1-1":
        raise ValueError("Unsupported Stage-1 config schema")
    feature = config["features"]
    encoding = config["encoding"]
    if feature.get("resize") != "bilinear" or not feature.get("l2_normalize", False):
        raise ValueError("v1 requires bilinear resize and L2-normalized features")
    if encoding.get("method") != "ridge" or encoding.get("pca_mode") != "leakage_free":
        raise ValueError("v1 supports only leakage-free Ridge")
    if encoding.get("alpha_mode") != "shared_across_targets":
        raise ValueError("Stage 1 requires shared alpha selection")
    if encoding.get("metric") != "pearson" or encoding.get("response_normalization") != "none":
        raise ValueError("Stage 1 requires raw-response Pearson evaluation")
    if not encoding.get("shuffle", False) or not encoding.get("fit_intercept", False):
        raise ValueError("v1 requires shuffled folds and a fitted Ridge intercept")
    if int(encoding["outer_folds"]) < 2 or int(encoding["inner_folds"]) < 2:
        raise ValueError("outer_folds and inner_folds must be at least two")
    if not encoding.get("alphas") or any(float(value) < 0 for value in encoding["alphas"]):
        raise ValueError("encoding.alphas must contain non-negative values")
    if not config.get("models"):
        raise ValueError("At least one model is required")
    for key, spec in config["models"].items():
        if not spec.get("layers") or len(set(spec["layers"])) != len(spec["layers"]):
            raise ValueError(f"{key} must define unique candidate layers")
        if len(spec["preprocess"]["mean"]) != 3 or len(spec["preprocess"]["std"]) != 3:
            raise ValueError(f"{key} preprocessing must contain three-channel mean/std")
        if len(spec.get("sha256", "")) != 64:
            raise ValueError(f"{key} must define a SHA-256 checkpoint hash")
        if spec["family"] == "dinov2_vit" and int(spec.get("image_size", -1)) != int(feature["image_size"]):
            raise ValueError("DINOv2 model and feature image sizes must match")


def _extract_features(
    config: dict,
    manifest: pd.DataFrame,
    model_key: str,
    overwrite: bool,
    layers: list[str] | None = None,
) -> list[dict]:
    spec = config["models"][model_key]
    paths = project_paths(config)
    runtime = config["runtime"]
    feature = config["features"]
    device = runtime["device"]
    extractor, weight_metadata = load_backbone(
        spec, paths["weights"], device, amp=bool(feature["amp"])
    )
    dataset = StimulusDataset(
        manifest["image_path"], spec["preprocess"]["mean"], spec["preprocess"]["std"], feature["image_size"]
    )
    loader = DataLoader(
        dataset,
        batch_size=int(runtime["feature_batch_size"]),
        shuffle=False,
        num_workers=int(runtime.get("num_workers", 0)),
        pin_memory=device.startswith("cuda"),
    )
    layers = layers or list(spec["layers"])
    outputs = {layer: [] for layer in layers}
    shapes = {}
    prefix_tokens = int(getattr(extractor.model, "num_prefix_tokens", 1))
    for images in loader:
        with torch.no_grad():
            activations = extractor.extract_features(images, layers)
        for layer in layers:
            activation = activations[layer]
            shapes.setdefault(layer, list(activation.shape[1:]))
            values = adapt_activation(
                activation,
                family=spec["family"],
                pool_size=int(feature["spatial_pool_size"]),
                l2_normalize=bool(feature["l2_normalize"]),
                num_prefix_tokens=prefix_tokens,
            )
            outputs[layer].append(values.cpu().numpy().astype(np.float32))
    records = []
    for layer, blocks in outputs.items():
        values = np.concatenate(blocks)
        path = feature_path(config, model_key, layer)
        atomic_npy(path, values, overwrite)
        metadata = {
            "model_key": model_key,
            "layer": layer,
            "activation_shape": shapes[layer],
            "feature_shape": list(values.shape),
            "preprocess": spec["preprocess"],
            "image_size": int(feature["image_size"]),
            "spatial_pool_size": int(feature["spatial_pool_size"]),
            "l2_normalize": bool(feature["l2_normalize"]),
            **weight_metadata,
        }
        atomic_json(path.with_suffix(".json"), metadata, overwrite)
        records.append(metadata)
    return records


def _score_summary(scores: np.ndarray) -> dict:
    finite = scores[np.isfinite(scores)]
    if not finite.size:
        raise ValueError("No finite encoding scores")
    top_count = min(50, len(finite))
    return {
        "median_score": float(np.median(finite)),
        "mean_score": float(np.mean(finite)),
        "positive_fraction": float(np.mean(finite > 0)),
        "top50_mean_score": float(np.mean(np.sort(finite)[-top_count:])),
        "finite_unit_count": int(len(finite)),
    }


def run_stage1(config: dict, model_keys: list[str] | None = None, overwrite: bool = False) -> dict:
    validate_stage1_config(config)
    root = _output(config)
    manifest, qc = validate_neural_data(config)
    resolved = {key: value for key, value in config.items() if key != "_config_path"}
    selected_models = model_keys or list(config["models"])
    unknown = [key for key in selected_models if key not in config["models"]]
    if unknown:
        raise KeyError(f"Unknown Stage-1 models: {unknown}")
    resolved_path = root / "resolved_config.json"
    if resolved_path.is_file() and not overwrite:
        existing = json.loads(resolved_path.read_text(encoding="utf-8"))
        if existing != resolved:
            raise ValueError("Stage-1 output exists for a different resolved config")
    else:
        atomic_json(resolved_path, resolved, overwrite)
    scope = {
        "selected_models": selected_models,
        "resolved_config_sha256": sha256_file(resolved_path),
        "device": config["runtime"]["device"],
        "feature_batch_size": int(config["runtime"]["feature_batch_size"]),
    }
    scope_name = "__".join(selected_models)
    scope_path = root / "execution_scopes" / f"{scope_name}.json"
    if scope_path.is_file() and not overwrite:
        if json.loads(scope_path.read_text(encoding="utf-8")) != scope:
            raise ValueError("Existing execution scope has different settings")
    else:
        atomic_json(scope_path, scope, overwrite)
    for path, writer, value in (
        (root / "qc" / "data_manifest.csv", atomic_csv, manifest),
        (root / "qc" / "data_qc.json", atomic_json, qc),
    ):
        if overwrite or not path.exists():
            writer(path, value, overwrite)
    responses = read_responses(config)
    normalized, _, std = zscore_columns(responses)
    population = normalized.mean(axis=1)
    ranking = manifest.copy()
    ranking["population_z_mean"] = population
    ranking = ranking.sort_values("population_z_mean", ascending=False)
    tuning_ranking = root / "tuning" / "population_ranking.csv"
    if overwrite or not tuning_ranking.exists():
        atomic_csv(tuning_ranking, ranking, overwrite)
    tuning_summary = root / "tuning" / "summary.json"
    if overwrite or not tuning_summary.exists():
        atomic_json(tuning_summary, {
        "valid_units": int(np.sum(std > np.finfo(np.float32).eps)),
        "top_image_ids": ranking.head(50)["image_id"].astype(int).tolist(),
        "bottom_image_ids": ranking.tail(50)["image_id"].astype(int).tolist(),
        }, overwrite)

    records = []
    encoding = config["encoding"]
    for model_key in selected_models:
        missing = [layer for layer in config["models"][model_key]["layers"] if not feature_path(config, model_key, layer).is_file()]
        if missing:
            _extract_features(config, manifest, model_key, overwrite, missing)
        for layer in config["models"][model_key]["layers"]:
            score_path, prediction_path, result_path = result_paths(config, model_key, layer)
            if result_path.is_file() and not overwrite:
                records.append(json.loads(result_path.read_text(encoding="utf-8")))
                continue
            features = np.load(feature_path(config, model_key, layer), mmap_mode="r")
            start = time.perf_counter()
            result = nested_ridge_cv(
                features,
                responses,
                alphas=[float(value) for value in encoding["alphas"]],
                outer_folds=int(encoding["outer_folds"]),
                inner_folds=int(encoding["inner_folds"]),
                pca_components=int(encoding["pca_components"]),
                random_state=int(encoding["random_state"]),
            )
            result.scores[std <= np.finfo(np.float32).eps] = np.nan
            fold_assignment = np.full(len(responses), -1, dtype=np.int16)
            for fold, (_, test) in enumerate(result.fold_indices):
                fold_assignment[test] = fold
            record = {
                "model_key": model_key,
                "family": config["models"][model_key]["family"],
                "architecture": config["models"][model_key]["architecture"],
                "layer": layer,
                **_score_summary(result.scores),
                "elapsed_seconds": time.perf_counter() - start,
            }
            atomic_npz(score_path, overwrite, scores=result.scores, selected_alphas=result.selected_alphas, fold_assignment=fold_assignment)
            atomic_npy(prediction_path, result.predictions.astype(np.float32), overwrite)
            atomic_json(result_path, record, overwrite)
            records.append(record)
    table = pd.DataFrame(records)
    atomic_csv(root / "partial_model_layer_results.csv", table, overwrite=True)
    return {"status": "complete", "models": selected_models, "result_count": len(records)}


def _compare(left: dict, right: dict, tolerance: float) -> int:
    delta = float(left["median_score"]) - float(right["median_score"])
    if abs(delta) >= tolerance:
        return -1 if delta > 0 else 1
    for key in ("mean_score", "positive_fraction", "top50_mean_score"):
        delta = float(left[key]) - float(right[key])
        if delta:
            return -1 if delta > 0 else 1
    left_name = f"{left['model_key']}::{left['layer']}"
    right_name = f"{right['model_key']}::{right['layer']}"
    return (left_name > right_name) - (left_name < right_name)


def finalize_stage1(config: dict, overwrite: bool = False) -> dict:
    validate_stage1_config(config)
    records = []
    for model_key, spec in config["models"].items():
        for layer in spec["layers"]:
            path = result_paths(config, model_key, layer)[2]
            if not path.is_file():
                raise FileNotFoundError(f"Missing Stage-1 result: {path}")
            records.append(json.loads(path.read_text(encoding="utf-8")))
    tolerance = float(config["ranking"]["median_tie_tolerance"])
    ranked = sorted(records, key=cmp_to_key(lambda a, b: _compare(a, b, tolerance)))
    table = pd.DataFrame(ranked)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    root = _output(config)
    atomic_csv(root / "model_layer_ranking.csv", table, overwrite)
    winner = ranked[0]
    score_path, prediction_path, _ = result_paths(config, winner["model_key"], winner["layer"])
    result = {
        "status": "complete",
        "winner_id": f"{winner['model_key']}::{winner['layer']}",
        "winner": winner,
        "scores_sha256": sha256_file(score_path),
        "oof_predictions_sha256": sha256_file(prediction_path),
        "feature_sha256": sha256_file(feature_path(config, winner["model_key"], winner["layer"])),
        "resolved_config_sha256": sha256_file(root / "resolved_config.json"),
        "software": {name: importlib.metadata.version(name) for name in (
            "numpy", "scipy", "pandas", "h5py", "scikit-learn", "torch", "torchvision", "timm"
        )},
    }
    result["stage1_status"] = "awaiting_stage2"
    atomic_json(root / "winner.json", result, overwrite)
    return result
