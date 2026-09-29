from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from .stage1 import StimulusDataset, feature_path, result_paths
from .utils.diffusion import CLIP_MEAN, CLIP_STD, NeuralGuidedDiffusionPipeline
from .utils.encoding import (
    bh_fdr,
    fit_per_target_readout,
    permutation_pvalues,
    tuning_correlation,
    zscore_columns,
)
from .utils.io import (
    atomic_csv,
    atomic_json,
    atomic_npz,
    load_config,
    project_paths,
    read_responses,
    read_unit_coordinates,
    resolve_path,
    sha256_file,
    validate_neural_data,
    verify_weight_tree,
)
from .utils.models import adapt_activation, load_backbone


def validate_stage2_config(config: dict) -> None:
    if config.get("schema_version") != "single-photon-mei-stage2-1":
        raise ValueError("Unsupported Stage-2 config schema")
    selection = config["selection"]
    if selection["selection_strategy"] not in {"top_encoding", "greedy_tuning_pearson"}:
        raise ValueError("Unknown selection strategy")
    if int(selection["n_targets"]) < 1 or int(selection["permutation_count"]) < 1:
        raise ValueError("n_targets and permutation_count must be positive")
    if selection.get("fdr_method") != "bh" or selection.get("tuning_metric") != "signed_pearson":
        raise ValueError("v1 requires BH-FDR and signed Pearson tuning")
    if not selection.get("permutation_two_sided", False):
        raise ValueError("v1 requires two-sided permutation tests")
    if selection["selection_strategy"] == "greedy_tuning_pearson":
        if not selection.get("pairwise_threshold_strict", False):
            raise ValueError("v1 diversity selection requires a strict pairwise threshold")
        if not -1 < float(selection["maximum_pairwise_tuning_r"]) <= 1:
            raise ValueError("maximum_pairwise_tuning_r must be in (-1, 1]")
    readout = config["readout"]
    if readout.get("alpha_mode") != "per_target" or readout.get("pca_mode") != "leakage_free":
        raise ValueError("Stage 2 requires per-target leakage-free Ridge")
    if readout.get("response_normalization") != "zscore_per_target" or not readout.get("fit_intercept", False):
        raise ValueError("Stage 2 requires target z-scoring and a fitted intercept")
    generation = config["generation"]
    required = {
        "prompt": "",
        "classifier_free_guidance_scale": 0,
        "sag_scale": 0,
        "scheduler": "DPMSolverMultistepScheduler",
        "objective": "maximize_target_z",
        "batch_size": 1,
    }
    for key, expected in required.items():
        if generation.get(key) != expected:
            raise ValueError(f"v1 requires generation.{key}={expected!r}")
    if generation.get("target_resize") != "bilinear" or int(generation["target_preprocess_size"]) != 224:
        raise ValueError("v1 requires 224-pixel bilinear target preprocessing")
    if generation.get("dtype") not in {"float16", "float32"}:
        raise ValueError("generation.dtype must be float16 or float32")
    if int(generation["image_size"]) % 8:
        raise ValueError("generation.image_size must be divisible by eight")
    if len(set(generation["seeds"])) != len(generation["seeds"]):
        raise ValueError("generation.seeds must be unique")
    regime_names = [regime["name"] for regime in generation["regimes"]]
    if len(set(regime_names)) != len(regime_names):
        raise ValueError("generation regime names must be unique")
    expected_count = len(generation["regimes"]) * len(generation["seeds"])
    if int(generation["expected_images_per_target"]) != expected_count:
        raise ValueError("expected_images_per_target does not match regimes × seeds")


def _stage1_config(config: dict) -> dict:
    return load_config(resolve_path(config["_config_path"], config["stage1_config"]))


def _run_root(config: dict) -> Path:
    return project_paths(config)["output"] / "stage2_runs" / config["run_name"]


def _winner(config: dict) -> tuple[dict, str, str]:
    stage1 = _stage1_config(config)
    path = project_paths(stage1)["output"] / "stage1" / "winner.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    status = json.loads(path.read_text(encoding="utf-8"))
    model_key, layer = status["winner_id"].split("::", 1)
    return stage1, model_key, layer


def greedy_select_diverse_targets(responses, eligible, scores, *, threshold, target_count):
    normalized, _, std = zscore_columns(responses)
    if np.any(std[eligible] <= np.finfo(np.float32).eps):
        raise ValueError("Eligible units include zero-variance responses")
    order = np.lexsort((eligible, -scores[eligible]))
    candidates = eligible[order]
    selected, audit = [], []
    for encoding_rank, unit in enumerate(candidates, 1):
        if selected:
            correlations = normalized[:, selected].T @ normalized[:, unit] / normalized.shape[0]
            nearest = int(np.argmax(correlations))
            maximum = float(correlations[nearest])
            accepted = maximum < threshold
            nearest_unit = int(selected[nearest])
            nearest_rank = nearest + 1
        else:
            maximum, nearest_unit, nearest_rank, accepted = np.nan, -1, -1, True
        audit.append({
            "encoding_rank_among_eligible": encoding_rank,
            "unit_id_zero_based": int(unit),
            "oof_pearson_r": float(scores[unit]),
            "accepted": accepted,
            "selection_rank": len(selected) + 1 if accepted else np.nan,
            "max_tuning_r_to_previously_selected": maximum,
            "most_similar_selected_unit": nearest_unit,
            "most_similar_selected_rank": nearest_rank,
        })
        if accepted:
            selected.append(int(unit))
            if len(selected) == target_count:
                break
    return np.asarray(selected, dtype=np.int64), pd.DataFrame(audit)


def select_targets(config: dict, overwrite: bool = False) -> dict:
    validate_stage2_config(config)
    stage1, model_key, layer = _winner(config)
    score_path, prediction_path, _ = result_paths(stage1, model_key, layer)
    scores = np.load(score_path)["scores"]
    predictions = np.load(prediction_path, mmap_mode="r")
    responses = read_responses(config)
    selection = config["selection"]
    p_values = permutation_pvalues(
        responses,
        predictions,
        n_permutations=int(selection["permutation_count"]),
        random_state=int(selection["random_state"]),
        device=config["runtime"]["device"],
    )
    q_values = bh_fdr(p_values)
    minimum = float(selection["minimum_oof_r"])
    score_pass = scores > minimum if selection["minimum_oof_r_strict"] else scores >= minimum
    eligible_mask = np.isfinite(scores) & score_pass & (q_values < float(selection["maximum_fdr_q"]))
    eligible = np.flatnonzero(eligible_mask)
    target_count = int(selection["n_targets"])
    if len(eligible) < target_count:
        raise RuntimeError(f"Only {len(eligible)} units pass eligibility for {target_count} targets")
    strategy = selection["selection_strategy"]
    if strategy == "top_encoding":
        order = np.lexsort((eligible, -scores[eligible]))
        selected = eligible[order][:target_count]
        audit = pd.DataFrame({
            "encoding_rank_among_eligible": np.arange(1, target_count + 1),
            "unit_id_zero_based": selected,
            "oof_pearson_r": scores[selected],
            "accepted": True,
            "selection_rank": np.arange(1, target_count + 1),
            "max_tuning_r_to_previously_selected": np.nan,
            "most_similar_selected_unit": -1,
            "most_similar_selected_rank": -1,
        })
    else:
        selected, audit = greedy_select_diverse_targets(
            responses,
            eligible,
            scores,
            threshold=float(selection["maximum_pairwise_tuning_r"]),
            target_count=target_count,
        )
        if len(selected) != target_count:
            atomic_csv(_run_root(config) / "selection_audit.csv", audit, overwrite)
            raise RuntimeError(f"Only {len(selected)} diverse targets could be selected")
    coordinates = read_unit_coordinates(config)
    audit_units = audit["unit_id_zero_based"].to_numpy(int)
    audit["permutation_p"] = p_values[audit_units]
    audit["fdr_q"] = q_values[audit_units]
    audit["row_zero_based"] = coordinates[audit_units, 0]
    audit["column_zero_based"] = coordinates[audit_units, 1]
    root = _run_root(config)
    resolved = {key: value for key, value in config.items() if key != "_config_path"}
    atomic_json(root / "resolved_config.json", resolved, overwrite)
    atomic_csv(root / "selection_audit.csv", audit, overwrite)
    accepted = audit[audit["accepted"]].sort_values("selection_rank")
    table = pd.DataFrame({
        "rank": np.arange(1, target_count + 1),
        "selection_rank": np.arange(1, target_count + 1),
        "encoding_rank_among_eligible": accepted["encoding_rank_among_eligible"].to_numpy(int),
        "unit_id_zero_based": selected,
        "row_zero_based": coordinates[selected, 0],
        "column_zero_based": coordinates[selected, 1],
        "oof_pearson_r": scores[selected],
        "permutation_p": p_values[selected],
        "fdr_q": q_values[selected],
    })
    atomic_csv(root / "selected_targets.csv", table, overwrite)
    selected_raw = responses[:, selected]
    selected_z, response_mean, response_std = zscore_columns(selected_raw)
    correlation = tuning_correlation(selected_raw)
    manifest, _ = validate_neural_data(config)
    atomic_npz(
        root / "selected_target_responses.npz",
        overwrite,
        raw_responses=selected_raw,
        z_responses=selected_z,
        image_ids=manifest["image_id"].to_numpy(int),
        target_units=selected,
        target_coordinates=coordinates[selected],
    )
    atomic_npz(root / "tuning_rdm.npz", overwrite, target_units=selected, correlation=correlation, rdm=1 - correlation)
    readout = config["readout"]
    features = np.load(feature_path(stage1, model_key, layer), mmap_mode="r")
    fitted = fit_per_target_readout(
        features,
        selected_z,
        alphas=[float(value) for value in readout["alphas"]],
        cv_folds=int(readout["cv_folds"]),
        pca_components=int(readout["pca_components"]),
        random_state=int(readout["random_state"]),
    )
    atomic_npz(
        root / "primary_readout.npz",
        overwrite,
        target_units=selected,
        weight=fitted.weight.astype(np.float32),
        bias=fitted.bias.astype(np.float32),
        response_mean=response_mean,
        response_std=response_std,
        selected_alphas=fitted.selected_alphas,
        cv_scores=fitted.cv_scores,
        natural_predictions=fitted.predictions.astype(np.float32),
    )
    pair = correlation[np.triu_indices(target_count, 1)]
    result = {
        "status": "complete",
        "winner_id": f"{model_key}::{layer}",
        "selection_strategy": strategy,
        "eligible_unit_count": int(len(eligible)),
        "selected_target_count": target_count,
        "candidates_scanned": int(len(audit)),
        "maximum_pairwise_tuning_r": float(pair.max()) if len(pair) else None,
        "median_pairwise_tuning_r": float(np.median(pair)) if len(pair) else None,
        "selected_targets_sha256": sha256_file(root / "selected_targets.csv"),
        "readout_sha256": sha256_file(root / "primary_readout.npz"),
        "resolved_config_sha256": sha256_file(root / "resolved_config.json"),
    }
    atomic_json(root / "target_selection.json", result, overwrite)
    return result


def _model_input(image: Image.Image, mean, std, size: int) -> torch.Tensor:
    array = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def _make_objective(extractor, spec, layer, weight, bias, feature_config, device):
    weight_tensor = torch.as_tensor(weight, dtype=torch.float32, device=device)
    bias_tensor = torch.as_tensor(bias, dtype=torch.float32, device=device)
    clip_mean = torch.tensor(CLIP_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor(CLIP_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    mean = torch.tensor(spec["preprocess"]["mean"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(spec["preprocess"]["std"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    prefix = int(getattr(extractor.model, "num_prefix_tokens", 1))

    def objective(clip_normalized_image):
        rgb = clip_normalized_image.float() * clip_std + clip_mean
        activation = extractor.extract_features((rgb - mean) / std, layer)
        features = adapt_activation(
            activation,
            family=spec["family"],
            pool_size=int(feature_config["spatial_pool_size"]),
            l2_normalize=bool(feature_config["l2_normalize"]),
            num_prefix_tokens=prefix,
        )
        prediction = features @ weight_tensor + bias_tensor
        if prediction.numel() != 1:
            raise ValueError("Generation supports one target and one image at a time")
        return -prediction.reshape(())

    return objective


def _predict_image(image, extractor, spec, layer, weight, bias, feature_config, device):
    tensor = _model_input(image, spec["preprocess"]["mean"], spec["preprocess"]["std"], int(feature_config["image_size"]))
    prefix = int(getattr(extractor.model, "num_prefix_tokens", 1))
    with torch.no_grad():
        activation = extractor.extract_features(tensor.to(device), layer)
        features = adapt_activation(
            activation,
            family=spec["family"],
            pool_size=int(feature_config["spatial_pool_size"]),
            l2_normalize=bool(feature_config["l2_normalize"]),
            num_prefix_tokens=prefix,
        )
    return float((features.float() @ torch.as_tensor(weight, device=device).float() + float(bias)).item())


def _clipping_fraction(image: Image.Image) -> float:
    values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return float(np.mean((values <= 1) | (values >= 254)))


def generate_mei(config: dict) -> dict:
    validate_stage2_config(config)
    stage1, model_key, layer = _winner(config)
    paths = project_paths(config)
    root = _run_root(config)
    targets_path = root / "selected_targets.csv"
    readout_path = root / "primary_readout.npz"
    selection_status_path = root / "target_selection.json"
    resolved_path = root / "resolved_config.json"
    if not all(path.is_file() for path in (targets_path, readout_path, selection_status_path, resolved_path)):
        raise FileNotFoundError("Run stage2 select first")
    resolved = {key: value for key, value in config.items() if key != "_config_path"}
    if json.loads(resolved_path.read_text(encoding="utf-8")) != resolved:
        raise ValueError("Stage-2 config differs from the config used for target selection")
    selection_status = json.loads(selection_status_path.read_text(encoding="utf-8"))
    if selection_status["selected_targets_sha256"] != sha256_file(targets_path):
        raise ValueError("Selected-target hash differs from target-selection provenance")
    if selection_status["readout_sha256"] != sha256_file(readout_path):
        raise ValueError("Readout hash differs from target-selection provenance")
    targets = pd.read_csv(targets_path)
    readout = np.load(readout_path)
    target_units = np.asarray(readout["target_units"], dtype=int)
    if not np.array_equal(targets["unit_id_zero_based"].to_numpy(int), target_units):
        raise ValueError("Selected-target and readout order differ")
    readout_hash = sha256_file(readout_path)
    target_hash = sha256_file(targets_path)
    config_hash = sha256_file(resolved_path)
    spec = stage1["models"][model_key]
    device = config["runtime"]["device"]
    extractor, _ = load_backbone(spec, paths["weights"], device, amp=bool(stage1["features"]["amp"]))
    generation = config["generation"]
    dtype = {"float16": torch.float16, "float32": torch.float32}[generation["dtype"]]
    diffusion_path = paths["weights"] / generation["diffusion_checkpoint"]
    verify_weight_tree(paths["weights"], generation["diffusion_checkpoint"])
    pipe = NeuralGuidedDiffusionPipeline.from_local(diffusion_path, device=device, dtype=dtype)
    manifest_path = root / "mei" / "mei_manifest.csv"
    records = pd.read_csv(manifest_path).to_dict("records") if manifest_path.is_file() else []
    record_by_key = {
        (int(row["unit_id_zero_based"]), str(row["regime"]), int(row["seed"])): index
        for index, row in enumerate(records)
    }
    for row in targets.itertuples(index=False):
        unit = int(row.unit_id_zero_based)
        position = int(np.flatnonzero(target_units == unit)[0])
        pipe.objective = _make_objective(
            extractor,
            spec,
            layer,
            readout["weight"][position],
            float(readout["bias"][position]),
            stage1["features"],
            device,
        )
        for regime in generation["regimes"]:
            name = str(regime["name"])
            scale = float(regime["neural_guidance_scale"])
            for seed in generation["seeds"]:
                key = (unit, name, int(seed))
                if key in record_by_key and records[record_by_key[key]].get("status") == "complete":
                    path = Path(records[record_by_key[key]]["image_path"])
                    if (
                        not path.is_file()
                        or records[record_by_key[key]]["readout_sha256"] != readout_hash
                        or records[record_by_key[key]]["selected_targets_sha256"] != target_hash
                        or records[record_by_key[key]]["resolved_config_sha256"] != config_hash
                    ):
                        raise ValueError("Existing MEI manifest does not match files/readout")
                    continue
                image_path = root / "mei" / "images" / f"unit_{unit:05d}_{name}_scale_{scale:g}_seed_{int(seed)}.png"
                if image_path.exists() and key not in record_by_key:
                    raise FileExistsError(f"Orphan image: {image_path}")
                base = {
                    "status": "pending",
                    "unit_id_zero_based": unit,
                    "target_rank": int(row.rank),
                    "encoding_rank_among_eligible": int(row.encoding_rank_among_eligible),
                    "row_zero_based": int(row.row_zero_based),
                    "column_zero_based": int(row.column_zero_based),
                    "model_key": model_key,
                    "layer": layer,
                    "regime": name,
                    "neural_guidance_scale": scale,
                    "seed": int(seed),
                    "steps": int(generation["steps"]),
                    "height": int(generation["image_size"]),
                    "width": int(generation["image_size"]),
                    "image_path": str(image_path.resolve()),
                    "readout_sha256": readout_hash,
                    "selected_targets_sha256": target_hash,
                    "resolved_config_sha256": config_hash,
                }
                if key not in record_by_key:
                    record_by_key[key] = len(records)
                    records.append(base)
                    atomic_csv(manifest_path, pd.DataFrame(records), overwrite=True)
                if image_path.is_file():
                    with Image.open(image_path) as existing:
                        if existing.size != (int(generation["image_size"]), int(generation["image_size"])):
                            raise ValueError(f"Incomplete MEI has the wrong dimensions: {image_path}")
                        image = existing.convert("RGB").copy()
                else:
                    image = pipe.generate(
                        seed=int(seed),
                        neural_guidance_scale=scale,
                        steps=int(generation["steps"]),
                        height=int(generation["image_size"]),
                        width=int(generation["image_size"]),
                        device=device,
                    )
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    partial = image_path.with_name(image_path.name + ".partial")
                    image.save(partial, format="PNG", compress_level=6)
                    partial.replace(image_path)
                prediction = _predict_image(
                    image, extractor, spec, layer, readout["weight"][position], readout["bias"][position], stage1["features"], device
                )
                records[record_by_key[key]] = {
                    **base,
                    "status": "complete",
                    "primary_prediction_z": prediction,
                    "clipping_fraction": _clipping_fraction(image),
                }
                atomic_csv(manifest_path, pd.DataFrame(records), overwrite=True)
    expected = len(targets) * int(generation["expected_images_per_target"])
    complete = pd.DataFrame(records)
    if len(complete) != expected or not (complete["status"] == "complete").all():
        raise RuntimeError("MEI generation is incomplete")
    summary = {
        "status": "complete",
        "image_count": int(len(complete)),
        "target_count": int(len(targets)),
        "manifest_sha256": sha256_file(manifest_path),
        "readout_sha256": readout_hash,
        "selected_targets_sha256": sha256_file(targets_path),
    }
    atomic_json(root / "mei" / "generation_summary.json", summary, overwrite=True)
    return summary


def _features_for_images(stage1: dict, config: dict, model_key: str, layer: str, image_paths, device: str) -> np.ndarray:
    spec = stage1["models"][model_key]
    extractor, _ = load_backbone(spec, project_paths(config)["weights"], device, amp=bool(stage1["features"]["amp"]))
    dataset = StimulusDataset(image_paths, spec["preprocess"]["mean"], spec["preprocess"]["std"], stage1["features"]["image_size"])
    loader = DataLoader(dataset, batch_size=int(config["runtime"]["evaluation_batch_size"]), shuffle=False)
    prefix = int(getattr(extractor.model, "num_prefix_tokens", 1))
    output = []
    for images in loader:
        with torch.no_grad():
            activation = extractor.extract_features(images, layer)
            output.append(adapt_activation(
                activation,
                family=spec["family"],
                pool_size=int(stage1["features"]["spatial_pool_size"]),
                l2_normalize=bool(stage1["features"]["l2_normalize"]),
                num_prefix_tokens=prefix,
            ).cpu().numpy())
    return np.concatenate(output)


def _percentiles(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return 100.0 * np.mean(reference[None, :, :] <= values[:, None, :], axis=1)


def peer_review(config: dict, overwrite: bool = False) -> dict:
    validate_stage2_config(config)
    stage1, winner_model, _ = _winner(config)
    root = _run_root(config)
    manifest_path = root / "mei" / "mei_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run stage2 generate first")
    mei = pd.read_csv(manifest_path)
    generation_status = json.loads((root / "mei" / "generation_summary.json").read_text(encoding="utf-8"))
    if generation_status["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("MEI manifest differs from generation provenance")
    if not (mei["status"] == "complete").all() or any(not Path(path).is_file() for path in mei["image_path"]):
        raise RuntimeError("Peer review requires complete MEI files")
    targets = pd.read_csv(root / "selected_targets.csv")
    units = targets["unit_id_zero_based"].to_numpy(int)
    responses = read_responses(config)
    selected_z, _, _ = zscore_columns(responses[:, units])
    ranking = pd.read_csv(project_paths(stage1)["output"] / "stage1" / "model_layer_ranking.csv")
    peers = []
    for model_key in stage1["models"]:
        if model_key == winner_model:
            continue
        record = ranking[ranking["model_key"] == model_key].sort_values("rank").iloc[0]
        peers.append((model_key, str(record.layer)))
    if len(peers) != 3:
        raise ValueError("Peer review requires exactly three non-winning backbones")
    readout_config = config["readout"]
    device = config["runtime"]["device"]
    models = []
    primary = np.load(root / "primary_readout.npz")
    if not (mei["readout_sha256"] == sha256_file(root / "primary_readout.npz")).all():
        raise ValueError("MEI rows do not match the primary readout")
    if not (mei["selected_targets_sha256"] == sha256_file(root / "selected_targets.csv")).all():
        raise ValueError("MEI rows do not match selected targets")
    winner_layer = json.loads((project_paths(stage1)["output"] / "stage1" / "winner.json").read_text())["winner_id"].split("::", 1)[1]
    models.append((winner_model, winner_layer, primary["weight"], primary["bias"], primary["natural_predictions"]))
    peer_dir = root / "peer_review"
    for model_key, layer in peers:
        natural_features = np.load(feature_path(stage1, model_key, layer), mmap_mode="r")
        fitted = fit_per_target_readout(
            natural_features,
            selected_z,
            alphas=[float(v) for v in readout_config["alphas"]],
            cv_folds=int(readout_config["cv_folds"]),
            pca_components=int(readout_config["pca_components"]),
            random_state=int(readout_config["random_state"]),
        )
        atomic_npz(peer_dir / f"{model_key}_readout.npz", overwrite, target_units=units, weight=fitted.weight, bias=fitted.bias, natural_predictions=fitted.predictions, selected_alphas=fitted.selected_alphas)
        models.append((model_key, layer, fitted.weight, fitted.bias, fitted.predictions))
    prediction_matrices = {}
    percentile_matrices = {}
    for model_key, layer, weight, bias, natural_predictions in models:
        features = _features_for_images(stage1, config, model_key, layer, mei["image_path"], device)
        predictions = features @ np.asarray(weight).T + np.asarray(bias)
        percentiles = _percentiles(predictions, np.asarray(natural_predictions))
        prediction_matrices[model_key] = predictions
        percentile_matrices[model_key] = percentiles
        atomic_npz(peer_dir / f"{model_key}_prediction_matrices.npz", overwrite, target_units=units, natural_predictions=natural_predictions, mei_predictions=predictions, mei_percentiles=percentiles)

    review = config["peer_review"]
    dino_model = review["naturalness"]["model_key"]
    dino_layer = review["naturalness"]["layer"]
    natural_features = np.asarray(np.load(feature_path(stage1, dino_model, dino_layer), mmap_mode="r"))
    generated_features = _features_for_images(stage1, config, dino_model, dino_layer, mei["image_path"], device)
    components = min(int(review["naturalness"]["pca_components"]), len(natural_features) - 1, natural_features.shape[1])
    pca = PCA(n_components=components, random_state=int(readout_config["random_state"]))
    natural_reduced = pca.fit_transform(natural_features)
    generated_reduced = pca.transform(generated_features)
    neighbor_count = int(review["naturalness"]["neighbors"])
    neighbors = NearestNeighbors(
        n_neighbors=min(neighbor_count + 1, len(natural_reduced)), metric="euclidean"
    ).fit(natural_reduced)
    natural_distance = neighbors.kneighbors(natural_reduced, return_distance=True)[0][:, 1:].mean(axis=1)
    generated_distance = neighbors.kneighbors(
        generated_reduced,
        n_neighbors=min(neighbor_count, len(natural_reduced)),
        return_distance=True,
    )[0].mean(axis=1)
    distance_percentile = 100 * np.mean(natural_distance[None, :] <= generated_distance[:, None], axis=1)

    rows = []
    peer_keys = [key for key, _ in peers]
    for index, image_row in enumerate(mei.itertuples(index=False)):
        target_position = int(np.flatnonzero(units == int(image_row.unit_id_zero_based))[0])
        primary_percentile = float(percentile_matrices[winner_model][index, target_position])
        peer_values = [float(percentile_matrices[key][index, target_position]) for key in peer_keys]
        selectivity = sum(
            100 * np.mean(prediction_matrices[key][index] <= prediction_matrices[key][index, target_position]) >= float(review["peer_selectivity_percentile"])
            for key in peer_keys
        )
        regime = str(image_row.regime)
        natural_limit = float(review["naturalness"]["maximum_percentile_by_regime"][regime])
        clipping_limit = float(review["maximum_clipping_fraction_by_regime"][regime])
        passed = (
            primary_percentile >= float(review["primary_target_percentile"])
            and sum(value >= float(review["peer_target_percentile"]) for value in peer_values) >= int(review["minimum_peer_count"])
            and np.median(peer_values) >= float(review["peer_median_percentile"])
            and selectivity >= int(review["minimum_peer_selectivity_count"])
            and distance_percentile[index] <= natural_limit
            and float(image_row.clipping_fraction) <= clipping_limit
        )
        row = image_row._asdict()
        row.update({
            "naturalness_distance": float(generated_distance[index]),
            "naturalness_percentile": float(distance_percentile[index]),
            "primary_target_percentile": primary_percentile,
            "peer_target_percentile_median": float(np.median(peer_values)),
            "peer_count_target_percentile_pass": int(sum(value >= float(review["peer_target_percentile"]) for value in peer_values)),
            "peer_count_selectivity_pass": int(selectivity),
            "cross_model_pass": bool(passed),
        })
        for key, value in zip(peer_keys, peer_values):
            row[f"{key}_target_percentile"] = value
        rows.append(row)
    result = pd.DataFrame(rows)
    atomic_csv(peer_dir / "peer_predictions.csv", result, overwrite)
    atomic_csv(peer_dir / "consensus_mei.csv", result[result["cross_model_pass"]], overwrite)
    unit_summary = result.groupby(["unit_id_zero_based", "regime"])["cross_model_pass"].agg(["sum", "count"]).reset_index()
    unit_summary["unit_validated"] = unit_summary["sum"] >= int(review["minimum_passing_seeds_per_regime"])
    atomic_csv(peer_dir / "unit_consensus_summary.csv", unit_summary, overwrite)
    summary = {
        "status": "complete",
        "image_count": int(len(result)),
        "passing_images": int(result["cross_model_pass"].sum()),
        "validated_units": int(unit_summary.groupby("unit_id_zero_based")["unit_validated"].any().sum()),
    }
    atomic_json(peer_dir / "peer_review_summary.json", summary, overwrite)
    return summary


def final_report(config: dict, overwrite: bool = False) -> dict:
    root = _run_root(config)
    generation = json.loads((root / "mei" / "generation_summary.json").read_text(encoding="utf-8"))
    peer = json.loads((root / "peer_review" / "peer_review_summary.json").read_text(encoding="utf-8"))
    report = {
        "run_name": config["run_name"],
        "generated_images": generation["image_count"],
        "cross_model_passing_images": peer["passing_images"],
        "validated_units": peer["validated_units"],
        "sha256": {
            "selected_targets.csv": sha256_file(root / "selected_targets.csv"),
            "primary_readout.npz": sha256_file(root / "primary_readout.npz"),
            "mei_manifest.csv": sha256_file(root / "mei" / "mei_manifest.csv"),
            "peer_predictions.csv": sha256_file(root / "peer_review" / "peer_predictions.csv"),
        },
    }
    atomic_json(root / "report" / "final_report.json", report, overwrite)
    text = "# Single-Photon MEI Final Report\n\n" + "\n".join(
        f"- {key.replace('_', ' ').title()}: `{value}`" for key, value in report.items() if key != "sha256"
    ) + "\n\nCross-model agreement is computational QC; experimental validation requires new recordings.\n"
    from .utils.io import atomic_text
    atomic_text(root / "report" / "final_report.md", text, overwrite)
    return report
