from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


REQUIRED_DATASETS = (
    "responses",
    "image_ids",
    "unit_coords_zero_based",
    "pool_counts",
    "unit_pixel_offsets",
    "unit_pixel_indices",
    "map_mask",
)


def load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    return path.resolve()


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def verify_weight_tree(root: str | Path, relative_prefix: str | Path) -> list[dict]:
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    prefix = Path(relative_prefix).as_posix().rstrip("/") + "/"
    records = [record for record in manifest.get("files", []) if record["path"].startswith(prefix)]
    if not records:
        raise ValueError(f"Weight manifest contains no files under {relative_prefix}")
    for record in records:
        path = root / record["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Offline weight manifest mismatch: {path}")
    return records


def atomic_json(path: str | Path, value: dict, overwrite: bool = False) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False), overwrite)


def atomic_text(path: str | Path, text: str, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def atomic_csv(path: str | Path, table: pd.DataFrame, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    partial = path.with_name(path.name + ".partial")
    table.to_csv(partial, index=False)
    partial.replace(path)


def atomic_npy(path: str | Path, array: np.ndarray, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    partial = path.with_name(path.name + ".partial.npy")
    np.save(partial, array)
    partial.replace(path)


def atomic_npz(path: str | Path, overwrite: bool = False, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    partial = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(partial, **arrays)
    partial.replace(path)


def load_config(path: str | Path) -> dict:
    config = load_json(path)
    config["_config_path"] = str(Path(path).resolve())
    return config


def project_paths(config: dict) -> dict[str, Path]:
    config_path = config["_config_path"]
    paths = config["paths"]
    return {
        "neural_data": resolve_path(config_path, paths["neural_data"]),
        "stimuli": resolve_path(config_path, paths["stimulus_image_folder"]),
        "weights": resolve_path(config_path, paths["model_weights"]),
        "output": resolve_path(config_path, paths["output"]),
    }


def validate_neural_data(config: dict) -> tuple[pd.DataFrame, dict]:
    paths = project_paths(config)
    h5_path = paths["neural_data"]
    image_root = paths["stimuli"]
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    names = config.get("data", {}).get("datasets", {})
    dataset = lambda key: names.get(key, key)
    with h5py.File(h5_path, "r") as handle:
        missing = [dataset(key) for key in REQUIRED_DATASETS if dataset(key) not in handle]
        if missing:
            raise KeyError(f"Missing HDF5 datasets: {missing}")
        responses = handle[dataset("responses")]
        response_dtype = str(responses.dtype)
        image_ids = np.asarray(handle[dataset("image_ids")])
        coordinates = np.asarray(handle[dataset("unit_coords_zero_based")])
        counts = np.asarray(handle[dataset("pool_counts")])
        offsets = np.asarray(handle[dataset("unit_pixel_offsets")])
        members = np.asarray(handle[dataset("unit_pixel_indices")])
        mask = np.asarray(handle[dataset("map_mask")], dtype=bool)
        if responses.ndim != 2:
            raise ValueError("responses must have shape (n_stimuli, n_units)")
        n_stimuli, n_units = responses.shape
        if image_ids.shape != (n_stimuli,) or np.unique(image_ids).size != n_stimuli:
            raise ValueError("image_ids must be unique and match response rows")
        if coordinates.shape != (n_units, 2):
            raise ValueError("unit_coords_zero_based must have shape (n_units, 2)")
        if counts.shape != (n_units,) or offsets.shape != (n_units + 1,):
            raise ValueError("pool count/offset arrays do not match response columns")
        if not np.array_equal(np.diff(offsets), counts) or offsets[0] != 0 or offsets[-1] != len(members):
            raise ValueError("pool CSR membership is inconsistent")
        if np.any(counts <= 0):
            raise ValueError("every pooled unit must contain at least one raw pixel")
        expected = np.flatnonzero(mask.reshape(-1))
        if not np.array_equal(np.sort(members), expected):
            raise ValueError("pool members do not cover map_mask exactly once")
        if np.any(coordinates < 0) or np.any(coordinates[:, 0] >= mask.shape[0]) or np.any(coordinates[:, 1] >= mask.shape[1]):
            raise ValueError("unit coordinates are outside map_mask")
        finite = True
        zero_variance = 0
        minimum = np.inf
        maximum = -np.inf
        for start in range(0, n_stimuli, 32):
            block = np.asarray(responses[start : start + 32])
            finite &= bool(np.isfinite(block).all())
            minimum = min(minimum, float(block.min()))
            maximum = max(maximum, float(block.max()))
        if not finite:
            raise ValueError("responses contain NaN or Inf")
        sample_std = np.std(np.asarray(responses), axis=0)
        zero_variance = int(np.sum(sample_std <= np.finfo(np.float32).eps))

    data = config.get("data", {})
    pattern = data.get("image_filename_pattern", "nsd_1000_{image_id:05d}.jpg")
    rows = []
    for response_row, image_id in enumerate(image_ids.astype(int)):
        image_path = image_root / pattern.format(image_id=int(image_id))
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        rows.append({
            "response_row_zero_based": response_row,
            "image_id": int(image_id),
            "image_id_base": int(data.get("image_id_base", 1)),
            "image_path": str(image_path.resolve()),
        })
    manifest = pd.DataFrame(rows)
    qc = {
        "status": "valid",
        "responses_shape": [n_stimuli, n_units],
        "response_dtype": response_dtype,
        "image_count": n_stimuli,
        "unit_count": n_units,
        "zero_variance_units": zero_variance,
        "response_minimum": minimum,
        "response_maximum": maximum,
        "image_id_base": int(data.get("image_id_base", 1)),
        "h5_sha256": sha256_file(h5_path),
    }
    return manifest, qc


def read_responses(config: dict) -> np.ndarray:
    paths = project_paths(config)
    name = config.get("data", {}).get("datasets", {}).get("responses", "responses")
    with h5py.File(paths["neural_data"], "r") as handle:
        return np.asarray(handle[name], dtype=np.float32)


def read_unit_coordinates(config: dict) -> np.ndarray:
    paths = project_paths(config)
    name = config.get("data", {}).get("datasets", {}).get("unit_coords_zero_based", "unit_coords_zero_based")
    with h5py.File(paths["neural_data"], "r") as handle:
        return np.asarray(handle[name], dtype=np.int32)
