import json

import h5py
import numpy as np
from PIL import Image

from single_photon_mei.utils.io import load_config, validate_neural_data


def test_data_contract_and_one_based_filenames(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    for image_id in (2, 1):
        Image.new("RGB", (4, 4), (image_id, 0, 0)).save(image_root / f"image_{image_id:03d}.jpg")
    h5_path = tmp_path / "data.h5"
    with h5py.File(h5_path, "w") as handle:
        handle["responses"] = np.array([[1, 2], [3, 5]], dtype=np.float32)
        handle["image_ids"] = np.array([2, 1])
        handle["unit_coords_zero_based"] = np.array([[0, 0], [0, 1]])
        handle["pool_counts"] = np.array([1, 1])
        handle["unit_pixel_offsets"] = np.array([0, 1, 2])
        handle["unit_pixel_indices"] = np.array([0, 1])
        handle["map_mask"] = np.array([[1, 1]], dtype=bool)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "paths": {"neural_data": str(h5_path), "stimulus_image_folder": str(image_root), "model_weights": "weights", "output": "out"},
        "data": {"image_id_base": 1, "image_filename_pattern": "image_{image_id:03d}.jpg", "datasets": {}},
    }), encoding="utf-8")
    manifest, qc = validate_neural_data(load_config(config_path))
    assert manifest["image_id"].tolist() == [2, 1]
    assert manifest["response_row_zero_based"].tolist() == [0, 1]
    assert qc["responses_shape"] == [2, 2]
