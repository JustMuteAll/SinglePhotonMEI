import argparse
import importlib.abc
import json
import os
import sys
from pathlib import Path


class BlockLegacyRepositories(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "NeuroPredictor" or fullname.startswith("NeuroPredictor.") or fullname == "analysis_utils" or fullname.startswith("analysis_utils."):
            raise ImportError(f"blocked legacy dependency: {fullname}")
        return None


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--models-config")
args = parser.parse_args()
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFUSERS_OFFLINE="1")
sys.meta_path.insert(0, BlockLegacyRepositories())
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from single_photon_mei.utils.io import load_config, project_paths, validate_neural_data
from single_photon_mei.utils.models import load_backbone


config = load_config(args.config)
model_config = load_config(args.models_config) if args.models_config else config
manifest, qc = validate_neural_data(config)
loaded = []
for key, spec in model_config["models"].items():
    extractor, metadata = load_backbone(spec, project_paths(model_config)["weights"], "cpu", amp=False)
    loaded.append({"model": key, "checkpoint_sha256": metadata["checkpoint_sha256"]})
    del extractor
print(json.dumps({"status": "pass", "manifest_rows": len(manifest), "qc": qc, "loaded_models": loaded}, indent=2))
