import importlib.metadata
import os
import sys

import torch


expected = {
    "numpy": "1.26.4",
    "scipy": "1.15.3",
    "pandas": "2.2.3",
    "h5py": "3.14.0",
    "Pillow": "11.0.0",
    "matplotlib": "3.10.3",
    "scikit-learn": "1.6.1",
    "torch": "2.7.0+cu128",
    "torchvision": "0.22.0+cu128",
    "timm": "1.0.15",
    "diffusers": "0.33.1",
    "transformers": "4.52.3",
    "accelerate": "1.8.1",
    "safetensors": "0.5.3",
}
errors = []
if sys.version_info[:2] != (3, 10):
    errors.append(f"Python 3.10 required, found {sys.version.split()[0]}")
for name, version in expected.items():
    actual = importlib.metadata.version(name)
    if actual != version:
        errors.append(f"{name}: expected {version}, found {actual}")
for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DIFFUSERS_OFFLINE"):
    if os.environ.get(name) != "1":
        errors.append(f"{name}=1 is required")
print(f"torch={torch.__version__}, cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
if errors:
    raise SystemExit("\n".join(errors))
print("environment check passed")
