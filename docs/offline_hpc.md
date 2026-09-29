# Offline HPC setup

This repository contains the code and weight-placement contract, but it does not include model files, a Python wheelhouse, or a packed Conda environment.

On a connected machine with the same operating system, Python minor version and CUDA target as the HPC, prepare dependencies from the pins in `environment-lock.txt`. For pip, download the matching PyTorch/CUDA wheels from the official PyTorch index and the remaining wheels into one directory. Transfer that directory and install with network indexes disabled:

```bash
python -m pip install --no-index --find-links /path/to/wheelhouse -e .
```

Alternatively, prepare and transfer a Conda-packed environment. Do not solve a new environment on the offline node if numerical regression against the validated run is required.

Before execution:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
python scripts/check_environment.py
python scripts/build_weight_manifest.py model_weights model_weights/manifest.recheck.json
```

Compare `manifest.recheck.json` with the transferred `manifest.json`. The CLI also sets the three offline variables and all runtime loaders explicitly disable pretrained downloads.

HPC model jobs can be split as:

```bash
python -m single_photon_mei stage1 run --config configs/stage1.json --model alexnet
python -m single_photon_mei stage1 run --config configs/stage1.json --model resnet50
python -m single_photon_mei stage1 run --config configs/stage1.json --model robust_resnet50_l2_eps0.5
python -m single_photon_mei stage1 run --config configs/stage1.json --model dinov2_vitb14
python -m single_photon_mei stage1 finalize --config configs/stage1.json
```

Use a shared filesystem with atomic rename support. Do not run two jobs for the same model/layer and output directory simultaneously.
