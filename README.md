# SinglePhotonMEI

SinglePhotonMEI is a standalone, offline-capable pipeline for fitting image-to-neural-response encoding models and generating maximally exciting images (MEIs) for pooled single-photon recordings.

The code is organized into two stages:

1. **Stage 1 — encoding-model selection:** validate the neural data, summarize population tuning, extract features from 4 backbones × 5 candidate layers, fit leakage-free nested Ridge models, and rank model/layer combinations by out-of-fold (OOF) prediction accuracy.
2. **Stage 2 — target selection and MEI generation:** select units by encoding accuracy, optionally enforce tuning diversity, refit per-target readouts, generate MEIs with neural guidance through Stable Diffusion 2.1 base, and optionally run cross-model peer review.

This repository does not import NeuroPred or Utils. Runtime model downloads are disabled: all weights must be supplied locally.

## Supported models

| Model | Role | Candidate layers |
|---|---|---|
| AlexNet, ImageNet1K V1 | Stage 1 candidate / possible primary model | `features.1`, `features.4`, `features.7`, `features.9`, `features.11` |
| ResNet50, ImageNet1K V2 | Stage 1 candidate / peer model | `relu`, `layer1`, `layer2`, `layer3`, `layer4` |
| Robust ResNet50, ImageNet L2 epsilon 0.5 | Stage 1 candidate / peer model | `relu`, `layer1`, `layer2`, `layer3`, `layer4` |
| DINOv2 ViT-B/14 with register tokens | Stage 1 candidate, peer model, and naturalness features | `blocks.0`, `blocks.2`, `blocks.5`, `blocks.8`, `blocks.11` |
| Stable Diffusion 2.1 base | MEI image generator | Local diffusers directory |

## Environment

The validated environment is:

- Python 3.10
- PyTorch 2.7.0 + CUDA 12.8
- torchvision 0.22.0 + CUDA 12.8
- timm 1.0.15
- diffusers 0.33.1
- transformers 4.52.3
- scikit-learn 1.6.1
- NumPy 1.26.4
- NVIDIA RTX 4080 SUPER with 16 GB VRAM

Exact package versions are recorded in `pyproject.toml` and `environment-lock.txt`. A CUDA-capable NVIDIA GPU is strongly recommended for feature extraction and required for practical 512 × 512, 50-step MEI generation.

### Online environment preparation

Create a Python 3.10 environment. For example, with Conda:

```bash
conda create -n single_photon_mei python=3.10 -y
conda activate single_photon_mei
```

Install the validated PyTorch CUDA build, then install this package:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

Check the environment:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
python scripts/check_environment.py
```

On PowerShell, use `$env:HF_HUB_OFFLINE="1"` and the corresponding syntax for the other variables. The CLI also sets these offline flags before importing model libraries.

### Fully offline HPC installation

On a connected machine with the same operating system, Python minor version, architecture, and CUDA target as the HPC:

1. Download the matching PyTorch/torchvision wheels from the official PyTorch index.
2. Download all remaining dependencies from `pyproject.toml` into a wheelhouse.
3. Transfer the wheelhouse, this repository, model weights, neural H5 file, and stimulus images to the HPC.
4. Install without an index:

```bash
python -m pip install --no-index --find-links /path/to/wheelhouse -e .
```

A prebuilt wheelhouse or packed Conda environment is not included. See `docs/offline_hpc.md` for additional HPC guidance.

## Required inputs

### 1. Neural response H5

The pipeline starts from pooled responses; raw-mask-to-pool preprocessing is outside this repository. The H5 file must contain:

| Dataset | Required shape | Meaning |
|---|---:|---|
| `responses` | `(n_stimuli, n_units)` | Neural response matrix |
| `image_ids` | `(n_stimuli,)` | Unique stimulus identifiers |
| `unit_coords_zero_based` | `(n_units, 2)` | Representative `(row, column)` for each pooled unit |
| `pool_counts` | `(n_units,)` | Number of valid raw pixels in each pooled unit |
| `unit_pixel_offsets` | `(n_units + 1,)` | CSR offsets into `unit_pixel_indices` |
| `unit_pixel_indices` | `(n_valid_pixels,)` | Flattened zero-based raw-pixel indices |
| `map_mask` | `(height, width)` | Valid-pixel mask |

Response row `k` must correspond exactly to `image_ids[k]`. Unit IDs are always zero-based response-column indices: unit `2539` is `responses[:, 2539]`, i.e. the 2540th column in one-based language.

`unit_coords_zero_based` is a representative coordinate only. Recover exact raw pixels through the CSR membership arrays:

```python
flat = unit_pixel_indices[unit_pixel_offsets[u]:unit_pixel_offsets[u + 1]]
rows, columns = numpy.unravel_index(flat, map_mask.shape)
```

See `docs/data_format.md` for the complete contract and validator behavior.

### 2. Stimulus image folder

The default example assumes 1-based image IDs and filenames such as:

```text
stimulus_images/
  nsd_1000_00001.jpg
  nsd_1000_00002.jpg
  ...
```

The mapping is configured by:

```json
{
  "image_id_base": 1,
  "image_filename_pattern": "nsd_1000_{image_id:05d}.jpg"
}
```

The validator checks every image before analysis.

### 3. Offline model weights

The GitHub repository intentionally keeps `model_weights/` empty except for `.gitkeep`. Download links are placeholders until they are supplied by the project owner.

| Model | Expected local path | SHA-256 | Download link |
|---|---|---|---|
| AlexNet ImageNet1K V1 | `model_weights/alexnet/alexnet-owt-7be5be79.pth` | `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` | **TBD — to be supplied** |
| ResNet50 ImageNet1K V2 | `model_weights/resnet50/resnet50-11ad3fa6.pth` | `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` | **TBD — to be supplied** |
| Robust ResNet50 L2 epsilon 0.5 | `model_weights/robust_resnet50/resnet50_l2_eps0.5.ckpt` | `a5fc6fcc54946b73af7bd74289b0003cc9d2744fa0ff55c1a3d9fb2bce8fbd30` | **TBD — to be supplied** |
| DINOv2 ViT-B/14 reg4 | `model_weights/dinov2/model.safetensors` | `c24ecfb4a1d8ca79193f6b9efcffc461872a09ddac43a0931357e4802931a006` | **TBD — to be supplied** |
| Stable Diffusion 2.1 base | `model_weights/diffusion/stable-diffusion-2-1-base/` | Per-file checksums in `docs/model_weights.md` | **TBD — to be supplied** |

After placing every weight file, build the required manifest:

```bash
python scripts/build_weight_manifest.py model_weights model_weights/manifest.json
```

The loaders use `weights=None`, `pretrained=False`, and `local_files_only=True`. Missing files and hash mismatches fail immediately; there is no online or random-weight fallback.

## Configuration

Copy the examples before editing:

```bash
cp configs/stage1.example.json configs/stage1.json
cp configs/stage2.example.json configs/stage2.json
```

Paths are resolved relative to the config file. At minimum, update:

- `paths.neural_data`
- `paths.stimulus_image_folder`
- `paths.model_weights`
- `paths.output`
- checkpoint paths and hashes if your files differ
- runtime device and batch sizes

Stage 1 exposes candidate models/layers, image preprocessing, full-resolution spatial-feature flattening, L2 normalization, PCA dimension, outer/inner folds, alpha grid, random seed, scoring, and ranking rules. No spatial pooling is applied: CNN feature maps are flattened at their native hook resolution, while DINOv2 patch tokens are restored to their native square grid and then flattened.

Stage 1 schema `single-photon-mei-stage1-2` is not feature-compatible with earlier 4 × 4-pooled runs. Use a new output directory and rerun Stage 1 and Stage 2 from the beginning. Native-resolution features, especially shallow ResNet layers, require substantially more disk space, host memory, and PCA compute than pooled features.

Stage 2 exposes Top-N or tuning-diverse selection, OOF/FDR eligibility, tuning-correlation threshold, per-target readout settings, regimes, neural guidance scales, seeds, image size, denoising steps, dtype, and peer-review thresholds. Version 1 intentionally rejects non-empty prompts, classifier-free guidance, SAG, and generation batches larger than one because those branches were not validated.

See `docs/configuration.md` for details.

## Running the pipeline

### Validate inputs

```bash
python -m single_photon_mei validate --config configs/stage1.json
```

### Stage 1 — encoding-model selection

Run all configured models and finalize the ranking:

```bash
python -m single_photon_mei stage1 run --config configs/stage1.json
python -m single_photon_mei stage1 finalize --config configs/stage1.json
```

For an HPC, split feature extraction and layer search by model:

```bash
python -m single_photon_mei stage1 run --config configs/stage1.json --model alexnet
python -m single_photon_mei stage1 run --config configs/stage1.json --model resnet50
python -m single_photon_mei stage1 run --config configs/stage1.json --model robust_resnet50_l2_eps0.5
python -m single_photon_mei stage1 run --config configs/stage1.json --model dinov2_vitb14
python -m single_photon_mei stage1 finalize --config configs/stage1.json
```

Every job must use the same config and shared output directory. Do not run two jobs for the same model/layer simultaneously.

### Stage 2 — target selection and MEI generation

```bash
python -m single_photon_mei stage2 select --config configs/stage2.json
python -m single_photon_mei stage2 generate --config configs/stage2.json
```

Optional cross-model peer review and final report:

```bash
python -m single_photon_mei stage2 peer-review --config configs/stage2.json
python -m single_photon_mei stage2 report --config configs/stage2.json
```

The example Stage 2 config selects 50 tuning-diverse units and generates 10 MEIs per target: five seeds at neural guidance 20 and the same five seeds at guidance 80.

## Outputs and resume behavior

Stage 1 writes:

- data manifest, QC, and input provenance;
- population-response ranking;
- per-model/layer feature caches;
- OOF predictions, unit-wise Pearson scores, fold assignments, and selected alphas;
- model/layer ranking and winner metadata.

Stage 2 writes:

- selected targets and full candidate audit;
- raw/z-scored target responses and tuning RDM;
- target-specific full-data readout;
- generated PNGs and a per-image manifest;
- generation QC, peer predictions, consensus tables, and final report.

Selection and finalization do not overwrite existing outputs unless `--overwrite` is explicit. MEI generation is resumable: each manifest row is bound to hashes of the resolved config, selected targets, and readout. Complete matching images are skipped; mismatches fail rather than silently mixing runs.

## Reproducibility and validation

The standalone implementation was regression-tested against the original workflow on the real 1,000-image, 10,914-unit dataset:

- AlexNet `features.11` features, OOF predictions, and unit scores matched exactly.
- Nested-CV fold alphas and Top-N targets matched exactly.
- Per-target readout weights, biases, and natural-image predictions matched to numerical precision.
- Tuning-diverse selection returned the same ordered 50 targets.
- A real 2-target × 2-regime × 1-seed smoke generated four 512 × 512, 50-step MEIs.

GPU diffusion is not guaranteed to be bitwise deterministic. In the validated CUDA environment, repeated fixed-seed runs had identical initial latents, image correlation 0.99437, and mean absolute pixel difference 3.80/255. Compare parameters, target predictions, clipping, and repeated-run stability rather than assuming identical PNG hashes across GPU runs.

Run the unit suite with:

```bash
python -m pytest -q
```

The local validation report documents what was and was not executed. The complete 4 × 5 search, a full 500-image regeneration, and complete peer-review were not rerun after migration. Cross-model agreement is computational QC; biological MEI validity requires new neural recordings.

## Repository layout

```text
configs/                    Example Stage 1 and Stage 2 configs
docs/                       Data, configuration, weight, HPC, and validation notes
model_weights/              Empty in Git; populate locally
scripts/                    Environment and weight-manifest checks
src/single_photon_mei/      Standalone implementation
tests/                      Synthetic and offline-loader tests
validation/                 Reusable comparison/isolation scripts and validation report
```

## License

This project is released under the MIT License. See `LICENSE`.
