# SinglePhotonMEI

SinglePhotonMEI is a standalone, offline-capable pipeline for fitting image-to-neural-response encoding models and generating maximally exciting images (MEIs) for pooled single-photon recordings.

The code is organized into two stages:

1. **Stage 1 — encoding-model selection:** validate the neural data, summarize population tuning, extract features from 4 backbones × 5 candidate layers, fit leakage-free nested Ridge models, and rank model/layer combinations by out-of-fold (OOF) prediction accuracy.
2. **Stage 2 — target selection and MEI generation:** select units by encoding accuracy, optionally enforce tuning diversity, refit per-target readouts, generate MEIs with neural guidance through Stable Diffusion 2.1 base, and optionally run cross-model peer review.


## Supported models

| Model | Candidate layers |
|---|---|
| AlexNet, ImageNet1K V1 | `features.1`, `features.4`, `features.7`, `features.9`, `features.11` |
| ResNet50, ImageNet1K V2 |  `relu`, `layer1`, `layer2`, `layer3`, `layer4` |
| Robust ResNet50, ImageNet L2 epsilon 0.5 |  `relu`, `layer1`, `layer2`, `layer3`, `layer4` |
| DINOv2 ViT-B/14 with register tokens |  `blocks.0`, `blocks.2`, `blocks.5`, `blocks.8`, `blocks.11` |

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

## Required inputs

Prepare the neural data, stimulus images, runtime configs, and offline model weights before running the pipeline. A recommended project layout is:

```text
SinglePhotonMEI/
├── data/
│   ├── neural_data.h5
│   └── stimulus_images/
├── configs/
│   ├── stage1.json
│   └── stage2.json
└── model_weights/
```

- `data/neural_data.h5`: pooled neural responses and the unit/pixel metadata used by both stages.
- `data/stimulus_images/`: stimulus images aligned to the response rows through the image IDs stored in the H5 file.
- `configs/stage1.json`: model, layer, feature-extraction, PCA, Ridge, cross-validation, path, and runtime settings for encoding-model selection.
- `configs/stage2.json`: target-selection, final-readout, MEI-generation, peer-review, path, and runtime settings.
- `model_weights/`: all backbone and diffusion weights required for fully offline execution. Arrange the files exactly as follows:

```text
SinglePhotonMEI/
└── model_weights/
    ├── alexnet/
    │   └── alexnet-owt-7be5be79.pth
    ├── resnet50/
    │   └── resnet50-11ad3fa6.pth
    ├── robust_resnet50/
    │   └── resnet50_l2_eps0.5.ckpt
    ├── dinov2/
    │   └── model.safetensors
    └── diffusion/
        └── stable-diffusion-2-1-base/
            ├── model_index.json
            ├── scheduler/
            ├── text_encoder/
            ├── tokenizer/
            ├── unet/
            └── vae/
```

The paths in `stage1.json` and `stage2.json` must point to these local inputs. The repository does not download missing data or model weights at runtime.

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

## License

This project is released under the MIT License. See `LICENSE`.
