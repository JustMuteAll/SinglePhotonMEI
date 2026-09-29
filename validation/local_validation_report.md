# Local standalone validation report

Date: 2026-09-29

Environment: Python 3.10, PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128, RTX 4080 SUPER

## Confirmed

- Source imports succeed with offline environment variables enabled.
- Import blocking for `NeuroPredictor` and `analysis_utils` passed while validating the real H5 and loading all four backbones from the standalone weight tree.
- All 17 materialized weight files occupy 3,480,126,477 bytes; the copied tree contains no links. SHA-256 checks passed for AlexNet, ResNet50, Robust-ResNet50 and DINOv2. Diffusion files were verified against `model_weights/manifest.json` before generation.
- Real-data contract: 1,000 stimuli x 10,914 pooled units; 1,000 images; zero non-finite responses and zero zero-variance units. H5 SHA-256: `fb9bd0b40336922989df79eb1de11cf1f25272e9efd7e5c633f27197e9d33e4d`.
- Lightweight Stage 1 ran AlexNet `features.11` with the full 5 x 5 nested CV, 50-component leakage-free PCA and nine-alpha grid. Encoding time was 26.69 seconds.
- Legacy/new Stage 1 feature shape was `(1000, 4096)`. Features, OOF predictions and all 10,914 Pearson scores had maximum absolute difference 0. The five selected outer-fold alphas were exactly `[1, 1, 1, 10, 1]` in both implementations.
- Top-2 target order was exactly `[2539, 2468]`. Per-target alphas were `[10, 10]`; readout weights, biases and natural predictions had maximum absolute difference 0 versus the legacy output (CV-score difference `1.67e-16`).
- Diversity selection replayed against the legacy eligibility mask selected the same ordered 50 units, scanned 1,014 candidates and had maximum accepted signed tuning correlation `0.7998443246`.
- Four real 50-step MEIs completed: 2 targets x 1 seed x natural/strong. All are valid 512 x 512 RGB PNGs; manifest count is four and no partial files remain.
- Fixed-seed diffusion repeatability is not bitwise on the current CUDA stack. The initial latent was exactly equal; two independent target-2539/natural/seed-1021 runs had pixel correlation `0.99437`, mean absolute pixel difference `3.80/255`, target-prediction difference `0.00973`, and clipping-fraction difference `0.00113`. This is recorded as numerical repeatability, not identical-file determinism.

## Not claimed

- The complete 4-backbone x 5-layer search was not rerun in the standalone repository; deterministic equivalence was established on the legacy winning layer and all four local loaders were smoke-tested.
- The standalone repository did not regenerate 500 MEIs. The real generation smoke is four images.
- Peer-review code is implemented and unit/import checked, but a full three-peer real-data run was not executed here.
- MEIs remain computational predictions; biological validity requires subsequent single-photon recording.

No Git repository, remote, commit or GitHub upload was created during this migration.
