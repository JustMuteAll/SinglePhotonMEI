# Configuration interface

## Stage 1

`paths` locates the H5, image folder, offline weights and output. `data` maps H5 dataset names and the image-ID filename rule. Each `models` entry defines an architecture, family, local checkpoint, SHA-256, candidate layers and preprocessing statistics.

Result-bearing feature settings are image size, bilinear resize, L2 normalization and AMP. Spatial activations are not pooled: CNN feature maps are flattened at the hook's native resolution, and DINOv2 patch tokens are restored to their native square grid before flattening. Encoding settings are the Ridge alpha grid, outer/inner folds, shared-alpha mode, leakage-free PCA dimension, raw response scale, Pearson metric, shuffled folds, seed and fitted intercept. Ranking uses median OOF Pearson followed by mean Pearson, positive-score fraction and Top-50 mean when medians differ by less than the configured tolerance.

`stage1 run --model KEY` limits one execution to a configured model for HPC scheduling. The model subset, resolved-config hash, device and batch size are saved under `stage1/execution_scopes/`. This option does not alter scientific settings.

## Stage 2

Selection supports:

- `top_encoding`: eligible units sorted by OOF Pearson, then zero-based unit ID.
- `greedy_tuning_pearson`: the same order, accepting a candidate only if its signed NSD tuning Pearson is strictly below the configured threshold for every selected target.

Eligibility is configured by minimum OOF Pearson, two-sided response-prediction permutation count and BH-FDR maximum q. Readout settings contain the per-target alpha grid, folds, leakage-free PCA, target z-scoring, seed and intercept.

Generation exposes regime names and neural guidance strengths, seeds, images per target, resolution, steps, DPM solver, dtype and batch size. Version 1 fixes the prompt to empty, CFG and SAG to zero, batch size to one, and the objective to maximizing one target's predicted z-response. Target preprocessing must remain 224-pixel bilinear and match Stage 1.

Peer-review settings define the DINO PCA/5-NN naturalness baseline, per-regime naturalness and clipping limits, primary/peer response percentiles, cross-target selectivity and the required passing seeds.

## Output and resume rules

Stage 1 stores QC, stimulus manifest, population ranking, per-layer feature arrays, OOF predictions/scores/fold alphas, ranking and winner provenance. Stage 2 stores the selection audit, target response/RDM arrays, full-data readout, per-image manifest, generation QC, peer predictions and report.

Selection refuses existing output unless `--overwrite` is explicit. Generation is resumable and binds every row to the resolved Stage 2 config, selected-target file and readout hashes. Complete rows are skipped only when all hashes and the image file agree.
