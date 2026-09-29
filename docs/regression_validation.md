# Regression validation

Validation distinguishes implementation checks from completed real-data runs.

1. Unit tests cover the H5 contract, Pearson/Ridge/PCA, target-selection boundary behavior, local-only loaders and config rejection for unsupported diffusion branches.
2. `validation/compare_with_legacy.py` compares stored old/new arrays: feature shape/value, fold assignment, alpha, OOF prediction, score, selected target order, tuning matrix and exported readout prediction.
3. Real lightweight validation uses one backbone/layer on the real 1,000-image dataset, then selects two targets and generates one seed in each of the natural and strong regimes (four 50-step images).
4. Isolation validation runs with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `DIFFUSERS_OFFLINE=1`, the standalone `src` as `PYTHONPATH`, and an import blocker for `NeuroPredictor` and `analysis_utils`.

CPU/scikit-learn outputs should agree near machine precision. GPU AMP feature comparison uses explicit absolute and relative tolerances while requiring fold assignments, selected alphas, target ordering and ranking to match exactly. If diffusion kernels are not bitwise deterministic, compare initial seed/config, target prediction, clipping fraction and within-environment repeatability rather than claiming identical PNG hashes.
