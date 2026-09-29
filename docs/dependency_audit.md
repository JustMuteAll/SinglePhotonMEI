# Dependency audit

The original workflow was implemented in `NeuroPred/Scripts/single_photon_mei.py` and called code from both NeuroPred and Utils. The standalone package does not import either repository.

| Original symbol | Original repository | Workflow use | Standalone replacement | Decision |
|---|---|---|---|---|
| `NeuroPredictor.Encoder.Encoder` | NeuroPred | Nested Ridge, leakage-free PCA, per-target alpha, prediction and linear readout export | `utils/encoding.py`: `nested_ridge_cv`, `fit_per_target_readout` | Retained only for Ridge/PCA/Pearson |
| `analysis_utils.cv` helpers | Utils | Reproducible shuffled folds and leakage-free preprocessing | Explicit `KFold`, fold-local PCA and Ridge in `utils/encoding.py` | Minimal reimplementation |
| `analysis_utils.metrics.pearson_by_target` | Utils | Unit-wise OOF Pearson | `utils/encoding.py:pearson_by_target` | Minimal reimplementation |
| `NeuroPredictor.Utils.get_extractor` | NeuroPred | Construct torchvision/timm extractors | `utils/models.py:load_backbone` | Replaced; only local checkpoints are accepted |
| `NeuroPredictor.FeatExtractor.TorchvisionFeatureExtractor` and `TimmFeatureExtractor` | NeuroPred | Layer hooks, AMP, robust checkpoint loading | `utils/models.py:HookExtractor`, `_strip_prefixes` | Reduced to the four supported backbones |
| `NeuroPredictor.brain_guide_pipeline.mypipelineSAG` | NeuroPred | Stable Diffusion latent optimization | `utils/diffusion.py:NeuralGuidedDiffusionPipeline` | Reduced to empty prompt, CFG=0, SAG=0 and DPM solver |
| PLS, OpenCLIP, SSL/factorized models and visualization helpers | NeuroPred | Not used by the approved single-photon workflow | None | Not migrated |

The internal code deliberately uses ordinary functions and two small stateful wrappers (a hook extractor and a diffusion pipeline). It contains no plugin or model registry. A source scan in the regression tests rejects imports containing `NeuroPredictor` or `analysis_utils`.
