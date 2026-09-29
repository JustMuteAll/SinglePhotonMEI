# Required offline model weights

The GitHub repository intentionally leaves `model_weights/` empty except for `.gitkeep`. Runtime network access is disabled. Every checkpoint must be obtained separately, copied into the paths below, and verified before running the workflow.

Download links are intentionally left as `TBD` until supplied by the project owner.

## Backbone checkpoints

| Model | Local path relative to `model_weights/` | SHA-256 | Download link |
|---|---|---|---|
| AlexNet ImageNet1K V1 | `alexnet/alexnet-owt-7be5be79.pth` | `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` | **TBD — to be supplied** |
| ResNet50 ImageNet1K V2 | `resnet50/resnet50-11ad3fa6.pth` | `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` | **TBD — to be supplied** |
| Robust ResNet50 L2 epsilon 0.5 | `robust_resnet50/resnet50_l2_eps0.5.ckpt` | `a5fc6fcc54946b73af7bd74289b0003cc9d2744fa0ff55c1a3d9fb2bce8fbd30` | **TBD — to be supplied** |
| DINOv2 ViT-B/14 reg4 | `dinov2/model.safetensors` | `c24ecfb4a1d8ca79193f6b9efcffc461872a09ddac43a0931357e4802931a006` | **TBD — to be supplied** |

## Stable Diffusion 2.1 base

Place a complete, materialized diffusers directory at:

```text
model_weights/diffusion/stable-diffusion-2-1-base/
  model_index.json
  scheduler/scheduler_config.json
  text_encoder/config.json
  text_encoder/pytorch_model.bin
  tokenizer/merges.txt
  tokenizer/special_tokens_map.json
  tokenizer/tokenizer_config.json
  tokenizer/vocab.json
  unet/config.json
  unet/diffusion_pytorch_model.bin
  vae/config.json
  vae/diffusion_pytorch_model.bin
```

Download link: **TBD — to be supplied**.

Validated per-file SHA-256 values:

| Relative path below `stable-diffusion-2-1-base/` | SHA-256 |
|---|---|
| `model_index.json` | `b552458d23b0134202102f459379c4e21e221da8c5208c07831ad008b6b44cf7` |
| `scheduler/scheduler_config.json` | `11ac5627d7df0fa344b875c4b5722b1767a8a2aa1684c2cf8b4d614300127234` |
| `text_encoder/config.json` | `3026979f213950ec09e8ed8c4ca55d977122d18c054996dbf2d645dc5259d688` |
| `text_encoder/pytorch_model.bin` | `f2a06cf32cf585d03b55fef302142a5321b761ec440113925f64f4ceaffc7730` |
| `tokenizer/merges.txt` | `9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a` |
| `tokenizer/special_tokens_map.json` | `f118ab3a983206e4f32583448de6bd6aae4ee21869135cef1f5848a753cdaab6` |
| `tokenizer/tokenizer_config.json` | `01ed91143a3787fccc6b660edca1739c20d2c3b7aa0845707a01fe4195163051` |
| `tokenizer/vocab.json` | `e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349` |
| `unet/config.json` | `f9b9833e4d273bae4b88419af5cd303d43875bf4f99a959e38686d83cd4b6c4e` |
| `unet/diffusion_pytorch_model.bin` | `39b5e4464ce1057da982b12aaa71fbaeb9832ec8e54fceb4ed2c867c92e51a53` |
| `vae/config.json` | `8c3991264745f5e72db7f42c866d9c2742ab7abac1a58f0de4a17d5164dcd96a` |
| `vae/diffusion_pytorch_model.bin` | `11bc15ceb385823b4adb68bd5bdd7568d0c706c3de5ea9ebcb0b807092fc9030` |

Copy actual files rather than Hugging Face cache symlinks. The runtime calls diffusers with `local_files_only=True` and `use_safetensors=False` for this validated `.bin` layout.

## Verification

After all files are present, generate the manifest required by Stage 2:

```bash
python scripts/build_weight_manifest.py model_weights model_weights/manifest.json
```

Inspect the result and retain it locally with the weights. The manifest is ignored by Git because `model_weights/` is intentionally not versioned.

The backbone loaders also compare the checkpoint hashes in the Stage 1 config. Missing files, missing manifest entries, and hash mismatches are fatal; there is no download or fallback path.
