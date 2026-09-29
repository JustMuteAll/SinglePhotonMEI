from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors

from .io import sha256_file


def _checkpoint_state(checkpoint) -> dict:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state dictionary")
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    return checkpoint


def _strip_prefixes(state: dict, model: nn.Module) -> dict:
    model_keys = set(model.state_dict())
    cleaned = {}
    unexpected = []
    for original, value in state.items():
        key = original
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "attacker.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        if key in model_keys:
            cleaned[key] = value
        elif not key.startswith(("normalizer.", "normalize.")):
            unexpected.append(original)
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys: {unexpected[:10]}")
    return cleaned


class HookExtractor:
    def __init__(self, model: nn.Module, device: str, amp: bool):
        self.model = model.eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.device = device
        self.amp = amp

    def _autocast(self):
        if self.amp and self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.amp.autocast(device_type="cuda")
        return nullcontext()

    def extract_features(self, images: torch.Tensor, layers: str | list[str]):
        names = [layers] if isinstance(layers, str) else list(layers)
        modules = dict(self.model.named_modules())
        missing = [name for name in names if name not in modules]
        if missing:
            raise ValueError(f"Unknown model layers: {missing}")
        outputs = {}
        hooks = []
        for name in names:
            hooks.append(modules[name].register_forward_hook(
                lambda _module, _inputs, output, key=name: outputs.__setitem__(key, output)
            ))
        try:
            with self._autocast():
                self.model(images.to(self.device))
        finally:
            for hook in hooks:
                hook.remove()
        if any(name not in outputs for name in names):
            raise RuntimeError("Feature hook did not capture every requested layer")
        return outputs[names[0]] if isinstance(layers, str) else outputs


def _verify_checkpoint(path: Path, expected_hash: str | None) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if expected_hash and actual.lower() != expected_hash.lower():
        raise ValueError(f"Checkpoint SHA-256 mismatch for {path}: {actual}")
    return actual


def load_backbone(spec: dict, weights_root: Path, device: str, amp: bool = True) -> tuple[HookExtractor, dict]:
    checkpoint = (weights_root / spec["checkpoint"]).resolve()
    actual_hash = _verify_checkpoint(checkpoint, spec.get("sha256"))
    family = spec["family"]
    if family in {"alexnet", "resnet50", "robust_resnet50"}:
        from torchvision import models

        constructor = getattr(models, spec["architecture"])
        model = constructor(weights=None)
        state = _checkpoint_state(torch.load(checkpoint, map_location="cpu", weights_only=False))
        if family == "robust_resnet50":
            state = _strip_prefixes(state, model)
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                f"Checkpoint is not an exact {spec['architecture']} match: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
    elif family == "dinov2_vit":
        import timm

        model = timm.create_model(spec["architecture"], pretrained=False)
        incompatible = model.load_state_dict(load_safetensors(str(checkpoint)), strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                f"DINOv2 checkpoint mismatch: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        image_size = int(spec.get("image_size", 224))
        if hasattr(model, "set_input_size"):
            model.set_input_size(img_size=(image_size, image_size))
    else:
        raise ValueError(f"Unsupported model family: {family}")
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_hash,
        "architecture": spec["architecture"],
        "family": family,
        "mean": list(spec["preprocess"]["mean"]),
        "std": list(spec["preprocess"]["std"]),
    }
    return HookExtractor(model, device, amp), metadata


def adapt_activation(
    activation: torch.Tensor,
    *,
    family: str,
    l2_normalize: bool,
    num_prefix_tokens: int = 1,
) -> torch.Tensor:
    if family == "dinov2_vit":
        if activation.ndim != 3:
            raise ValueError(f"ViT activation must be 3D, got {activation.shape}")
        patches = activation[:, num_prefix_tokens:, :]
        side = math.isqrt(patches.shape[1])
        if side * side != patches.shape[1]:
            raise ValueError("ViT patch-token count is not square after removing prefix tokens")
        activation = patches.transpose(1, 2).reshape(patches.shape[0], patches.shape[2], side, side)
    if activation.ndim != 4:
        raise ValueError(f"Spatial activation must be 4D, got {activation.shape}")
    features = activation.flatten(1).float()
    return F.normalize(features, p=2, dim=1, eps=1e-12) if l2_normalize else features
