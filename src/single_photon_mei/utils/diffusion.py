from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class NeuralGuidedDiffusionPipeline:
    """Minimal offline Stable Diffusion path used by the validated MEI workflow."""

    def __init__(self, pipe: StableDiffusionPipeline):
        self.pipe = pipe
        self.objective: Callable[[torch.Tensor], torch.Tensor] | None = None
        self._clip_stats_cache = {}

    @classmethod
    def from_local(cls, path: str | Path, *, device: str, dtype: torch.dtype):
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(path)
        pipe = StableDiffusionPipeline.from_pretrained(
            str(path),
            torch_dtype=dtype,
            local_files_only=True,
            use_safetensors=False,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)
        return cls(pipe)

    def _clip_stats(self, image: torch.Tensor):
        key = (image.device.type, image.device.index, image.dtype)
        if key not in self._clip_stats_cache:
            mean = torch.tensor(CLIP_MEAN, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
            std = torch.tensor(CLIP_STD, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
            self._clip_stats_cache[key] = mean, std
        return self._clip_stats_cache[key]

    @torch.enable_grad()
    def _guided_noise(self, latents, timestep, text_embeddings, noise_pred, scale):
        if self.objective is None:
            raise RuntimeError("A scalar neural objective must be assigned before generation")
        variables = latents.detach().requires_grad_()
        latent_input = self.pipe.scheduler.scale_model_input(variables, timestep)
        guided_noise = self.pipe.unet(latent_input, timestep, encoder_hidden_states=text_embeddings).sample
        alpha = self.pipe.scheduler.alphas_cumprod[timestep]
        beta = 1 - alpha
        predicted_x0 = (variables - beta.sqrt() * guided_noise) / alpha.sqrt()
        mix = predicted_x0 * beta.sqrt() + variables * (1 - beta.sqrt())
        decoded = self.pipe.vae.decode(mix / self.pipe.vae.config.scaling_factor).sample
        image = (decoded / 2 + 0.5).clamp(0, 1).float()
        mean, std = self._clip_stats(image)
        normalized = (F.interpolate(image, size=224, mode="bilinear", align_corners=False) - mean) / std
        loss = self.objective(normalized) * float(scale)
        if loss.numel() != 1:
            raise ValueError("Neural objective must return exactly one scalar")
        gradient = -torch.autograd.grad(loss, variables)[0]
        return noise_pred - beta.sqrt() * gradient, variables.detach()

    @torch.no_grad()
    def generate(
        self,
        *,
        seed: int,
        neural_guidance_scale: float,
        steps: int,
        height: int,
        width: int,
        device: str,
    ):
        pipe = self.pipe
        device = torch.device(device)
        generator = torch.Generator(device=device).manual_seed(int(seed))
        prompt_embeddings, _ = pipe.encode_prompt(
            prompt="",
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        pipe.scheduler.set_timesteps(int(steps), device=device)
        latents = pipe.prepare_latents(
            1,
            pipe.unet.config.in_channels,
            int(height),
            int(width),
            prompt_embeddings.dtype,
            device,
            generator,
            None,
        )
        extra = pipe.prepare_extra_step_kwargs(generator, 0.0)
        for index, timestep in enumerate(pipe.scheduler.timesteps):
            latent_input = pipe.scheduler.scale_model_input(latents, timestep)
            noise = pipe.unet(latent_input, timestep, encoder_hidden_states=prompt_embeddings).sample
            noise, latents = self._guided_noise(
                latents, timestep, prompt_embeddings, noise, neural_guidance_scale
            )
            latents = pipe.scheduler.step(noise, timestep, latents, **extra).prev_sample
        image = pipe.decode_latents(latents)
        return pipe.numpy_to_pil(np.asarray(image))[0]
