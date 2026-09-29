from unittest.mock import patch

import torch

from single_photon_mei.utils.diffusion import NeuralGuidedDiffusionPipeline


def test_diffusion_loader_forces_local_files(tmp_path):
    with patch("single_photon_mei.utils.diffusion.StableDiffusionPipeline.from_pretrained") as loader:
        fake = loader.return_value
        fake.scheduler.config = {}
        fake.to.return_value = fake
        with patch("single_photon_mei.utils.diffusion.DPMSolverMultistepScheduler.from_config", return_value=fake.scheduler):
            NeuralGuidedDiffusionPipeline.from_local(tmp_path, device="cpu", dtype=torch.float32)
    assert loader.call_args.kwargs["local_files_only"] is True
    assert loader.call_args.kwargs["use_safetensors"] is False
