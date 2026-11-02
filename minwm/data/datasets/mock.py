"""Mock latent dataset for smoke-testing the training data flow.

Generates random latents + camera params matching the CameraLatentLMDBDataset
contract, with no LMDB / disk dependency. For pipeline tests only.
"""

import torch
from torch.utils.data import Dataset

from minwm.processors import DISP_UNSHUFFLE

__all__ = ["MockCameraLatentDataset"]


class MockCameraLatentDataset(Dataset):
    """Random-latent dataset matching the camera-SFT batch contract.

    Args:
        length (int): number of samples.
        num_frames (int): latent frame count F.
        channels (int): latent channel count C.
        height (int): latent height H.
        width (int): latent width W.
        with_camera (bool): emit ``viewmats``/``Ks`` for PRoPE.
        ode_steps (int): if > 0, emit an ``ode_latent`` trajectory
            ``[ode_steps, F, C, H, W]`` (most-noisy -> clean) instead of a single
            ``clean_latent``. Must be ``len(denoising_step_list) + 1`` for the ODE
            recipe.
        with_disparity (bool): also emit dense ``disparity`` ``[F, 1, 4H, 4W]`` uniform
            in ``[0, 1]``, the input :class:`~minwm.processors.DisparityChannel` expects.
        seed (int): base RNG seed (per-sample deterministic).
    """

    def __init__(
        self,
        length: int = 16,
        num_frames: int = 20,
        channels: int = 16,
        height: int = 60,
        width: int = 104,
        with_camera: bool = True,
        ode_steps: int = 0,
        with_disparity: bool = False,
        seed: int = 0,
    ):
        self.length = length
        self.shape = (num_frames, channels, height, width)
        self.num_frames = num_frames
        self.with_camera = with_camera
        self.ode_steps = ode_steps
        self.with_disparity = with_disparity
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        g = torch.Generator().manual_seed(self.seed + idx)
        if self.ode_steps > 0:
            sample = {
                "prompts": f"mock prompt {idx}",
                "ode_latent": torch.randn(self.ode_steps, *self.shape, generator=g),
            }
        else:
            sample = {
                "prompts": f"mock prompt {idx}",
                "clean_latent": torch.randn(*self.shape, generator=g),
            }
        if self.with_disparity:
            frames, _, height, width = self.shape
            sample["disparity"] = torch.rand(
                frames, 1, DISP_UNSHUFFLE * height, DISP_UNSHUFFLE * width, generator=g
            )
        if self.with_camera:
            sample["viewmats"] = torch.eye(4).unsqueeze(0).repeat(self.num_frames, 1, 1)
            sample["Ks"] = torch.eye(3).unsqueeze(0).repeat(self.num_frames, 1, 1)
        return sample
