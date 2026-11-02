"""Latent and latent-noise input preparation for inference."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .base import BatchPreprocessor, InferenceInputPreprocessor, InferenceRuntime


class LatentNoise(InferenceInputPreprocessor):
    """Create or place the ``noise`` tensor used as the generation seed."""

    def __init__(self, key: str = "noise") -> None:
        self.key = key

    def _latent_shape(self, inference_cfg: dict) -> tuple[int, int, int, int]:
        channels, height, width = inference_cfg["latent_shape"]
        return (
            int(inference_cfg["num_frames"]),
            int(channels),
            int(height),
            int(width),
        )

    def __call__(self, batch: dict, runtime: InferenceRuntime) -> dict:
        if self.key in batch and batch[self.key] is not None:
            batch[self.key] = batch[self.key].to(device=runtime.device, dtype=runtime.dtype)
            return batch

        prompts = batch["prompts"]
        num_frames, channels, height, width = self._latent_shape(runtime.inference_cfg)
        batch[self.key] = torch.randn(
            [len(prompts), num_frames, channels, height, width],
            device=runtime.device,
            dtype=runtime.dtype,
        )
        return batch


class InitialLatent(InferenceInputPreprocessor):
    """VAE-encode input images into a clean latent prefix.

    The generation pipeline consumes only ``initial_latent``; image loading and
    VAE calling stay in this input preprocessor. The supported VAE contract is
    either Wan's ``encode(list[C,T,H,W]) -> list[C,T,h,w]`` or a diffusers-style
    ``encode(B,C,T,H,W)`` result exposing ``latent_dist``.
    """

    def __init__(self, image_key: str = "image", output_key: str = "initial_latent") -> None:
        self.image_key = image_key
        self.output_key = output_key

    @staticmethod
    def _target_size(runtime: InferenceRuntime) -> tuple[int, int]:
        cfg = runtime.inference_cfg
        _, latent_h, latent_w = cfg["latent_shape"]
        return (
            int(cfg.get("target_height", int(latent_h) * 16)),
            int(cfg.get("target_width", int(latent_w) * 16)),
        )

    @staticmethod
    def _load(path: str, height: int, width: int) -> torch.Tensor:
        from PIL import Image

        image = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
        return (torch.from_numpy(array).permute(2, 0, 1) / 255.0 - 0.5) * 2.0

    @torch.no_grad()
    def __call__(self, batch: dict, runtime: InferenceRuntime) -> dict:
        source = batch.get(self.image_key)
        if source is None:
            return batch

        batch_size = len(batch["prompts"])
        height, width = self._target_size(runtime)
        if isinstance(source, str):
            images = torch.stack([self._load(source, height, width)] * batch_size)
        elif isinstance(source, (list, tuple)) and source and isinstance(source[0], str):
            if len(source) != batch_size:
                raise ValueError(f"got {len(source)} images for batch size {batch_size}")
            images = torch.stack([self._load(path, height, width) for path in source])
        elif isinstance(source, torch.Tensor):
            images = source
            if images.ndim == 3:
                images = images.unsqueeze(0)
            if images.ndim == 5:
                if images.shape[2] != 1:
                    raise ValueError("InitialLatent accepts one image per sample")
                images = images.squeeze(2)
            if images.ndim != 4:
                raise ValueError("image tensor must have shape [B,C,H,W] or [B,C,1,H,W]")
        else:
            raise TypeError("image must be a path, list of paths, or tensor")

        vae = runtime.components["vae"]
        images = images.to(device=runtime.device, dtype=runtime.dtype)
        videos = [image.unsqueeze(1) for image in images]
        try:
            encoded = vae.encode(videos)
        except (TypeError, AttributeError):
            encoded = vae.encode(torch.stack(videos))

        if isinstance(encoded, (list, tuple)):
            latent = torch.stack(list(encoded))
        else:
            latent_dist = getattr(encoded, "latent_dist", None)
            latent = latent_dist.mode() if latent_dist is not None else encoded
            scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", 1.0)
            latent = latent * scaling_factor
        if latent.ndim != 5:
            raise ValueError(f"VAE encode must return [B,C,T,H,W], got {tuple(latent.shape)}")
        batch[self.output_key] = latent.permute(0, 2, 1, 3, 4).to(runtime.dtype)
        return batch


class LatentToDevice(BatchPreprocessor):
    """Move + cast latent and camera tensors onto the working device/dtype.

    Args:
        dtype (torch.dtype, optional): working dtype for the moved tensors. Left
            ``None`` when built from config; the recipe injects its runtime dtype
            in :meth:`~minwm.engine.recipe_base.Recipe.build_preprocessors`.
        keys (tuple[str, ...]): batch keys to move if present. Defaults cover
            every recipe's latent + camera tensors.
    """

    def __init__(
        self,
        dtype: torch.dtype | None = None,
        keys: tuple[str, ...] = ("clean_latent", "ode_latent", "viewmats", "Ks"),
    ) -> None:
        self.dtype = dtype
        self.keys = keys

    def __call__(self, batch: dict, device: torch.device) -> dict:
        for key in self.keys:
            if key in batch and batch[key] is not None:
                batch[key] = batch[key].to(device=device, dtype=self.dtype)
        return batch


DISP_UNSHUFFLE = 4
DISP_CHANNELS = DISP_UNSHUFFLE**2


def disparity_to_channels(disparity: Tensor) -> Tensor:
    """Pixel-unshuffle dense disparity onto the latent grid.

    Args:
        disparity (Tensor): ``[..., 1, H * DISP_UNSHUFFLE, W * DISP_UNSHUFFLE]``.

    Returns:
        Tensor: ``[..., DISP_CHANNELS, H, W]``; the value at pixel ``(4h + i, 4w + j)``
        lands at channel ``i * 4 + j`` of latent cell ``(h, w)``. Values are unchanged.
    """
    return F.pixel_unshuffle(disparity, DISP_UNSHUFFLE)


def disparity_from_channels(channels: Tensor) -> Tensor:
    """Pixel-shuffle disparity channels back to dense disparity.

    The exact inverse of :func:`disparity_to_channels`.

    Args:
        channels (Tensor): ``[..., DISP_CHANNELS, H, W]``.

    Returns:
        Tensor: ``[..., 1, H * DISP_UNSHUFFLE, W * DISP_UNSHUFFLE]``, values unchanged.

    Raises:
        ValueError: if the channel dim is not ``DISP_CHANNELS``.
    """
    if channels.ndim < 3 or channels.shape[-3] != DISP_CHANNELS:
        raise ValueError(
            f"expected {DISP_CHANNELS} disparity channels at dim -3, got {tuple(channels.shape)}"
        )
    return F.pixel_shuffle(channels, DISP_UNSHUFFLE)


class DisparityChannel(BatchPreprocessor):
    """Append dense ``disparity`` to ``clean_latent`` as ``DISP_CHANNELS`` extra channels.

    Reads ``disparity`` ``[B, F, 1, 4H, 4W]`` (dense disparity, 4× the latent grid per
    side), with values ``d`` in ``[0, 1]`` (1 = nearest), maps it to ``scale·(2d − 1)``,
    pixel-unshuffles it by :data:`DISP_UNSHUFFLE` (see :func:`disparity_to_channels`) and
    writes ``clean_latent`` ``[B, F, C + DISP_CHANNELS, H, W]`` on the latent's device and
    dtype, disparity channels after the latent channels. Place it after
    :class:`LatentToDevice` and before :class:`~minwm.processors.flow.FlowNoise`, so the
    noise, the flow target and the clean context all cover the new channels unchanged.

    Args:
        scale (float, optional): disparity amplitude ``a``. ``2d − 1`` first maps
            ``[0, 1]`` to ``[−1, 1]``; Wan latents are normalized to about unit std per
            channel. Leave it ``None``: ``a`` is set once, as the model's
            ``disparity_scale``, and bound here by :meth:`resolve_scale`.

    Raises:
        ValueError: if ``scale`` is still unresolved, if ``disparity`` is not
            ``[B, F, 1, 4H, 4W]`` for a ``[B, F, C, H, W]`` latent, or if it has values
            outside ``[0, 1]`` (NaN included).
    """

    def __init__(self, scale: float | None = None) -> None:
        self.scale = scale

    def resolve_scale(self, model_scale: float | None) -> None:
        """Bind ``scale`` to the model's ``disparity_scale``.

        Args:
            model_scale (float | None): the model's ``disparity_scale``.

        Raises:
            ValueError: if ``model_scale`` is ``None``, or ``scale`` was set explicitly
                to a different value.
        """
        if model_scale is None:
            raise ValueError("DisparityChannel needs a model with disparity_scale set")
        if self.scale is not None and self.scale != model_scale:
            raise ValueError(
                f"DisparityChannel scale={self.scale} differs from the model's "
                f"disparity_scale={model_scale}; set a only on the model"
            )
        self.scale = model_scale

    def __call__(self, batch: dict, device: torch.device) -> dict:
        if self.scale is None:
            raise ValueError(
                "DisparityChannel scale is unresolved; the recipe binds it from the model's "
                "disparity_scale (see resolve_scale)"
            )
        clean, disparity = batch["clean_latent"], batch["disparity"]
        height, width = clean.shape[3:]
        expected = (*clean.shape[:2], 1, height * DISP_UNSHUFFLE, width * DISP_UNSHUFFLE)
        if tuple(disparity.shape) != expected:
            raise ValueError(f"disparity must be {expected}, got {tuple(disparity.shape)}")
        # NaN fails both comparisons, so this also rejects non-finite values.
        bad = ~((disparity >= 0) & (disparity <= 1))
        if bool(bad.any()):
            raise ValueError(
                f"disparity must be finite and lie in [0, 1]; {int(bad.sum())} of "
                f"{bad.numel()} values do not"
            )
        channels = disparity_to_channels(self.scale * (2 * disparity.float() - 1))
        batch["clean_latent"] = torch.cat([clean, channels.to(clean)], dim=2)
        return batch
