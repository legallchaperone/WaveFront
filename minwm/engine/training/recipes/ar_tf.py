"""Autoregressive teacher-forcing recipe."""

from typing import Any

import torch
from torch import nn

from ..losses import mse_loss
from ._flow_base import SingleForwardFlowRecipe


class ARTFRecipe(SingleForwardFlowRecipe):
    """Autoregressive teacher forcing: single causal forward with clean context.

    Validation scores the denoising error on held-out clips at fixed noise levels.
    Each clip is noised to every ``val_sigmas`` level with noise seeded by
    ``val_seed`` plus the clip's index, so the numbers are comparable across steps
    and runs. Every frame gets the same level, and the forward is the training one.

    Args:
        val_sigmas (tuple[float, ...]): noise levels σ in ``(0, 1]`` to validate at.
        val_seed (int): base seed of the fixed validation noise.
        **kwargs: forwarded to :class:`SingleForwardFlowRecipe`.

    Raises:
        ValueError: if a ``val_sigmas`` entry is outside ``(0, 1]``.
    """

    def __init__(
        self,
        val_sigmas: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
        val_seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not all(0 < sigma <= 1 for sigma in val_sigmas):
            raise ValueError(f"val_sigmas must lie in (0, 1], got {tuple(val_sigmas)}")
        self.val_sigmas = tuple(val_sigmas)
        self.val_seed = val_seed

    def validation_step(self, model: nn.Module, batch: dict, index: int) -> dict[str, float]:
        """Unweighted velocity MSE at each ``val_sigmas`` level, plus their mean.

        Args:
            model (nn.Module): the model under evaluation.
            batch (dict): one collated held-out sample.
            index (int): the sample's index in the val dataset; seeds its noise.

        Returns:
            dict[str, float]: ``val/loss@<σ>`` for each level and ``val/loss``, their mean.
        """
        device = next(model.parameters()).device
        batch_size, num_frames = batch["clean_latent"].shape[:2]
        metrics: dict[str, float] = {}
        for sigma in self.val_sigmas:
            timestep = sigma * self.scheduler.num_train_timesteps
            noised = self.preprocess(
                {
                    **batch,
                    "timestep": torch.full((batch_size, num_frames), timestep, device=device),
                    "noise_generator": torch.Generator().manual_seed(self.val_seed + index),
                },
                device,
            )
            cond = self.adapter.conditioning(noised, batch_size, device)
            shared = {k: noised[k] for k in self.SHARED_INPUT_KEYS if noised.get(k) is not None}
            with self.autocast():
                pred = self.adapter.denoise(model, **shared, **cond)
            loss = mse_loss(pred, noised["target"], mask=noised.get("loss_mask"))
            metrics[f"val/loss@{sigma:g}"] = loss.item()
        metrics["val/loss"] = sum(metrics.values()) / len(self.val_sigmas)
        return metrics
