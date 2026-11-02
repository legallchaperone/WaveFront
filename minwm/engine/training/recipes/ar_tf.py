"""Autoregressive teacher-forcing recipe."""

from typing import Any

import torch
from torch import Tensor, nn

from minwm.processors import DISP_CHANNELS, DisparityChannel

from ..losses import mse_loss
from ._flow_base import SingleForwardFlowRecipe


class ARTFRecipe(SingleForwardFlowRecipe):
    """Autoregressive teacher forcing: single causal forward with clean context.

    Validation scores the denoising error on held-out clips at fixed noise levels.
    Each clip is noised to every ``val_sigmas`` level with noise seeded by
    ``val_seed`` plus the clip's index, so the numbers are comparable across steps
    and runs. Every frame gets the same level, and the forward is the training one.
    With a :class:`~minwm.processors.DisparityChannel` in the chain, the trailing
    ``DISP_CHANNELS`` disparity channels are scored separately (velocity MSE, and the
    x0 error in ``d`` units). Its scale is bound to the model's ``disparity_scale``
    before each training and validation step.

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

    def train_one_step(
        self,
        model: nn.Module,
        batch: Any,
        optimizers: dict[str, torch.optim.Optimizer],
        step: int,
        auxiliary_models: dict[str, nn.Module] | None = None,
    ) -> dict[str, float]:
        """Bind the disparity scale to ``model``, then run the single-forward step.

        Args:
            model (nn.Module): the trainer-owned primary model.
            batch (Any): one dataloader batch.
            optimizers (dict[str, torch.optim.Optimizer]): steps ``optimizers["main"]``.
            step (int): current global step.
            auxiliary_models (dict[str, nn.Module], optional): unused.

        Returns:
            dict[str, float]: ``{"loss": <scalar>}``.
        """
        self._bind_disparity_scale(model)
        return super().train_one_step(model, batch, optimizers, step, auxiliary_models)

    def validation_step(self, model: nn.Module, batch: dict, index: int) -> dict[str, float]:
        """Unweighted validation errors at each ``val_sigmas`` level, plus their means.

        Args:
            model (nn.Module): the model under evaluation.
            batch (dict): one collated held-out sample.
            index (int): the sample's index in the val dataset; seeds its noise.

        Returns:
            dict[str, float]: ``val/loss_rgb@<σ>``, the velocity MSE on the latent
            channels; with disparity also ``val/loss_disp@<σ>`` (its velocity MSE) and
            ``val/disp_err@<σ>`` (mean ``|d̂ − d|`` of the x0 estimate, ``d`` in
            ``[0, 1]``). Each metric's mean over the levels is logged without ``@<σ>``.
        """
        self._bind_disparity_scale(model)
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
            for name, value in self._val_errors(pred, noised).items():
                metrics[f"{name}@{sigma:g}"] = value
        for name in {key.split("@")[0] for key in metrics}:
            values = [v for k, v in metrics.items() if k.startswith(f"{name}@")]
            metrics[name] = sum(values) / len(values)
        return metrics

    def _bind_disparity_scale(self, model: nn.Module) -> None:
        if self.batch_preprocessors is None:
            self.batch_preprocessors = self.build_preprocessors()
        for preprocessor in self.batch_preprocessors:
            if isinstance(preprocessor, DisparityChannel):
                preprocessor.resolve_scale(getattr(model, "disparity_scale", None))

    def _val_errors(self, pred: Tensor, noised: dict) -> dict[str, float]:
        target, mask = noised["target"], noised.get("loss_mask")
        disparity = next(
            (p for p in self.batch_preprocessors if isinstance(p, DisparityChannel)), None
        )
        c = target.shape[2] - DISP_CHANNELS if disparity is not None else target.shape[2]
        errors = {"val/loss_rgb": mse_loss(pred[:, :, :c], target[:, :, :c], mask=mask).item()}
        if disparity is None:
            return errors
        errors["val/loss_disp"] = mse_loss(pred[:, :, c:], target[:, :, c:], mask=mask).item()
        x0 = self.scheduler.predict_x0(noised["noisy"].float(), pred.float(), noised["timestep"])
        # The channels hold scale·(2d − 1), so dividing by 2·scale gives the error in d.
        # Pixel unshuffle only rearranges values, so the mean over all disparity
        # channels is the mean over the dense disparity values.
        err = (x0[:, :, c:] - noised["clean_latent"][:, :, c:].float()).abs()
        err = err / (2 * disparity.scale)
        if mask is not None:
            err = err[torch.broadcast_to(mask.bool(), err.shape)]
        errors["val/disp_err"] = err.mean().item()
        return errors
