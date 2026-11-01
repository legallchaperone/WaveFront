"""Loss computation for flow-matching stages: the pure MSE op + config components.

``mse_loss`` is the stateless weighted/masked MSE shared by every
stage. A :class:`BasicLoss` wraps it into the config-buildable
``(pred, batch, scheduler) -> loss`` seam used by the flow recipes; configs pick
one via ``type``::

    recipe:
      type: BiSFTRecipe
      loss:
        type: ODERegressionLoss
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from minwm.engine.events import get_event_storage, has_event_storage
from minwm.sampling.schedulers import FlowMatchingScheduler

__all__ = [
    "mse_loss",
    "BasicLoss",
    "FlowMatchingLoss",
    "JointFlowMatchingLoss",
    "ODERegressionLoss",
]


def mse_loss(
    pred: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    mask: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Weighted, optionally masked MSE between a prediction and its target.

    Args:
        pred (Tensor): model prediction (velocity or ``x0``).
        target (Tensor): ground-truth target, same shape as ``pred``.
        weight (Tensor, optional): per-element or per-sample weight, broadcastable.
        mask (Tensor, optional): boolean mask, ``True`` = include in the loss.
        reduction (str): ``"mean"`` (scalar) or ``"none"`` (per-element).

    Returns:
        Tensor: scalar loss when ``reduction="mean"``, else the per-element loss.
    """
    loss = (pred.float() - target.float()).pow(2)
    if weight is not None:
        loss = loss * weight
    if mask is not None:
        mask = mask.to(device=loss.device, dtype=torch.bool)
        if mask.shape != loss.shape:
            # Left-align a leading-dims mask (e.g. per-frame ``[B, F]``) against
            # the ``[B, F, C, H, W]`` loss by padding trailing singleton dims,
            # then broadcast. ``torch.broadcast_to`` alone is right-aligned and
            # would try to match ``F`` against ``W``.
            if mask.ndim < loss.ndim:
                mask = mask.reshape(mask.shape + (1,) * (loss.ndim - mask.ndim))
            mask = torch.broadcast_to(mask, loss.shape)
        loss = loss[mask]
    if reduction == "mean":
        return loss.mean()
    return loss


class BasicLoss(ABC):
    """Map a model's flow prediction + the preprocessed batch to a scalar loss."""

    @abstractmethod
    def __call__(self, pred: Tensor, batch: dict, scheduler: FlowMatchingScheduler) -> Tensor:
        """Compute the scalar training loss.

        Args:
            pred (Tensor): the model's flow (velocity) prediction ``[B, F, C, H, W]``.
            batch (dict): the preprocessed batch (``target``, ``weight``, ``noisy``,
                ``timestep``, ... — keys depend on the preprocessor chain).
            scheduler (FlowMatchingScheduler): the recipe's flow-match scheduler, for
                losses that need a flow → ``x0`` conversion.

        Returns:
            Tensor: scalar loss to backpropagate.
        """
        raise NotImplementedError


class FlowMatchingLoss(BasicLoss):
    """Weighted velocity MSE: ``mse_loss(pred, target, weight)``.

    Used by bidirectional SFT and AR diffusion teacher forcing.
    """

    def __call__(self, pred: Tensor, batch: dict, scheduler: FlowMatchingScheduler) -> Tensor:
        return mse_loss(
            pred,
            batch["target"],
            weight=batch.get("weight"),
            mask=batch.get("loss_mask"),
        )


class JointFlowMatchingLoss(BasicLoss):
    """Weighted velocity MSE on the latents plus ``disp_weight`` times the same on disparity.

    For ``use_disparity`` models, whose last channel is disparity. Each term is a
    mean over its own elements, so at ``disp_weight=1`` a token's latents (64 values
    per token at Wan's 16 channels and 2x2 patches) and its disparity (4 values)
    count equally. The unweighted terms are also recorded as ``loss_rgb`` /
    ``loss_disp`` on the active :class:`~minwm.engine.events.EventStorage`.

    Args:
        disp_weight (float): weight λ of the disparity term.
        latent_channels (int): latent channels before the disparity channel.

    Raises:
        ValueError: if the prediction does not have ``latent_channels + 1`` channels.
    """

    def __init__(self, disp_weight: float = 1.0, latent_channels: int = 16) -> None:
        self.disp_weight = disp_weight
        self.latent_channels = latent_channels

    def __call__(self, pred: Tensor, batch: dict, scheduler: FlowMatchingScheduler) -> Tensor:
        if pred.shape[2] != self.latent_channels + 1:
            raise ValueError(
                f"expected {self.latent_channels} latent + 1 disparity channels, "
                f"got {pred.shape[2]}"
            )
        target, c = batch["target"], self.latent_channels
        kwargs = dict(weight=batch.get("weight"), mask=batch.get("loss_mask"))
        loss_rgb = mse_loss(pred[:, :, :c], target[:, :, :c], **kwargs)
        loss_disp = mse_loss(pred[:, :, c:], target[:, :, c:], **kwargs)
        if has_event_storage():
            get_event_storage().put_scalars(loss_rgb=loss_rgb.item(), loss_disp=loss_disp.item())
        return loss_rgb + self.disp_weight * loss_disp


class ODERegressionLoss(BasicLoss):
    """ODE regression in ``x0`` space: convert the flow prediction to ``x0`` and
    regress against the trajectory's near-clean target, masking already-clean frames.
    """

    def __call__(self, pred: Tensor, batch: dict, scheduler: FlowMatchingScheduler) -> Tensor:
        timestep = batch["timestep"]
        x0_pred = scheduler.predict_x0(batch["noisy"], pred, timestep)
        mask = timestep != 0
        return mse_loss(x0_pred, batch["target"], mask=mask)
