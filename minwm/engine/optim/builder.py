"""Optimizer construction helpers.

Centralizes the trainable-parameter selection + optimizer wiring that every
single-optimizer recipe repeated inline. A recipe's ``build_optimizers`` becomes
a one-liner over :func:`build_optimizer`; multi-optimizer recipes (e.g. DMD's
generator + critic) call it once per group.

The optimizer class is config-selectable: the ``cfg`` dict's ``name`` key is a
short framework class name such as ``"Muon"``, or an explicit external path
such as the default ``"torch.optim:AdamW"``. The remaining keys are passed
through as constructor kwargs. The class can't live under a ``type`` key in the
recipe node because the optimizer needs the module's parameters injected at build
time, which only exist after the model is constructed — so the lazy config engine
must leave the node as a plain dict rather than instantiating it eagerly.
"""

import torch
from torch import Tensor, nn

from minwm.config.lazy import locate

_DEFAULT_OPTIMIZER = "torch.optim:AdamW"


def trainable_params(module: nn.Module) -> list[Tensor]:
    """Return the module's parameters that require gradients.

    Args:
        module (nn.Module): the module whose parameters are optimized.

    Returns:
        list[Tensor]: parameters with ``requires_grad=True``, in module order.
    """
    return [p for p in module.parameters() if p.requires_grad]


def build_optimizer(module: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Build a config-selected optimizer over a module's trainable parameters.

    Args:
        module (nn.Module): the module to optimize.
        cfg (dict): optimizer config. The optional ``name`` key is a
            short framework name or ``"module.path:ClassName"`` string
            (default ``torch.optim:AdamW``). The optional ``lr_scales`` key maps a
            parameter-name prefix to a multiplier on ``lr`` (e.g.
            ``{"disp_embed.": 50.0}``); parameters matching a prefix get their own
            group, the first matching prefix wins. Every other key is forwarded as a
            constructor kwarg (e.g. ``lr``, ``betas``, ``weight_decay``).

    Returns:
        torch.optim.Optimizer: optimizer over the module's trainable parameters.

    Raises:
        ValueError: if an ``lr_scales`` prefix matches no trainable parameter.
    """
    cls = locate(cfg.get("name", _DEFAULT_OPTIMIZER))
    kwargs = {k: v for k, v in cfg.items() if k not in ("name", "lr_scales")}
    lr_scales = cfg.get("lr_scales")
    if not lr_scales:
        return cls(trainable_params(module), **kwargs)

    scaled: dict[str, list[Tensor]] = {prefix: [] for prefix in lr_scales}
    rest = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        prefix = next((p for p in lr_scales if name.startswith(p)), None)
        (rest if prefix is None else scaled[prefix]).append(param)
    unmatched = [prefix for prefix, params in scaled.items() if not params]
    if unmatched:
        raise ValueError(f"lr_scales prefixes match no trainable parameter: {unmatched}")
    groups = [{"params": rest}] + [
        {"params": params, "lr": kwargs["lr"] * lr_scales[prefix]}
        for prefix, params in scaled.items()
    ]
    return cls(groups, **kwargs)
