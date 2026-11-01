"""Tests for BiSFTRecipe config wiring (scheduler knobs reach the scheduler).

These guard the build path the trainer uses (``minwm.config.lazy.build`` ->
``BiSFTRecipe(**cfg)``): the Wan phase0 config passes ``sigma_min`` /
``extra_one_step`` at the recipe level, so BiSFTRecipe must accept and forward
them to its FlowMatchingScheduler through the shared flow base.
"""

import torch

from minwm.config.lazy import build
from minwm.engine.training.recipes.bi_sft import BiSFTRecipe
from minwm.sampling.schedulers import FlowMatchingScheduler


def _recipe_cfg(**overrides) -> dict:
    cfg = dict(
        type="minwm.engine.training.recipes.bi_sft:BiSFTRecipe",
        loss=dict(type="minwm.engine.training.losses:FlowMatchingLoss"),
        timestep_shift=5.0,
    )
    cfg.update(overrides)
    return cfg


class TestFlowRecipeSchedulerWiring:
    def test_builds_with_legacy_schedule_knobs(self):
        recipe = build(_recipe_cfg(sigma_min=0.0, extra_one_step=True))
        assert isinstance(recipe, BiSFTRecipe)
        assert recipe.scheduler.sigma_min == 0.0
        assert recipe.scheduler.extra_one_step is True

    def test_defaults_match_bare_scheduler(self):
        recipe = build(_recipe_cfg())
        bare = FlowMatchingScheduler()
        assert recipe.scheduler.sigma_min == bare.sigma_min
        assert recipe.scheduler.extra_one_step == bare.extra_one_step

    def test_legacy_knobs_change_sigma_table(self):
        legacy = build(_recipe_cfg(sigma_min=0.0, extra_one_step=True)).scheduler
        default = build(_recipe_cfg()).scheduler
        assert not torch.equal(legacy.sigmas, default.sigmas)


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


class TestGradNormGuard:
    """Two-threshold clip/skip behaviour of ``Recipe.optimizer_step`` (single proc)."""

    def _recipe(self, **kw):
        return build(_recipe_cfg(**kw))

    def test_skip_below_scale_threshold_raises(self):
        import pytest

        with pytest.raises(ValueError):
            self._recipe(max_grad_norm=10.0, skip_grad_norm=1.0)

    def _step_with_grad(self, recipe, grad_scale: float) -> bool:
        model = _Tiny()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.lin.weight.detach().clone()
        # Forge a known-norm gradient by backward on a scaled sum.
        out = model(torch.ones(1, 4))
        loss = out.sum() * grad_scale
        stepped = recipe.optimizer_step(loss, opt, model.parameters())
        moved = not torch.equal(before, model.lin.weight.detach())
        assert stepped == moved  # stepped iff weights changed
        return stepped

    def test_huge_grad_skips_step(self):
        recipe = self._recipe(max_grad_norm=1.0, skip_grad_norm=10.0)
        # grad_scale=1e6 -> norm far above 10 -> skip, weights unchanged.
        assert self._step_with_grad(recipe, 1e6) is False

    def test_small_grad_steps(self):
        recipe = self._recipe(max_grad_norm=1.0, skip_grad_norm=10.0)
        assert self._step_with_grad(recipe, 1e-3) is True

    def test_skip_disabled_always_steps(self):
        recipe = self._recipe(max_grad_norm=1.0)  # no skip threshold
        assert recipe.skip_grad_norm is None
        assert self._step_with_grad(recipe, 1e6) is True


class _IdentityAdapter:
    """Adapter stub: the model's output (identity of ``noisy``) is the flow prediction."""

    dtype = torch.float32

    def conditioning(self, batch, batch_size, device):
        return {}

    def denoise(self, model, *, noisy, timestep, clean=None, aug_t=None, viewmats=None, Ks=None):
        return model(noisy)


class _Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return x


class TestARTFValidation:
    def _recipe(self, disparity_scale=None, **kw):
        from minwm.engine.training.losses import FlowMatchingLoss
        from minwm.engine.training.recipes.ar_tf import ARTFRecipe

        chain = [
            dict(type="FlowNoise", uniform_across_frames=False, num_frame_per_block=2),
            dict(type="CleanContextNoiseAug", max_timestep=0),
        ]
        if disparity_scale is not None:
            chain.insert(0, dict(type="DisparityChannel", scale=disparity_scale))
        return ARTFRecipe(
            loss=FlowMatchingLoss(), adapter=_IdentityAdapter(), batch_preprocessors=chain, **kw
        )

    def test_fixed_sigmas_and_seeded_noise(self):
        import pytest

        from minwm.engine.training.losses import mse_loss

        recipe = self._recipe(val_sigmas=(0.3, 0.9), val_seed=5)
        model = _Identity()
        clean = torch.randn(1, 4, 3, 2, 2)
        out = recipe.validation_step(model, {"clean_latent": clean}, index=2)

        assert set(out) == {"val/loss_rgb@0.3", "val/loss_rgb@0.9", "val/loss_rgb"}
        assert out == recipe.validation_step(model, {"clean_latent": clean}, index=2)
        assert out != recipe.validation_step(model, {"clean_latent": clean}, index=3)
        noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(5 + 2))
        sigma = recipe.scheduler.sigma_for(torch.tensor(300.0))
        expected = mse_loss((1 - sigma) * clean + sigma * noise, noise - clean).item()
        assert out["val/loss_rgb@0.3"] == pytest.approx(expected)
        assert out["val/loss_rgb"] == pytest.approx(
            (out["val/loss_rgb@0.3"] + out["val/loss_rgb@0.9"]) / 2
        )

    def test_disparity_is_scored_separately(self):
        """With DisparityChannel: latents and disparity split, plus the x0 error in d units."""
        import pytest

        from minwm.engine.training.losses import mse_loss

        recipe = self._recipe(disparity_scale=2.0, val_sigmas=(0.5,), val_seed=0)
        clean = torch.randn(1, 4, 3, 2, 2)
        disparity = torch.rand(1, 4, 1, 2, 2)
        out = recipe.validation_step(
            _Identity(), {"clean_latent": clean, "disparity": disparity}, index=0
        )

        names = ("val/loss_rgb", "val/loss_disp", "val/disp_err")
        assert set(out) == {f"{n}@0.5" for n in names} | set(names)
        full = torch.cat([clean, 2.0 * (2 * disparity - 1)], dim=2)
        noise = torch.randn(full.shape, generator=torch.Generator().manual_seed(0))
        sigma = recipe.scheduler.sigma_for(torch.tensor(500.0))
        noisy = (1 - sigma) * full + sigma * noise  # the identity model predicts `noisy`
        target = noise - full
        assert out["val/loss_rgb@0.5"] == pytest.approx(
            mse_loss(noisy[:, :, :3], target[:, :, :3]).item()
        )
        assert out["val/loss_disp@0.5"] == pytest.approx(
            mse_loss(noisy[:, :, 3:], target[:, :, 3:]).item()
        )
        d_hat = ((noisy - sigma * noisy)[:, :, 3:] / 2.0 + 1) / 2  # x0 = x_t - σ·v, unscaled
        assert out["val/disp_err@0.5"] == pytest.approx((d_hat - disparity).abs().mean().item())

    def test_rejects_sigma_outside_unit_interval(self):
        import pytest

        with pytest.raises(ValueError, match="val_sigmas"):
            self._recipe(val_sigmas=(0.0, 0.5))
