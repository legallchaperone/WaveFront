"""Tests for ``DisparityChannel``: disparity joins the latent as a 17th channel (baseline ②)."""

import pytest
import torch
from torch.utils.data import default_collate

from minwm.data.datasets.mock import MockCameraLatentDataset
from minwm.processors import DisparityChannel

_CPU = torch.device("cpu")


def test_appends_scaled_channel_and_keeps_latents():
    clean = torch.randn(1, 3, 16, 2, 2)
    disparity = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 3, 1, 1, 1).expand(1, 3, 1, 2, 2)
    out = DisparityChannel(scale=2.0)({"clean_latent": clean, "disparity": disparity}, _CPU)

    latent = out["clean_latent"]
    assert latent.shape == (1, 3, 17, 2, 2)
    assert torch.equal(latent[:, :, :16], clean)
    expected = torch.tensor([-2.0, 0.0, 2.0]).reshape(1, 3, 1, 1).expand(1, 3, 2, 2)
    assert torch.equal(latent[:, :, 16], expected)


def test_follows_the_latent_dtype():
    clean = torch.randn(1, 2, 16, 2, 2, dtype=torch.bfloat16)
    batch = {"clean_latent": clean, "disparity": torch.rand(1, 2, 1, 2, 2)}
    assert DisparityChannel()(batch, _CPU)["clean_latent"].dtype == torch.bfloat16


@pytest.mark.parametrize(
    "disparity, match",
    [
        (torch.rand(1, 2, 2, 2), "must be"),  # missing the channel dim
        (torch.full((1, 2, 1, 2, 2), 1.5), r"\[0, 1\]"),  # not normalized
        (torch.full((1, 2, 1, 2, 2), float("nan")), r"\[0, 1\]"),  # NaN slips past min/max
    ],
)
def test_rejects_bad_disparity(disparity, match):
    batch = {"clean_latent": torch.randn(1, 2, 16, 2, 2), "disparity": disparity}
    with pytest.raises(ValueError, match=match):
        DisparityChannel()(batch, _CPU)


def test_stage1_chain_carries_disparity_through_noise_and_context():
    """A mock clip through the Stage 1 chain plus DisparityChannel gives 17-channel tensors."""
    from minwm.engine.training.losses import JointFlowMatchingLoss
    from minwm.engine.training.recipes.ar_tf import ARTFRecipe
    from minwm.modeling.wan21.adapter import Wan21Adapter

    dataset = MockCameraLatentDataset(
        length=1, num_frames=4, height=4, width=4, with_disparity=True
    )
    batch = default_collate([dataset[0]])
    disparity = batch["disparity"].clone()
    recipe = ARTFRecipe(
        loss=JointFlowMatchingLoss(),
        adapter=Wan21Adapter(causal=True, dtype="float32"),
        batch_preprocessors=[
            dict(type="LatentToDevice"),
            dict(type="DisparityChannel", scale=1.0),
            dict(type="FlowNoise", uniform_across_frames=False, num_frame_per_block=2),
            dict(type="CleanContextNoiseAug", max_timestep=0),
        ],
    )
    out = recipe.preprocess(batch, _CPU)

    for key in ("clean_latent", "noise", "noisy", "target", "clean"):
        assert out[key].shape == (1, 4, 17, 4, 4), key
    torch.testing.assert_close(out["clean"][:, :, 16:], 2 * disparity - 1)
