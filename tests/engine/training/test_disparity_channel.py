"""Tests for ``DisparityChannel``: dense disparity joins the latent as 16 channels (baseline ②)."""

import pytest
import torch
from torch.utils.data import default_collate

from minwm.data.datasets.mock import MockCameraLatentDataset
from minwm.processors import (
    DISP_CHANNELS,
    DISP_UNSHUFFLE,
    DisparityChannel,
    disparity_from_channels,
    disparity_to_channels,
)

_CPU = torch.device("cpu")


def test_constants():
    assert (DISP_UNSHUFFLE, DISP_CHANNELS) == (4, 16)


def test_appends_scaled_channels_and_keeps_latents():
    clean = torch.randn(1, 3, 16, 2, 2)
    disparity = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 3, 1, 1, 1).expand(1, 3, 1, 8, 8)
    out = DisparityChannel(scale=2.0)({"clean_latent": clean, "disparity": disparity}, _CPU)

    latent = out["clean_latent"]
    assert latent.shape == (1, 3, 32, 2, 2)
    assert torch.equal(latent[:, :, :16], clean)
    expected = torch.tensor([-2.0, 0.0, 2.0]).reshape(1, 3, 1, 1, 1).expand(1, 3, 16, 2, 2)
    assert torch.equal(latent[:, :, 16:], expected)


def test_channel_layout_per_latent_cell():
    """Pixel (4h + i, 4w + j) lands at channel i * 4 + j of cell (h, w)."""
    batch, frames, height, width = 2, 3, 2, 3
    disparity = torch.rand(batch, frames, 1, 4 * height, 4 * width)
    out = DisparityChannel(scale=1.0)(
        {"clean_latent": torch.zeros(batch, frames, 16, height, width), "disparity": disparity},
        _CPU,
    )["clean_latent"][:, :, 16:]

    mapped = 2 * disparity - 1
    for h in range(height):
        for w in range(width):
            for i in range(4):
                for j in range(4):
                    assert torch.equal(
                        out[:, :, i * 4 + j, h, w], mapped[:, :, 0, 4 * h + i, 4 * w + j]
                    ), (h, w, i, j)


def test_round_trip_is_bit_exact():
    disparity = torch.rand(2, 3, 1, 8, 12)
    channels = disparity_to_channels(disparity)
    assert channels.shape == (2, 3, 16, 2, 3)
    assert torch.equal(disparity_from_channels(channels), disparity)


def test_from_channels_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="16 disparity channels"):
        disparity_from_channels(torch.rand(1, 2, 1, 2, 2))


def test_follows_the_latent_dtype():
    clean = torch.randn(1, 2, 16, 2, 2, dtype=torch.bfloat16)
    batch = {"clean_latent": clean, "disparity": torch.rand(1, 2, 1, 8, 8)}
    assert DisparityChannel(scale=1.0)(batch, _CPU)["clean_latent"].dtype == torch.bfloat16


@pytest.mark.parametrize(
    "disparity, match",
    [
        (torch.rand(1, 2, 8, 8), "must be"),  # missing the channel dim
        (torch.rand(1, 2, 1, 2, 2), "must be"),  # on the latent grid, not dense
        (torch.full((1, 2, 1, 8, 8), 1.5), r"\[0, 1\]"),  # not normalized
        (torch.full((1, 2, 1, 8, 8), float("nan")), r"\[0, 1\]"),  # NaN slips past min/max
    ],
)
def test_rejects_bad_disparity(disparity, match):
    batch = {"clean_latent": torch.randn(1, 2, 16, 2, 2), "disparity": disparity}
    with pytest.raises(ValueError, match=match):
        DisparityChannel(scale=1.0)(batch, _CPU)


def test_scale_is_bound_from_the_model():
    channel = DisparityChannel()
    batch = {"clean_latent": torch.randn(1, 2, 16, 2, 2), "disparity": torch.rand(1, 2, 1, 8, 8)}
    with pytest.raises(ValueError, match="unresolved"):
        channel(batch, _CPU)
    channel.resolve_scale(0.5)
    assert channel.scale == 0.5
    channel.resolve_scale(0.5)  # binding again to the same value is fine
    assert channel(batch, _CPU)["clean_latent"].shape == (1, 2, 32, 2, 2)


def test_scale_must_match_the_model():
    with pytest.raises(ValueError, match="differs"):
        DisparityChannel(scale=1.0).resolve_scale(0.5)
    with pytest.raises(ValueError, match="disparity_scale set"):
        DisparityChannel().resolve_scale(None)


def test_mock_dataset_emits_dense_disparity():
    sample = MockCameraLatentDataset(
        length=1, num_frames=2, height=3, width=5, with_disparity=True
    )[0]
    assert sample["disparity"].shape == (2, 1, 12, 20)
    assert 0 <= sample["disparity"].min() and sample["disparity"].max() <= 1


def test_stage1_chain_carries_disparity_through_noise_and_context():
    """A mock clip through the Stage 1 chain plus DisparityChannel gives 32-channel tensors."""
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
        assert out[key].shape == (1, 4, 32, 4, 4), key
    torch.testing.assert_close(disparity_from_channels(out["clean"][:, :, 16:]), 2 * disparity - 1)
