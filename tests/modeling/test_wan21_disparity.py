"""Tests for the causal Wan ``use_disparity`` channels (baseline ②, joint depth)."""

import pytest
import torch
from torch.nn.attention.flex_attention import flex_attention

import minwm.modeling.wan21.blocks as wan_blocks
from minwm.modeling.wan21.adapter import Wan21Adapter
from minwm.modeling.wan21.causal import CausalWan21Model
from minwm.processors import DISP_CHANNELS, DisparityChannel, disparity_from_channels

_ARCH = dict(
    model_type="t2v",
    dim=64,
    ffn_dim=128,
    freq_dim=64,
    text_len=8,
    text_dim=64,
    num_heads=4,
    num_layers=2,
    in_dim=16,
    out_dim=16,
    num_frame_per_block=2,
)
_DISP = dict(use_disparity=True, disparity_scale=0.5)
_F, _H, _W = 4, 8, 8
_FRAME_SEQLEN = (_H // 2) * (_W // 2)
_DISP_KEYS = [
    "disp_embed.bias",
    "disp_embed.weight",
    "head.disp_head.bias",
    "head.disp_head.weight",
]


def _base_and_disp():
    """A base model with a non-zero head, and a use_disparity model holding its weights."""
    torch.manual_seed(0)
    base = CausalWan21Model(**_ARCH).eval()
    with torch.no_grad():
        base.head.head.weight.add_(0.1)  # the zero-init head would make every output 0
    disp = CausalWan21Model(**_ARCH, **_DISP).eval()
    missing, unexpected = disp.load_state_dict(base.state_dict(), strict=False)
    assert sorted(missing) == _DISP_KEYS and unexpected == []
    return base, disp


def _with_disparity(rgb):
    return torch.cat([rgb, torch.randn(DISP_CHANNELS, *rgb.shape[1:])])


@pytest.fixture
def eager_flex(monkeypatch):
    # torch.compile'd flex attention has no CPU backend on every platform; eager is
    # exact and fast enough at this size.
    monkeypatch.setattr(wan_blocks, "_get_flex_attention", lambda: flex_attention)


def _teacher_forcing(model, x, clean_x):
    """Training forward on a fixed timestep / context (seeded, so calls are comparable)."""
    torch.manual_seed(2)
    with torch.no_grad():
        return model(
            x=[x],
            clean_x=[clean_x],
            t=torch.full((1, _F), 500.0),
            context=[torch.randn(8, 64)],
            seq_len=_F * _FRAME_SEQLEN,
        )


def _assert_rgb_unchanged_and_disparity_zero(out_base, out_disp):
    assert out_disp.shape[1] == 16 + DISP_CHANNELS
    assert float(out_base.norm()) > 0
    torch.testing.assert_close(out_disp[:, :16], out_base)
    assert bool((out_disp[:, 16:] == 0).all())


class TestConstruction:
    def test_disabled_by_default(self):
        model = CausalWan21Model(**_ARCH)
        assert model.disp_embed is None and model.head.disp_head is None
        assert not any("disp" in key for key in model.state_dict())

    def test_enabled_branch_is_zero_initialised_and_in_config(self):
        model = CausalWan21Model(**_ARCH, **_DISP)
        params = dict(model.named_parameters())
        assert all(bool((params[key] == 0).all()) for key in _DISP_KEYS)
        assert model.config.use_disparity is True
        assert model.config.disparity_scale == 0.5 and model.disparity_scale == 0.5

    def test_requires_disparity_scale(self):
        with pytest.raises(ValueError, match="disparity_scale"):
            CausalWan21Model(**_ARCH, use_disparity=True)

    def test_rejects_i2v(self):
        with pytest.raises(ValueError, match="t2v"):
            CausalWan21Model(**{**_ARCH, "model_type": "i2v"}, **_DISP)


class TestZeroInitIsNoop:
    """Loading weights without the disparity branch must leave the latent output unchanged."""

    def test_teacher_forcing_forward(self, eager_flex):
        base, disp = _base_and_disp()
        torch.manual_seed(1)
        rgb, clean_rgb = torch.randn(16, _F, _H, _W), torch.randn(16, _F, _H, _W)
        out_base = _teacher_forcing(base, rgb, clean_rgb)
        out_disp = _teacher_forcing(disp, _with_disparity(rgb), _with_disparity(clean_rgb))
        _assert_rgb_unchanged_and_disparity_zero(out_base, out_disp)

    def test_kv_cache_inference_forward(self):
        base, disp = _base_and_disp()
        torch.manual_seed(1)
        chunk = torch.randn(16, 2, _H, _W)  # the first 2-frame AR block
        context = [torch.randn(8, 64)]
        adapter = Wan21Adapter(causal=True, text_len=8, text_dim=64, dtype="float32")
        outs = []
        for model, x in ((base, chunk), (disp, _with_disparity(chunk))):
            cache = adapter.rollout_init_cache(
                model,
                batch_size=1,
                num_frames=_F,
                frame_seqlen=_FRAME_SEQLEN,
                seq_len=_F * _FRAME_SEQLEN,
                dtype=torch.float32,
                device=torch.device("cpu"),
                cond={},
            )
            with torch.no_grad():
                outs.append(
                    model(
                        x=[x],
                        t=torch.full((1, 2), 500.0),
                        context=context,
                        seq_len=_F * _FRAME_SEQLEN,
                        kv_cache=cache["kv_cache"],
                        crossattn_cache=cache["crossattn_cache"],
                    )
                )
        _assert_rgb_unchanged_and_disparity_zero(*outs)


class TestDisparityInputIsUsed:
    def test_noisy_and_clean_disparity_both_reach_the_output(self, eager_flex):
        """With a trained (non-zero) disp_embed, both halves' disparity change the output."""
        _, disp = _base_and_disp()
        with torch.no_grad():
            disp.disp_embed.weight.normal_(std=0.1)
        torch.manual_seed(1)
        noisy = _with_disparity(torch.randn(16, _F, _H, _W))
        clean = _with_disparity(torch.randn(16, _F, _H, _W))
        shifted = [x.clone() for x in (noisy, clean)]
        for x in shifted:
            x[16:] += 1.0
        reference = _teacher_forcing(disp, noisy, clean)
        assert not torch.allclose(_teacher_forcing(disp, shifted[0], clean), reference)
        assert not torch.allclose(_teacher_forcing(disp, noisy, shifted[1]), reference)


def _dense_to_model_input(disparity):
    """Dense ``[F, 1, 4H, 4W]`` -> ``[16 + DISP_CHANNELS, F, H, W]`` via DisparityChannel."""
    frames, _, height, width = disparity.shape
    clean = torch.zeros(1, frames, 16, height // 4, width // 4)
    batch = DisparityChannel(scale=1.0)(
        {"clean_latent": clean, "disparity": disparity[None]}, torch.device("cpu")
    )
    return batch["clean_latent"][0].permute(1, 0, 2, 3)


def _model_output_to_dense(out):
    """``[16 + DISP_CHANNELS, F, H, W]`` model output -> dense ``[F, 1, 4H, 4W]``."""
    return disparity_from_channels(out[16:].permute(1, 0, 2, 3))


class TestOutputLayout:
    def test_shapes(self):
        model = CausalWan21Model(**_ARCH, **_DISP)
        assert model.disp_embed.weight.shape == (64, DISP_CHANNELS, 1, 2, 2)
        assert model.head.disp_head.weight.shape == (4 * DISP_CHANNELS, 64)

    def test_disparity_output_is_trailing_channels_per_latent_cell(self, eager_flex):
        """disp_head output ``p * 16 + k`` lands at channel ``16 + k`` of the token's pixel ``p``.

        Full path: dense disparity -> DisparityChannel -> forward -> disparity_from_channels.
        With ``disp_head`` reduced to a hand-set bias, the dense output is a known pattern.
        """
        base, disp = _base_and_disp()
        with torch.no_grad():
            disp.head.disp_head.bias.copy_(torch.arange(4.0 * DISP_CHANNELS))
        torch.manual_seed(1)
        x = _dense_to_model_input(torch.rand(_F, 1, 4 * _H, 4 * _W))
        clean_x = _dense_to_model_input(torch.rand(_F, 1, 4 * _H, 4 * _W))
        out = _teacher_forcing(disp, x, clean_x)[0]
        torch.testing.assert_close(out[:16], _teacher_forcing(base, x[:16], clean_x[:16])[0])

        rows = torch.arange(4 * _H)[:, None]
        cols = torch.arange(4 * _W)[None, :]
        patch = (rows // 4 % 2) * 2 + cols // 4 % 2
        channel = (rows % 4) * 4 + cols % 4
        expected = (patch * DISP_CHANNELS + channel).float().expand(_F, 1, -1, -1)
        assert torch.equal(_model_output_to_dense(out), expected)

    def test_embed_and_head_share_the_channel_layout(self):
        """With disp_embed one-hot and disp_head the identity, the head returns the input."""
        _, disp = _base_and_disp()
        patch, dim = 4, _ARCH["dim"]
        with torch.no_grad():
            disp.patch_embedding.bias.zero_()
            disp.head.modulation.zero_()
            weight = torch.zeros_like(disp.disp_embed.weight)
            for p in range(patch):
                for k in range(DISP_CHANNELS):
                    weight[p * DISP_CHANNELS + k, k, 0, p // 2, p % 2] = 1.0
            disp.disp_embed.weight.copy_(weight)
            disp.head.disp_head.weight.copy_(torch.eye(dim))
        disp.head.norm = torch.nn.Identity()

        frames, height, width = 2, 4, 6
        dense = torch.rand(frames, 1, 4 * height, 4 * width)
        x = _dense_to_model_input(dense)
        with torch.no_grad():
            tokens = disp._patch_embed(x[None]).flatten(2).transpose(1, 2)
            out = disp.unpatchify(
                disp.head(tokens, torch.zeros(1, frames, 1, dim)),
                torch.tensor([[frames, height // 2, width // 2]]),
            )[0]
        assert torch.equal(out[16:], x[16:])
        assert torch.equal(_model_output_to_dense(out), 2 * dense - 1)


class TestCheckpoint:
    def test_disparity_weights_load_strictly_only_with_use_disparity(self):
        state = CausalWan21Model(**_ARCH, **_DISP).state_dict()
        CausalWan21Model(**_ARCH, **_DISP).load_state_dict(state, strict=True)
        with pytest.raises(RuntimeError, match="disp_embed"):
            CausalWan21Model(**_ARCH).load_state_dict(state, strict=True)

    def test_stage1_checkpoint_loads_with_a_zero_disparity_branch(self, tmp_path):
        """The trainer's strict pretrained loader accepts a checkpoint without the branch."""
        from minwm.engine.checkpoint import load_model_weights

        base = CausalWan21Model(**_ARCH)
        with torch.no_grad():
            base.head.head.weight.add_(0.1)
        torch.save({"model": base.state_dict()}, tmp_path / "stage1.pt")
        disp = CausalWan21Model(**_ARCH, **_DISP)
        with torch.no_grad():
            disp.disp_embed.weight.add_(1.0)  # must come back as zero from the load

        load_model_weights(disp, str(tmp_path / "stage1.pt"))

        params = dict(disp.named_parameters())
        assert all(bool((params[key] == 0).all()) for key in _DISP_KEYS)
        assert torch.equal(disp.head.head.weight, base.head.head.weight)

    def test_other_missing_keys_still_fail_the_strict_load(self, tmp_path):
        from minwm.engine.checkpoint import load_model_weights

        state = CausalWan21Model(**_ARCH).state_dict()
        del state["head.head.weight"]
        torch.save({"model": state}, tmp_path / "broken.pt")
        with pytest.raises(RuntimeError, match="head.head.weight"):
            load_model_weights(CausalWan21Model(**_ARCH, **_DISP), str(tmp_path / "broken.pt"))
