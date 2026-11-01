"""Tests for the causal Wan ``use_disparity`` channel (baseline ②, joint depth)."""

import pytest
import torch
from torch.nn.attention.flex_attention import flex_attention

import minwm.modeling.wan21.blocks as wan_blocks
from minwm.modeling.wan21.adapter import Wan21Adapter
from minwm.modeling.wan21.causal import CausalWan21Model

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
    disp = CausalWan21Model(**_ARCH, use_disparity=True).eval()
    missing, unexpected = disp.load_state_dict(base.state_dict(), strict=False)
    assert sorted(missing) == _DISP_KEYS and unexpected == []
    return base, disp


def _with_disparity(rgb):
    return torch.cat([rgb, torch.randn(1, *rgb.shape[1:])])


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
    assert out_disp.shape[1] == 17
    assert float(out_base.norm()) > 0
    torch.testing.assert_close(out_disp[:, :16], out_base)
    assert bool((out_disp[:, 16] == 0).all())


class TestConstruction:
    def test_disabled_by_default(self):
        model = CausalWan21Model(**_ARCH)
        assert model.disp_embed is None and model.head.disp_head is None
        assert not any("disp" in key for key in model.state_dict())

    def test_enabled_branch_is_zero_initialised_and_in_config(self):
        model = CausalWan21Model(**_ARCH, use_disparity=True)
        params = dict(model.named_parameters())
        assert all(bool((params[key] == 0).all()) for key in _DISP_KEYS)
        assert model.config.use_disparity is True

    def test_rejects_i2v(self):
        with pytest.raises(ValueError, match="t2v"):
            CausalWan21Model(**{**_ARCH, "model_type": "i2v"}, use_disparity=True)


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
            x[16] += 1.0
        reference = _teacher_forcing(disp, noisy, clean)
        assert not torch.allclose(_teacher_forcing(disp, shifted[0], clean), reference)
        assert not torch.allclose(_teacher_forcing(disp, noisy, shifted[1]), reference)


class TestOutputLayout:
    def test_disparity_fills_each_tokens_2x2_pixels_in_order(self):
        """disp_head's 4 outputs per token land on that token's pixels, row-major."""
        base, disp = _base_and_disp()
        with torch.no_grad():
            disp.head.disp_head.bias.copy_(torch.arange(4.0))
        tokens = torch.randn(1, _F * _FRAME_SEQLEN, 64)
        e = torch.randn(1, _F, 1, 64)
        grid = torch.tensor([[_F, _H // 2, _W // 2]])
        with torch.no_grad():
            rgb = base.unpatchify(base.head(tokens, e), grid)[0]
            out = disp.unpatchify(disp.head(tokens, e), grid)[0]
        torch.testing.assert_close(out[:16], rgb)
        disparity = out[16]
        for value, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            assert bool((disparity[:, row::2, col::2] == value).all())


class TestCheckpoint:
    def test_disparity_weights_load_strictly_only_with_use_disparity(self):
        state = CausalWan21Model(**_ARCH, use_disparity=True).state_dict()
        CausalWan21Model(**_ARCH, use_disparity=True).load_state_dict(state, strict=True)
        with pytest.raises(RuntimeError, match="disp_embed"):
            CausalWan21Model(**_ARCH).load_state_dict(state, strict=True)

    def test_stage1_checkpoint_loads_with_a_zero_disparity_branch(self, tmp_path):
        """The trainer's strict pretrained loader accepts a checkpoint without the branch."""
        from minwm.engine.checkpoint import load_model_weights

        base = CausalWan21Model(**_ARCH)
        with torch.no_grad():
            base.head.head.weight.add_(0.1)
        torch.save({"model": base.state_dict()}, tmp_path / "stage1.pt")
        disp = CausalWan21Model(**_ARCH, use_disparity=True)
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
            load_model_weights(
                CausalWan21Model(**_ARCH, use_disparity=True), str(tmp_path / "broken.pt")
            )
