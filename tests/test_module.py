import pytest

import torch

from deepab_pytorch.deepab_pytorch import (
    Symmetrization,
    ResBlock1D,
    ResBlock2D,
    ResNet1D,
    ResNet2D,
    CrissCrossAttention,
    RecurrentCrissCrossAttention,
    DeepAb,
)


def test_bidirectional_single_layer_lstm_final_hidden_state():
    bsz, max_len, dim, d_hidden = 3, 4, 8, 16
    x = torch.randn(bsz, max_len, dim)

    bilstm = torch.nn.LSTM(
        input_size=dim,
        hidden_size=d_hidden,
        num_layers=1,
        bidirectional=True,
        batch_first=True,
    )
    lstm = torch.nn.LSTM(
        input_size=dim,
        hidden_size=d_hidden,
        num_layers=1,
        bidirectional=False,
        batch_first=True,
    )
    rev_lstm = torch.nn.LSTM(
        input_size=dim,
        hidden_size=d_hidden,
        num_layers=1,
        bidirectional=False,
        batch_first=True,
    )

    lstm.weight_ih_l0.data = bilstm.weight_ih_l0.data
    lstm.weight_hh_l0.data = bilstm.weight_hh_l0.data
    lstm.bias_ih_l0.data = bilstm.bias_ih_l0.data
    lstm.bias_hh_l0.data = bilstm.bias_hh_l0.data

    rev_lstm.weight_ih_l0.data = bilstm.weight_ih_l0_reverse.data
    rev_lstm.weight_hh_l0.data = bilstm.weight_hh_l0_reverse.data
    rev_lstm.bias_ih_l0.data = bilstm.bias_ih_l0_reverse.data
    rev_lstm.bias_hh_l0.data = bilstm.bias_hh_l0_reverse.data

    # outputs
    out_bi, (h_bi, c_bi) = bilstm(x)
    assert out_bi.shape == (bsz, max_len, d_hidden * 2)

    out_fwd, (h_fwd, c_fwd) = lstm(x)
    assert out_fwd.shape == (bsz, max_len, d_hidden)

    out_rev, (h_rev, c_rev) = rev_lstm(x.flip(1))
    assert out_rev.shape == (bsz, max_len, d_hidden)

    out_fwdrev = torch.cat([out_fwd, out_rev.flip(1)], dim=-1)
    assert torch.allclose(out_bi - out_fwdrev, torch.zeros_like(out_bi))

    # hidden states
    assert h_bi.shape == (2, bsz, d_hidden)
    assert torch.allclose(
        h_bi.permute(1, 0, 2),
        torch.cat(
            [out_bi[:, -1, :d_hidden].unsqueeze(1), out_bi[:, 0, d_hidden:].unsqueeze(1)],
            dim=1,
        ),
    )

    assert h_fwd.shape == (1, bsz, d_hidden)
    assert h_rev.shape == (1, bsz, d_hidden)

    assert torch.allclose(h_bi, torch.cat([h_fwd, h_rev], dim=0))


def test_Symmetrization():
    model = Symmetrization()

    b, c, h, w = 2, 3, 8, 8
    x = torch.randn(b, c, h, w)

    out = model(x)
    assert out.shape == (b, c, h, w)
    assert (out == out.transpose(-1, -2)).all()


def test_ResBlock1D():
    channel, kernel_size, stride = 8, 3, 1
    model = ResBlock1D(channel, kernel_size, stride)

    bsz, length = 2, 8
    x = torch.randn(bsz, channel, length)

    out = model(x)
    assert out.shape == (bsz, channel, length)


@pytest.mark.parametrize("dilation", [1, 2, 4, 8, 16])
def test_ResBlock2D(dilation):
    channel, kernel_size, stride = 8, 3, 1
    model = ResBlock2D(channel, kernel_size, stride, dilation=1)

    bsz, w, h = 2, 128, 128
    x = torch.randn(bsz, channel, w, h)

    out = model(x)
    assert out.shape == (bsz, channel, w, h)


def test_ResNet1D():
    in_channel, out_channel, kernel_size = 16, 8, 3
    n_blocks = 3
    model = ResNet1D(in_channel, out_channel, kernel_size, n_blocks)

    bsz, length = 2, 8
    x = torch.randn(bsz, in_channel, length)

    out = model(x)
    assert out.shape == (bsz, out_channel, length)


def test_ResNet2D():
    in_channel, out_channel, kernel_size = 16, 8, 3
    n_blocks = 3
    model = ResNet2D(in_channel, out_channel, kernel_size, n_blocks, dilation=[1, 2, 4, 8, 16])

    bsz, w, h = 2, 8, 8
    x = torch.randn(bsz, in_channel, w, h)

    out = model(x)
    assert out.shape == (bsz, out_channel, w, h)


def test_CrissCrossAttention():
    channel = 256
    model = CrissCrossAttention(channel)

    b, c, h, w = 2, 256, 32, 32
    x = torch.randn(b, c, h, w)

    assert model(x).shape == (b, c, h, w)


def test_RecurrentCrissCrossAttention():
    in_channel, inner_channel = 256, 256
    model = RecurrentCrissCrossAttention(in_channel, inner_channel)

    b, c, h, w = 2, 256, 32, 32
    x = torch.randn(b, c, h, w)

    assert model(x).shape == (b, c, h, w)
