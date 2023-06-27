import pytest

import torch


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
