import torch
import pytest

from deepab_pytorch import DeepAb


def test_DeepAb_init():
    model = DeepAb()
    assert model is not None


def test_DeepAb_shape():
    # smaller parameters
    res1d_out_channel = 32
    res1d_kernel_size = 17
    res1d_n_blocks = 2
    res2d_out_channel = 64
    res2d_kernel_size = 5
    res2d_n_blocks = 3
    n_target_bins = 37
    lm_ckpt = None  # for debug

    model = DeepAb(
        res1d_out_channel=res1d_out_channel,
        res1d_kernel_size=res1d_kernel_size,
        res1d_n_blocks=res1d_n_blocks,
        res2d_out_channel=res2d_out_channel,
        res2d_kernel_size=res2d_kernel_size,
        res2d_n_blocks=res2d_n_blocks,
        n_target_bins=n_target_bins,
        lm_ckpt=lm_ckpt,
    )

    bsz, seq_len = 32, 128

    seq_lm = torch.randint(0, 20, (bsz, seq_len + 3), dtype=torch.long)
    seq_lm[:, 0] = 20
    seq_lm[:, 30] = 21
    seq_lm[:, 60] = 22

    seq_onehot_resnet = torch.randn(bsz, 21, seq_len)

    out = model(seq_lm, seq_onehot_resnet)

    assert len(out) == 6
    for target in ["d_ca", "d_cb", "d_no", "omega", "theta", "phi"]:
        assert out[target].shape == (bsz, seq_len, seq_len, n_target_bins)
