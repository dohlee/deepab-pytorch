import torch
import pytest

from deepab_pytorch import AntibodyLanguageModel


def test_AntibodyLanguageModel_init():
    model = AntibodyLanguageModel()
    assert model is not None


def test_AntibodyLanguageModel_forward():
    bsz, max_len, d_input = 3, 32, 23
    x = torch.randn(bsz, max_len, d_input)

    model = AntibodyLanguageModel()
    out = model(x)
    assert out.shape == (bsz, max_len, 128)


def test_AntibodyLanguageModel_training_step():
    bsz, max_len = 3, 32

    seq = torch.randint(0, 23, (bsz, max_len))
    batch = {"seq": seq}

    model = AntibodyLanguageModel()
    loss = model.training_step(batch, 0)
    print(loss)

    assert loss.shape == ()  # should be scalar
