import torch
import pytest

from deepab_pytorch import AntibodyLanguageModel


def test_AntibodyLanguageModel_init():
    model = AntibodyLanguageModel()
    assert model is not None


def test_AntibodyLanguageModel_forward():
    bsz, max_len = 3, 32
    x = torch.randint(0, 20, (bsz, max_len + 3))

    x[:, 0] = 20
    x[:, 10] = 21
    x[:, 19] = 22

    model = AntibodyLanguageModel()
    out = model(x)
    assert out.shape == (bsz, max_len, 128)


def test_AntibodyLanguageModel_training_step():
    bsz, max_len = 3, 32

    seq = torch.randint(0, 24, (bsz, max_len + 3))
    batch = {"seq_lm": seq}

    model = AntibodyLanguageModel()
    loss = model.training_step(batch, 0)
    print(loss)

    assert loss.shape == ()  # should be scalar
