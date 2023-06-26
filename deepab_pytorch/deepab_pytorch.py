import torch
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F


class BiLSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class LSTMDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class AntibodyLanguageModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = BiLSTMEncoder()
        self.decoder = LSTMDecoder()

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass


class OneDimensionalResNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class TwoDimensionalResNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class DeepAb(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.language_model = AntibodyLanguageModel()

    def forward(self, x):
        pass
