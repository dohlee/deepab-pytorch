import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

import torch.nn.functional as F


class BiLSTMEncoder(nn.Module):
    def __init__(self, d_hidden=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=23,
            hidden_size=d_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.d_hidden = d_hidden

    def forward(self, x):
        out, _ = self.lstm(x)

        # take last hidden state of forward and first hidden state of backward
        d = self.d_hidden
        summary_vector = torch.cat([out[:, -1, :d], out[:, 0, d:]], dim=-1)

        return {
            "output": out,
            "summary_vector": summary_vector,
        }


class LSTMDecoder(nn.Module):
    def __init__(self, input_size=64, d_hidden=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=d_hidden,
            num_layers=2,
        )

    def forward(self, x):
        pass


class AntibodyLanguageModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.encoder = BiLSTMEncoder()
        self.decoder = LSTMDecoder()

        self.proj_summary_vector = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
        )

        # optimization
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


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
