import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

import torch.nn.functional as F
import random

from einops import rearrange


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size=23, d_hidden=64):
        super().__init__()
        self.d_hidden = d_hidden

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=d_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return out, (h, c)


class LSTMDecoder(nn.Module):
    def __init__(self, input_size=23, d_hidden=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=d_hidden,
            num_layers=2,
            bidirectional=False,  # decoder is not bidirectional
            batch_first=True,
        )
        self.to_out = nn.Linear(d_hidden, 23)

    def forward(self, x, h, c):
        out, (h, c) = self.lstm(x, (h, c))
        return self.to_out(out), (h, c)


class AntibodyLanguageModel(pl.LightningModule):
    def __init__(self, lr=1e-3, teacher_forcing_ratio=0.5):
        super().__init__()
        self.encoder = BiLSTMEncoder(
            input_size=23,
            d_hidden=64,
        )
        self.decoder = LSTMDecoder(
            input_size=23,
            d_hidden=64,
        )

        self.proj_h = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        self.proj_c = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
        )

        # optimization
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, x):
        # we only need encoder output when training is over
        output, _ = self.encoder(x)
        return output

    def training_step(self, batch, batch_idx):
        seq = F.one_hot(batch["seq"], num_classes=23).float()  # bsz, L, 23
        L = seq.size(1)
        device = seq.device

        # use typical seq2seq training here
        # NOTE: original implementation uses (encoder_h, input_seq) as input and
        # use None as initial hidden and cell state of decoder.
        #
        # Let's see if using (h, c) from encoder as initial hidden and cell state
        # works better than that. If not, I will just follow the original one.
        outputs = torch.zeros_like(seq).to(device)  # bsz, L, d_input

        # get encoder state
        _, (h, c) = self.encoder(seq)
        # dimension of h and c is tricky; 2 * num_layers, bsz, d_hidden

        # we need to rearrange it to num_layers, bsz, 2 * d_hidden
        h = rearrange(h, "(d l) b h -> l b (d h)", d=2)
        c = rearrange(c, "(d l) b h -> l b (d h)", d=2)
        # then project to decoder input size
        h = self.proj_h(h)
        c = self.proj_c(c)

        input = seq[:, 0]  # bsz, d_input
        for t in range(1, L):
            # decoder expects (bsz, L, d_input) as input, so
            decoder_out, (h, c) = self.decoder(rearrange(input, "b d -> b () d"), h, c)
            # save output
            outputs[:, t] = decoder_out.squeeze(1)

            if random.random() < self.teacher_forcing_ratio:
                input = seq[:, t]  # use ground truth as input
            else:
                input = outputs[:, t]

        # calculate loss
        loss = self.criterion(
            rearrange(outputs, "b L d -> (b L) d"),
            rearrange(batch["seq"], "b L -> (b L)"),
        )

        return loss

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