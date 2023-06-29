import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

import torch.nn.functional as F
import random

from einops import rearrange

PAD_TOKEN = 23


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
        self.to_out = nn.Linear(d_hidden + input_size, 23)

    def forward(self, input, h_enc, h, c):
        x = torch.cat([input, h_enc], dim=-1)

        if h is None and c is None:
            out, (h, c) = self.lstm(x)

            out = torch.cat([x, out], dim=-1)
            return self.to_out(out), (h, c)
        else:
            out, (h, c) = self.lstm(x, (h, c))

            out = torch.cat([x, out], dim=-1)
            return self.to_out(out), (h, c)

class AntibodyLanguageModel(pl.LightningModule):
    def __init__(self, lr=1e-3, teacher_forcing_ratio=0.5):
        super().__init__()
        self.encoder = BiLSTMEncoder(
            input_size=23,
            d_hidden=64,
        )
        self.decoder = LSTMDecoder(
            input_size=64 + 23,
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
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.save_hyperparameters()

    def forward(self, x):
        # we only need encoder output when training is over
        output, _ = self.encoder(x)
        return output

    def sample(self, batch, teacher_forcing_ratio):
        seq = F.one_hot(batch["seq"], num_classes=24).float()  # bsz, L, 24
        seq = seq[:, :, :-1]  # exclude pad token

        L = seq.size(1)
        device = seq.device

        # use typical seq2seq training here
        # NOTE: original implementation uses (encoder_h, input_seq) as input and
        # use None as initial hidden and cell state of decoder.
        #
        # Let's see if using (h, c) from encoder as initial hidden and cell state
        # works better than that. If not, I will just follow the original one.

        outputs = []  # will be bsz, L, d_input

        # get encoder state
        _, (h, c) = self.encoder(seq)
        # dimension of h and c is tricky; 2 * num_layers, bsz, d_hidden

        # we need to rearrange it to bsz, num_layers, 2 * d_hidden
        h = rearrange(h, "(d l) b h -> b l (d h)", d=2)
        c = rearrange(c, "(d l) b h -> b l (d h)", d=2)
        # then project to decoder input size
        h = self.proj_h(h)
        c = self.proj_c(c)

        h_enc = h[:, -1]
        h_enc = rearrange(h_enc, "b d -> b () d")

        input = seq[:, 0].unsqueeze(1)
        outputs.append(input)

        h_out, c_out = rearrange(h, "b l d -> l b d"), rearrange(c, "b l d -> l b d")
        for t in range(1, L):
            decoder_out, (h_out, c_out) = self.decoder(input, h_enc, h_out, c_out)
            # save output
            outputs.append(decoder_out)

            if random.random() < teacher_forcing_ratio:
                s = seq[:, t]  # use ground truth as input
            else:
                # use one-hot encoded predicted output as input
                # IDEA: use temperature-based sampling here?
                s = (
                    F.one_hot(torch.argmax(decoder_out, dim=-1), num_classes=23)
                    .float()
                    .squeeze(1)
                )

            input = rearrange(s, "b d -> b () d")

        return torch.cat(outputs, dim=1).to(device)

    def training_step(self, batch, batch_idx):
        outputs = self.sample(batch, self.teacher_forcing_ratio)

        # calculate loss
        loss = self.criterion(
            rearrange(outputs, "b L d -> (b L) d"), rearrange(batch["seq"], "b L -> (b L)")
        )

        acc_mask = (batch["seq"] != PAD_TOKEN).float()
        acc = torch.sum(
            (torch.argmax(outputs, dim=-1) == batch["seq"]) * acc_mask
        ) / torch.sum(acc_mask)

        self.log_dict(
            {"train/loss": loss.item(), "train/accuracy": acc.item()},
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.sample(batch, 0.0)  # no teacher forcing at validation

        # calculate loss
        loss = self.criterion(
            rearrange(outputs, "b L d -> (b L) d"), rearrange(batch["seq"], "b L -> (b L)")
        )

        acc_mask = (batch["seq"] != PAD_TOKEN).float()
        acc = torch.sum(
            (torch.argmax(outputs, dim=-1) == batch["seq"]) * acc_mask
        ) / torch.sum(acc_mask)

        self.log_dict(
            {"val/loss": loss.item(), "val/accuracy": acc.item()}, prog_bar=True, on_epoch=True
        )

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            },
        }


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class ResBlock1D(nn.Module):
    def __init__(self, channel, kernel_size, stride):
        super().__init__()

        self.layer = nn.Sequential(
            Residual(
                nn.Sequential(
                    nn.Conv1d(
                        channel, channel, kernel_size, stride, padding="same", bias=False
                    ),
                    nn.BatchNorm1d(channel),
                    nn.ReLU(),
                    nn.Conv1d(
                        channel, channel, kernel_size, stride, padding="same", bias=False
                    ),
                    nn.BatchNorm1d(channel),
                )
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class ResNet1D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, n_blocks):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList(
            [ResBlock1D(out_channel, kernel_size, stride=1) for _ in range(n_blocks)]
        )

    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)

        return x


class ResBlock2D(nn.Module):
    def __init__(self, channel, kernel_size, stride, dilation):
        super().__init__()

        self.layer = nn.Sequential(
            Residual(
                nn.Sequential(
                    nn.Conv2d(
                        channel,
                        channel,
                        kernel_size,
                        stride,
                        padding="same",
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(),
                    nn.Conv2d(
                        channel,
                        channel,
                        kernel_size,
                        stride,
                        padding="same",
                        dilation=dilation,
                        bias=False,
                    ),
                )
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class ResNet2D(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, n_blocks, dilation=[1, 2, 4, 8, 16]
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding="same", bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList(
            [
                ResBlock2D(
                    out_channel, kernel_size, stride=1, dilation=dilation[i % len(dilation)]
                )
                for i in range(n_blocks)
            ]
        )

    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)

        return x


class DeepAb(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.language_model = AntibodyLanguageModel()

    def forward(self, x):
        pass
