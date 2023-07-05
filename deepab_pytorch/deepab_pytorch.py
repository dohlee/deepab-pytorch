import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

import torch.nn.functional as F
import random
import wandb

from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from deepab_pytorch.loss import FocalLoss

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
    def __init__(self, d_enc=64, d_dec=64, lr=1e-3, teacher_forcing_ratio=0.5):
        super().__init__()
        self.encoder = BiLSTMEncoder(
            input_size=23,
            d_hidden=d_enc,
        )
        self.decoder = LSTMDecoder(
            input_size=d_enc + 23,
            d_hidden=d_dec,
        )

        self.proj_h = nn.Sequential(
            nn.Linear(d_enc * 2, d_dec),
            nn.Tanh(),
        )
        self.proj_c = nn.Sequential(
            nn.Linear(d_enc * 2, d_dec),
            nn.Tanh(),
        )

        # optimization
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.save_hyperparameters()

    def forward(self, seq):
        bsz, seq_len = seq.size(0), seq.size(1) - 3

        seq_onehot = F.one_hot(seq, num_classes=24).float()  # bsz, L+3, 24

        seq_onehot = seq_onehot[:, :, :-1]  # exclude pad token

        # we only need encoder output for amino acids when training is over
        output, _ = self.encoder(seq_onehot)  # bsz, L+3, d

        # discard embeddings for <sos>, <sep>, <eos> and <pad>
        aa_mask_list = seq < 20
        masked_output = torch.zeros(bsz, seq_len, output.size(-1))
        for i, aa_mask in enumerate(aa_mask_list):
            masked_output[i, : aa_mask.sum()] = output[i, aa_mask]

        return masked_output

    def sample(self, batch, teacher_forcing_ratio):
        seq = F.one_hot(batch["seq_lm"], num_classes=24).float()  # bsz, L, 24
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
            rearrange(outputs, "b L d -> (b L) d"), rearrange(batch["seq_lm"], "b L -> (b L)")
        )

        acc_mask = (batch["seq_lm"] != PAD_TOKEN).float()
        acc = torch.sum(
            (torch.argmax(outputs, dim=-1) == batch["seq_lm"]) * acc_mask
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
            rearrange(outputs, "b L d -> (b L) d"), rearrange(batch["seq_lm"], "b L -> (b L)")
        )

        acc_mask = (batch["seq_lm"] != PAD_TOKEN).float()
        acc = torch.sum(
            (torch.argmax(outputs, dim=-1) == batch["seq_lm"]) * acc_mask
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


class Symmetrization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + x.transpose(-1, -2)


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


class CrissCrossAttention(nn.Module):
    def __init__(self, channel, reduction_factor=8):
        super().__init__()

        self.to_q = nn.Conv2d(channel, channel // reduction_factor, kernel_size=1)
        self.to_k = nn.Conv2d(channel, channel // reduction_factor, kernel_size=1)
        self.to_v = nn.Conv2d(channel, channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        device = x.device
        bsz, seq_len = x.shape[0], x.shape[2]

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qh, kh, vh = map(lambda t: rearrange(t, "b c h w -> (b w) h c"), (q, k, v))
        qw, kw, vw = map(lambda t: rearrange(t, "b c h w -> (b h) w c"), (q, k, v))

        self_mask_h = repeat(
            torch.eye(seq_len, device=device), "i j -> (b w) i j", b=bsz, w=seq_len
        )
        logit_h = einsum("b i c, b j c -> b i j", qh, kh) + self_mask_h
        logit_w = einsum("b i c, b j c -> b i j", qw, kw)

        # logit_h = rearrange(logit_h, "(b w) i j -> b w i j", b=bsz)
        # logit_w = rearrange(logit_w, "(b h) i j -> b h i j", b=bsz)

        att_h = F.softmax(logit_h, dim=-1)
        att_w = F.softmax(logit_w, dim=-1)

        out_h = einsum("b i j, b j c -> b i c", att_h, vh)
        out_w = einsum("b i j, b j c -> b i c", att_w, vw)

        out_h = rearrange(out_h, "(b w) h c -> b c h w", b=bsz)
        out_w = rearrange(out_w, "(b h) w c -> b c h w", b=bsz)

        return self.gamma * (out_h + out_w) + x


class RecurrentCrissCrossAttention(nn.Module):
    def __init__(self, in_channel, inner_channel, kernel_size=3):
        super().__init__()

        self.criss_cross_att = CrissCrossAttention(inner_channel)

        self.layer = nn.Sequential(
            # project to inner_channel
            nn.Conv2d(in_channel, inner_channel, kernel_size, padding="same"),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(),
            # apply criss-cross attention 2 times
            self.criss_cross_att,
            self.criss_cross_att,
            # project to in_channel
            nn.Conv2d(inner_channel, in_channel, kernel_size, padding="same"),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class DeepAb(pl.LightningModule):
    def __init__(
        self,
        antibody_lm_d_enc=64,
        res1d_out_channel=32,
        res1d_kernel_size=17,
        res1d_n_blocks=3,
        res2d_out_channel=64,
        res2d_kernel_size=5,
        res2d_n_blocks=25,
        n_target_bins=37,
        lr=1e-2,
        lm_ckpt=None,
    ):
        super().__init__()
        self.language_model = AntibodyLanguageModel(d_enc=antibody_lm_d_enc)
        if lm_ckpt is not None:
            ckpt = torch.load(lm_ckpt)

            # attach pretrained antibody-LM weights
            if "pytorch-lightning_version" in ckpt:
                self.language_model.load_state_dict(ckpt["state_dict"])
            else:
                self.language_model.load_state_dict(ckpt)

        # define ResNets
        self.resnet1d = ResNet1D(21, res1d_out_channel, res1d_kernel_size, res1d_n_blocks)

        res2d_in_channel = 2 * (antibody_lm_d_enc * 2 + res1d_out_channel)  # 320
        self.resnet2d = ResNet2D(
            res2d_in_channel,
            res2d_out_channel,
            res2d_kernel_size,
            res2d_n_blocks,
            dilation=[1, 2, 4, 8, 16],
        )

        # prediction heads
        self.targets = ["d_ca", "d_cb", "d_no", "omega", "theta", "phi"]
        self.heads = nn.ModuleDict()

        for target in self.targets:
            self.heads[target] = nn.Sequential(
                nn.Conv2d(res2d_out_channel, n_target_bins, res2d_kernel_size, padding="same"),
                RecurrentCrissCrossAttention(n_target_bins, n_target_bins // 4, kernel_size=3),
                # symmetry is enforced for d_ca, d_cb and omega
                Symmetrization() if target in ["d_ca", "d_cb", "omega"] else nn.Identity(),
                Rearrange("b c h w -> b h w c"),
            )

        # optimization
        self.criterion = FocalLoss(gamma=2.0, reduction="none")
        self.lr = lr

    def forward(self, seq_lm, seq_onehot_resnet):
        # seq_lm: (batch_size, seq_len + 3)
        # seq_onehot_resnet: (batch_size, 21, seq_len)
        seq_len = seq_onehot_resnet.shape[-1]

        # get language model embeddings without gradients
        with torch.no_grad():
            lm_emb = self.language_model(seq_lm)  # (batch_size, seq_len, 128)
            lm_emb = rearrange(lm_emb, "b l c -> b c l")  # (batch_size, 128, seq_len)
            lm_emb = lm_emb.to(seq_onehot_resnet.device)

        # get resnet1d embeddings
        res1d_emb = self.resnet1d(seq_onehot_resnet)  # (batch_size, 64, seq_len)

        # concatenate embeddings from language model and resnet1d
        x_1d = torch.cat([lm_emb, res1d_emb], dim=1)  # (batch_size, 160, seq_len)
        # expand to 2d
        x_2d = torch.cat(
            [
                repeat(x_1d, "b c i -> b c i j", j=seq_len),
                repeat(x_1d, "b c i -> b c j i", j=seq_len),
            ],
            dim=1,
        )

        res2d_emb = self.resnet2d(x_2d)  # (batch_size, 64, seq_len, seq_len)

        out = {}
        for target in self.targets:
            # (batch_size, seq_len, seq_len, n_target_bins)
            out[target] = self.heads[target](res2d_emb)

        return out

    def training_step(self, batch, batch_idx):
        seq_lm = batch["seq_lm"]
        seq_onehot_resnet = batch["seq_onehot_resnet"]

        preds = self(seq_lm, seq_onehot_resnet)
        loss = 0

        targets = batch["target"]
        loss_mask = batch["loss_mask"]
        for t in self.targets:
            # preds[t]: (batch_size, seq_len, seq_len, n_target_bins)
            # targets[t]: (batch_size, seq_len, seq_len)
            # loss_mask[t]: (batch_size, seq_len, seq_len)
            loss_unmasked = self.criterion(preds[t], targets[t].long()).squeeze(-1)
            loss += (loss_unmasked * loss_mask[t]).sum() / loss_mask[t].sum()

        self.log("train/loss", loss, prog_bar=True, on_step=True)

        # log example predictions and targets as image
        if batch_idx % 25 == 0 and isinstance(self.logger, pl.loggers.WandbLogger):
            pred_img = []
            for t in self.targets:
                pred = preds[t][0].argmax(dim=-1).float()
                pred_img.append(pred)
            pred_img = torch.cat(pred_img, dim=1)

            target_img = []
            for t in self.targets:
                target = targets[t][0].float()
                target_img.append(target)
            target_img = torch.cat(target_img, dim=1)

            self.logger.experiment.log(
                {
                    "example": [
                        wandb.Image(
                            pred_img,
                            caption="Prediction (d_ca, d_cb, d_no, omega, theta, phi)",
                        ),
                        wandb.Image(
                            target_img, caption="Target (d_ca, d_cb, d_no, omega, theta, phi)"
                        ),
                    ]
                }
            )

        return loss

    def validation_step(self, batch, batch_idx):
        seq_lm = batch["seq_lm"]
        seq_onehot_resnet = batch["seq_onehot_resnet"]

        preds = self(seq_lm, seq_onehot_resnet)
        loss = 0

        targets = batch["target"]
        loss_mask = batch["loss_mask"]
        for t in self.targets:
            loss_unmasked = self.criterion(preds[t], targets[t].long()).squeeze(-1)
            loss += (loss_unmasked * loss_mask[t]).sum() / loss_mask[t].sum()

        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
