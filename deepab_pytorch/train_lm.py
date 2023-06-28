import argparse
import torch
import pytorch_lightning as pl
import pandas as pd
import os

from pytorch_lightning.callbacks import LearningRateMonitor
from deepab_pytorch import AntibodyLanguageModel
from deepab_pytorch.data import AntibodyLanguageModelDataModule


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta", required=True, help="Metadata file for train/validation data"
    )
    parser.add_argument(
        "--val-pct", type=float, default=0.1, help="Proportion of validation data to use."
    )
    parser.add_argument("-b", "--bsz", type=int, default=128, help="Batch size")
    parser.add_argument(
        "-e", "--epochs", type=int, default=5, help="Number of epochs to train"
    )
    parser.add_argument(
        "-l", "--learning-rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "-t",
        "--teacher-forcing-ratio",
        type=float,
        default=0.5,
        help="Teacher forcing ratio",
    )
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        default=False,
        help="Don't use wandb for logging",
    )
    return parser.parse_args()


def main():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.enabled = False

    args = parse_argument()
    pl.seed_everything(args.seed)

    if args.no_wandb:
        logger = None
    else:
        logger = pl.loggers.WandbLogger(
            project="deepab-pytorch",
            entity="dohlee",
        )

    lang_model = AntibodyLanguageModel(
        lr=args.learning_rate, teacher_forcing_ratio=args.teacher_forcing_ratio
    )

    dm = AntibodyLanguageModelDataModule(
        args.meta, batch_size=args.bsz, val_pct=args.val_pct, seed=args.seed
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=1,
    )

    trainer.fit(lang_model, datamodule=dm)


if __name__ == "__main__":
    main()
