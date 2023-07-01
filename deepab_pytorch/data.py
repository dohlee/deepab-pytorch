import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
from einops import rearrange
from torch.utils.data import Dataset, DataLoader

SOS_TOKEN = 20
SEP_TOKEN = 21
EOS_TOKEN = 22
PAD_TOKEN = 23
amino_acids = "ARNDCEQGHILKMFPSTWYV"
aa2i = {aa: i for i, aa in enumerate(amino_acids)}  # 20=<sos>, 21=<sep>, 22=<eos>, 23=<pad>


def collate_for_language_modeling(data):
    max_len_for_resnet = max(len(vh) + len(vl) for vh, vl in data)
    max_len_for_lm = max_len_for_resnet + 3

    batch = {}
    # tokenized sequence for language modeling
    seq_lm = torch.full((len(data), max_len_for_lm), PAD_TOKEN, dtype=torch.long)
    for i, (vh, vl) in enumerate(data):
        seq_tokenized = [SOS_TOKEN] + vh + [SEP_TOKEN] + vl + [EOS_TOKEN]
        seq_tokenized = torch.tensor(seq_tokenized, dtype=torch.long)
        seq_lm[i, : len(seq_tokenized)] = seq_tokenized
    batch["seq_lm"] = seq_lm

    return batch


def collate_for_structure_prediction(data):
    max_len_for_resnet = max(len(vh) + len(vl) for vh, vl in data)
    max_len_for_lm = max_len_for_resnet + 3

    batch = {}
    # tokenized sequence for language modeling
    seq_lm = torch.full((len(data), max_len_for_lm), PAD_TOKEN, dtype=torch.long)
    for i, (vh, vl) in enumerate(data):
        seq_tokenized = [SOS_TOKEN] + vh + [SEP_TOKEN] + vl + [EOS_TOKEN]
        seq_tokenized = torch.tensor(seq_tokenized, dtype=torch.long)
        seq_lm[i, : len(seq_tokenized)] = seq_tokenized
    batch["seq_lm"] = seq_lm

    # channelized sequence for ResNet input
    seq_onehot_resnet = torch.zeros((len(data), max_len_for_resnet, 21), dtype=torch.float)
    for i, (vh, vl) in enumerate(data):
        seq = torch.tensor(vh + vl, dtype=torch.long)
        seq_onehot = F.one_hot(seq, num_classes=20).float()
        seq_onehot_resnet[i, : len(seq_onehot), :20] = seq_onehot
        seq_onehot_resnet[i, len(vh), 20] = 1.0  # delimiter channel marks the end of VH
    batch["seq_onehot_resnet"] = rearrange(seq_onehot_resnet, "b l c -> b c l")

    return batch


class DeepAbDatasetForLanguageModeling(Dataset):
    def __init__(self, meta_df):
        super().__init__()
        self.meta_df = meta_df
        self.records = self.meta_df.to_records()

    def __len__(self):
        return len(self.meta_df)

    def _encode_amino_acid_seq(self, seq):
        return [aa2i[aa] for aa in seq]

    def __getitem__(self, i):
        r = self.records[i]

        vh = self._encode_amino_acid_seq(r.vh_seq)
        vl = self._encode_amino_acid_seq(r.vl_seq)

        # seq = [20] + vh + [21] + vl + [22]
        # seq = torch.tensor(seq, dtype=torch.long)

        return vh, vl


class DeepAbDataModuleForLanguageModeling(pl.LightningDataModule):
    def __init__(self, meta, batch_size=128, val_pct=0.1, seed=42):
        super().__init__()
        self.meta_df = pd.read_csv(meta)
        self.batch_size = batch_size

        self.validate = val_pct > 0

        if not self.validate:
            self.train_df = self.meta_df
        else:
            self.meta_df = self.meta_df.sample(frac=1, random_state=seed)

            n_train = int(len(self.meta_df) * (1 - val_pct))
            self.train_df = self.meta_df.iloc[:n_train]
            self.val_df = self.meta_df.iloc[n_train:]

    def setup(self, stage=None):
        self.train_set = DeepAbDatasetForLanguageModeling(self.train_df)
        self.val_set = DeepAbDatasetForLanguageModeling(self.val_df) if self.validate else None

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            collate_fn=collate_for_language_modeling,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            collate_fn=collate_for_language_modeling,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )


class DeepAbDatasetForStructurePrediction(Dataset):
    def __init__(self, meta_df, data_dir):
        super().__init__()
        self.meta_df = meta_df
        self.records = self.meta_df.to_records()

        self.data_dir = data_dir

    def __len__(self):
        return len(self.meta_df)

    def _encode_amino_acid_seq(self, seq):
        return [aa2i[aa] for aa in seq]

    def __getitem__(self, i):
        r = self.records[i]

        vh = self._encode_amino_acid_seq(r.vh_seq)
        vl = self._encode_amino_acid_seq(r.vl_seq)

        # seq = [20] + vh + [21] + vl + [22]
        # seq = torch.tensor(seq, dtype=torch.long)

        return vh, vl


class DeepAbDataModuleForStructurePrediction(pl.LightningDataModule):
    def __init__(self, meta, batch_size=128, val_pct=0.1, seed=42):
        super().__init__()
        self.meta_df = pd.read_csv(meta)
        self.batch_size = batch_size

        self.validate = val_pct > 0

        if not self.validate:
            self.train_df = self.meta_df
        else:
            self.meta_df = self.meta_df.sample(frac=1, random_state=seed)

            n_train = int(len(self.meta_df) * (1 - val_pct))
            self.train_df = self.meta_df.iloc[:n_train]
            self.val_df = self.meta_df.iloc[n_train:]

    def setup(self, stage=None):
        self.train_set = DeepAbDatasetForStructurePrediction(self.train_df)
        self.val_set = (
            DeepAbDatasetForStructurePrediction(self.val_df) if self.validate else None
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            collate_fn=collate_for_structure_prediction,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            collate_fn=collate_for_structure_prediction,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
