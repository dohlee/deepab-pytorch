import torch
import pytorch_lightning as pl
import pandas as pd

from torch.utils.data import Dataset, DataLoader

PAD_TOKEN = 0
amino_acids = "ARNDCEQGHILKMFPSTWYV"
aa2i = {aa: i for i, aa in enumerate(amino_acids, 4)}  # 0=<pad>, 1=<sos>, 2=<sep>, 3=<eos>


def collate(data):
    max_len = max(len(seq) for seq in data)

    batch = {}
    seq = torch.full((len(data), max_len), PAD_TOKEN, dtype=torch.long)
    for i, d in enumerate(data):
        seq[i, : len(d)] = d

    batch["seq"] = seq
    return batch


class AntibodyLanguageModelDataset(Dataset):
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

        seq = [0] + vh + [1] + vl + [2]
        seq = torch.tensor(seq, dtype=torch.long)

        return seq


class AntibodyLanguageModelDataModule(pl.LightningDataModule):
    def __init__(self, meta, batch_size=8, val_pct=0.1, seed=42):
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
        self.train_set = AntibodyLanguageModelDataset(self.train_df)
        self.val_set = AntibodyLanguageModelDataset(self.val_df) if self.validate else None

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            collate_fn=collate,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            collate_fn=collate,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
