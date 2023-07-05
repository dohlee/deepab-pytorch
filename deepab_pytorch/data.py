import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from protstruc import AntibodyFvStructure

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
    bsz = len(data)
    max_len_for_resnet = max(len(d["vh"]) + len(d["vl"]) for d in data)
    max_len_for_lm = max_len_for_resnet + 3

    batch = {}
    # tokenized sequence for language modeling
    seq_lm = torch.full((bsz, max_len_for_lm), PAD_TOKEN, dtype=torch.long)
    for i, d in enumerate(data):
        seq_tokenized = [SOS_TOKEN] + d["vh"] + [SEP_TOKEN] + d["vl"] + [EOS_TOKEN]
        seq_tokenized = torch.tensor(seq_tokenized, dtype=torch.long)
        seq_lm[i, : len(seq_tokenized)] = seq_tokenized
    batch["seq_lm"] = seq_lm

    # channelized sequence for ResNet input
    seq_onehot_resnet = torch.zeros((bsz, max_len_for_resnet, 21), dtype=torch.float)
    for i, d in enumerate(data):
        seq = torch.tensor(d["vh"] + d["vl"], dtype=torch.long)
        seq_onehot = F.one_hot(seq, num_classes=20).float()
        seq_onehot_resnet[i, : len(seq_onehot), :20] = seq_onehot
        seq_onehot_resnet[i, len(d["vh"]), 20] = 1.0  # delimiter channel marks the end of VH
    batch["seq_onehot_resnet"] = rearrange(seq_onehot_resnet, "b l c -> b c l")

    # target and loss mask
    target = {}
    loss_mask = {}
    for k in ["d_ca", "d_cb", "d_no", "omega", "theta", "phi"]:
        trg = torch.zeros((bsz, max_len_for_resnet, max_len_for_resnet))
        lm = torch.zeros((bsz, max_len_for_resnet, max_len_for_resnet))

        for i, d in enumerate(data):
            length = d[k].size(0)
            trg[i, :length, :length] = d[k]
            lm[i, :length, :length] = 1.0

        target[k] = trg
        loss_mask[k] = lm
    
    batch['target'] = target
    batch['loss_mask'] = loss_mask

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
    def __init__(self, meta_df, data_dir, n_target_bins=37):
        super().__init__()
        self.meta_df = meta_df
        self.records = self.meta_df.to_records()

        self.data_dir = data_dir
        self.n_target_bins = n_target_bins

    def __len__(self):
        return len(self.meta_df)

    def _encode_amino_acid_seq(self, seq):
        return [aa2i[aa] for aa in seq]

    def _bin_dist_target(self, target, interval_size=0.5):
        thresholds = torch.arange(1, self.n_target_bins) * interval_size
        return np.digitize(target, thresholds)

    def _bin_dihedral_target(self, target):
        interval_size = 2 * np.pi / self.n_target_bins
        thresholds = torch.arange(1, self.n_target_bins) * interval_size
        return np.digitize(target, thresholds)

    def _bin_planar_target(self, target):
        interval_size = np.pi / self.n_target_bins
        thresholds = torch.arange(1, self.n_target_bins) * interval_size
        return np.digitize(target, thresholds)

    def __getitem__(self, i):
        r = self.records[i]
        ret = {}

        pt_path = f"{self.data_dir}/{r.pdb_id}.pdb"
        struc = AntibodyFvStructure(
            pt_path, impute_missing_atoms=True, heavy_chain_id="H", light_chain_id="L"
        )

        vh = self._encode_amino_acid_seq(struc.get_seq(chain="H"))
        vl = self._encode_amino_acid_seq(struc.get_seq(chain="L"))
        ret["vh"] = vh
        ret["vl"] = vl

        irg = struc.inter_residue_geometry()

        for k in ["d_ca", "d_cb", "d_no", "omega", "theta", "phi"]:
            # binning
            if k in ["d_ca", "d_cb", "d_no"]:
                t = self._bin_dist_target(irg[k])
            elif k in ["omega", "theta"]:
                t = self._bin_dihedral_target(irg[k])
            else:
                t = self._bin_planar_target(irg[k])

            ret[k] = torch.tensor(t)

        return ret


class DeepAbDataModuleForStructurePrediction(pl.LightningDataModule):
    def __init__(self, meta_df, data_dir, n_target_bins, batch_size=128, val_pct=0.1, seed=42):
        super().__init__()
        self.meta_df = meta_df
        self.data_dir = data_dir
        self.n_target_bins = n_target_bins
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
        self.train_set = DeepAbDatasetForStructurePrediction(
            self.train_df,
            self.data_dir,
            self.n_target_bins,
        )
        self.val_set = (
            DeepAbDatasetForStructurePrediction(
                self.val_df,
                self.data_dir,
                self.n_target_bins,
            )
            if self.validate
            else None
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
