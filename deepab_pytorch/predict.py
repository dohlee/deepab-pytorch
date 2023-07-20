import argparse
import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import warnings
import protstruc
import protstruc.geometry as geom
import protstruc.io as io

from einops import rearrange
from deepab_pytorch import DeepAb
from deepab_pytorch.data import SOS_TOKEN, SEP_TOKEN, EOS_TOKEN, PAD_TOKEN

warnings.filterwarnings("ignore")

amino_acids = "ARNDCEQGHILKMFPSTWYV"
aa2i = {aa: i for i, aa in enumerate(amino_acids)}  # 20=<sos>, 21=<sep>, 22=<eos>, 23=<pad>


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vh", required=True, help="VH sequence")
    parser.add_argument("--vl", required=True, help="VL sequence")
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to structure prediction checkpoint",
    )
    parser.add_argument("--out-geometry", required=True, help="Output pt file path")
    parser.add_argument("--out-pdb", required=True, help="Output pdb file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def encode_amino_acid_seq(seq):
    return [aa2i[aa] for aa in seq]


def unbin_dist_target(mat, interval_size=0.5):
    # mat: (L, L, n_target_bins)
    n_target_bins = mat.shape[-1]

    bin_centers = np.arange(n_target_bins) * interval_size + interval_size / 2.0
    bin_centers = bin_centers.reshape(1, 1, -1)

    mask = mat.argmax(axis=-1) != n_target_bins - 1
    return (mat * bin_centers).sum(axis=-1), mask


def unbin_dihedral_target(mat):
    # mat: (L, L, n_target_bins)
    n_target_bins = mat.shape[-1]
    interval_size = 2 * np.pi / n_target_bins

    bin_centers = np.arange(n_target_bins) * interval_size + interval_size / 2.0
    bin_centers = bin_centers.reshape(1, 1, -1)

    return (mat * bin_centers).sum(axis=-1)


def unbin_planar_target(mat):
    # mat: (L, L, n_target_bins)
    n_target_bins = mat.shape[-1]
    interval_size = np.pi / n_target_bins

    bin_centers = np.arange(n_target_bins) * interval_size + interval_size / 2.0
    bin_centers = bin_centers.reshape(1, 1, -1)

    return (mat * bin_centers).sum(axis=-1)


def main():
    args = parse_argument()
    pl.seed_everything(args.seed)

    model = DeepAb()
    model.load_state_dict(torch.load(args.ckpt)["state_dict"])
    model.eval()

    # prepare data
    vh, vl = encode_amino_acid_seq(args.vh), encode_amino_acid_seq(args.vl)

    # tokenized sequence for language modeling
    seq_lm = [SOS_TOKEN] + vh + [SEP_TOKEN] + vl + [EOS_TOKEN]
    seq_lm = torch.tensor(seq_lm, dtype=torch.long).unsqueeze(0)

    # channelized sequence for ResNet input
    seq = torch.tensor(vh + vl, dtype=torch.long)
    seq_onehot_resnet = F.one_hot(seq, num_classes=21).float()
    seq_onehot_resnet[len(vh), 20] = 1.0  # delimiter channel marks the end of VH
    seq_onehot_resnet = rearrange(seq_onehot_resnet, "l c -> () c l")

    # predict and save
    with torch.no_grad():
        out = model(seq_lm, seq_onehot_resnet)
    torch.save(out, args.out_geometry)

    # structure realization
    out = {k: v.squeeze().softmax(dim=-1).cpu().numpy() for k, v in out.items()}

    d_cb, mask = unbin_dist_target(out["d_cb"])
    omega = unbin_dihedral_target(out["omega"])
    theta = unbin_dihedral_target(out["theta"])
    phi = unbin_planar_target(out["phi"])

    distmat = geom.reconstruct_backbone_distmat_from_interresidue_geometry(
        d_cb, omega, theta, phi, chain_breaks=[len(vh) - 1], mask=mask
    )

    coords = geom.initialize_backbone_with_mds(distmat, max_iter=500)
    sequences = [args.vh, args.vl]
    chain_ids = ["H", "L"]

    io.to_pdb(args.out_pdb, coords, sequences, chain_ids)


if __name__ == "__main__":
    main()
