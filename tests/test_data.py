import pytest
import deepab_pytorch.data as data
import pandas as pd

from torch.utils.data import DataLoader
from tqdm import tqdm


@pytest.fixture(scope="module")
def ds_struc_pred():
    meta_df = pd.read_csv("data/sabdab/meta.csv")
    dataset = data.DeepAbDatasetForStructurePrediction(
        meta_df=meta_df,
        data_dir="data/sabdab/pdb_renumbered",
    )

    yield dataset


def test_DeepAbDatasetForStructurePrediction_keys(ds_struc_pred):
    loader = DataLoader(ds_struc_pred, batch_size=1, shuffle=False)

    for batch in loader:
        break

    assert "vh" in batch
    assert "vl" in batch
    assert "d_ca" in batch
    assert "d_cb" in batch
    assert "d_no" in batch
    assert "omega" in batch
    assert "theta" in batch
    assert "phi" in batch


@pytest.mark.long
def test_DeepAbDatasetForStructurePrediction_single_loop(ds_struc_pred):
    loader = DataLoader(ds_struc_pred, batch_size=1, shuffle=False)
    for batch in tqdm(loader):
        pass


def test_collate_for_structure_prediction(ds_struc_pred):
    loader = DataLoader(
        ds_struc_pred,
        batch_size=8,
        collate_fn=data.collate_for_structure_prediction,
    )

    for batch in loader:
        break
