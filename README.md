# deepab-pytorch (wip)

[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://github.com/Lightning-AI/lightning)

![img](img/deepab_banner.png)

An unofficial re-implementation of DeepAb, an interpretable deep learning model for antibody structure prediction.

## Installation

```bash
pip install deepab-pytorch
```

## Usage

**DeepAb model**
```python
import torch
from deepab_pytorch import DeepAb

antibody_lm_d_enc = 64
res1d_out_channel = 32
res1d_kernel_size = 17
res1d_n_blocks = 3
res2d_out_channel = 64
res2d_kernel_size = 5
res2d_n_blocks = 25
n_target_bins = 37

model = DeepAb(
  antibody_lm_d_enc=antibody_lm_d_enc,
  res1d_out_channel=res1d_out_channel,
  res1d_kernel_size=res1d_kernel_size,
  res1d_n_blocks=res1d_n_blocks,
  res2d_out_channel=res2d_out_channel,
  res2d_kernel_size=res2d_kernel_size,
  res2d_n_blocks=res2d_n_blocks,
  n_target_bins=n_target_bins
)

bsz, seq_len = 32, 256

# dummy input for antibody language model
# seq_len + 3 because of special tokens <sos>, <sep>, <eos>,
# each represents the start of heavy chain, light chain
# and the end of sequence, respectively.
seq_lm = torch.randint(0, 20, (bsz, seq_len + 3), dtype=torch.long)

SOS_TOKEN, SEP_TOKEN, EOS_TOKEN = 20, 21, 22
seq_lm[:, 0] = SOS_TOKEN
seq_lm[:, 128] = SEP_TOKEN
seq_lm[:, 254] = EOS_TOKEN

# dummy input for ResNet model
seq_onehot_resnet = torch.randn(bsz, 21, seq_len)

# forward pass through the model
out = model(seq_lm, seq_onehot_resnet)

# Six 2D matrices of shape (B, L, L, n_bins) are produced
n_target_bins = 37

assert len(out) == 6
for target in ['d_ca', 'd_cb', 'd_no', 'omega', 'theta', 'phi']:
    assert out[target].shape == (bsz, seq_len, seq_len, n_target_bins)
```

## Testing
```bash
pytest -vs --disable-warnings
```

## Citation
```bibtex
@article{ruffolo2022antibody,
  title={Antibody structure prediction using interpretable deep learning},
  author={Ruffolo, Jeffrey A and Sulam, Jeremias and Gray, Jeffrey J},
  journal={Patterns},
  volume={3},
  number={2},
  pages={100406},
  year={2022},
  publisher={Elsevier}
}
```