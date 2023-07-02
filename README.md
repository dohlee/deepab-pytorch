# deepab-pytorch (wip)

[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://github.com/Lightning-AI/lightning)

![img](img/deepab_banner.png)

An unofficial re-implementation of DeepAb, an interpretable deep learning model for antibody structure prediction.

## Installation

```bash
pip install deepab-pytorch
```

## Usage

```python
import torch
from deepab_pytorch import DeepAb

model = DeepAb()
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