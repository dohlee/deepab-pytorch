import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.
    Focal loss aims to penalize incorrect predictions more heavily than easy ones,
    through down-weighting easy examples by a factor of (1 - p_t)^gamma,
    where p_t is the model's prediction of probability for the true class.

    For official implementation, refer to https://github.com/clcarwin/focal_loss_pytorch
    """

    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()

        self.gamma = gamma
        self.reduction = reduction

        # handle unknown reduction
        if self.reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"reduction='{self.reduction}' is not supported for FocalLoss.")

    def forward(self, input, target):
        log_pt = F.log_softmax(input, dim=-1)

        # gather the log probabilities with respect to targets
        target = target.unsqueeze(-1)
        log_pt = log_pt.gather(dim=-1, index=target)

        pt = log_pt.exp()

        # compute the loss
        loss = -((1 - pt) ** self.gamma) * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
