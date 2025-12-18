import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Binary Focal Loss for logits.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (B,) raw model outputs
        targets: (B,) binary labels 0/1
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets)

        focal_weight = (self.alpha * targets + (1 - self.alpha) * (1 - targets))
        focal_weight = focal_weight * (1 - pt).pow(self.gamma)

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
