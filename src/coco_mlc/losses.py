"""Loss functions for a strongly imbalanced multi-label problem."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pos_weight(Y: np.ndarray, cap: float | None = 20.0) -> torch.Tensor:
    """``pos_weight`` for ``BCEWithLogitsLoss``: negatives / positives per class.

    Without a cap, ``hair drier`` would get a weight around 640, which blows up
    the gradient and makes training unstable. The cap is a trade-off to tune
    on the validation set.
    """
    Y = np.asarray(Y)
    pos = Y.sum(axis=0).astype(np.float64)
    neg = Y.shape[0] - pos
    weight = np.divide(neg, pos, out=np.ones_like(neg), where=pos > 0)
    if cap is not None:
        weight = np.clip(weight, 1.0, cap)
    return torch.tensor(weight, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Sigmoid focal loss (Lin et al., 2017), multi-label version.

    Down-weights examples that are already well classified, so the gradient
    focuses on the rare positive labels.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce * (1 - p_t).pow(self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = loss * alpha_t
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """Asymmetric loss (Ben-Baruch et al., 2020), a standard choice on MS COCO.

    Two separate mechanisms: stronger focusing on negatives
    (``gamma_neg > gamma_pos``) and a hard clip (``clip``) that ignores
    negatives that are already very well classified. With about 2.9 positive
    classes out of 80, the negatives would otherwise dominate the gradient.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_pos = p
        p_neg = 1.0 - p
        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(p_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(p_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt = p_pos * targets + p_neg * (1 - targets)
                gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                modulation = (1 - pt).pow(gamma)
            loss = loss * modulation

        loss = -loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


LOSS_NAMES = ("bce", "bce_pos_weight", "focal", "asl")


def build_criterion(name: str, pos_weight: torch.Tensor | None = None, **kwargs) -> nn.Module:
    """Build the loss. Every option expects logits as input."""
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "bce_pos_weight":
        if pos_weight is None:
            raise ValueError("bce_pos_weight requires pos_weight")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "focal":
        return FocalLoss(**kwargs)
    if name == "asl":
        return AsymmetricLoss(**kwargs)
    raise ValueError(f"unknown loss: {name!r} (available: {LOSS_NAMES})")
