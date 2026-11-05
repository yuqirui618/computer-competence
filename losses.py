"""
Loss functions for the MambaOut chest X-ray multi-label task.

RGBLoss re-implements the Reference-Guided Gradient Balancing Loss
(``ClassAsyDiff``) from:
  Lin et al., "Reference-Guided Gradient Balancing Loss for Long-Tailed
  Multi-label Medical Image Classification", MICCAI 2026.
Official code: https://github.com/xup6YJ/RGB-Loss (GPL-3.0).

Only the loss formulation is used here; no medical pretraining weights or
external medical data from the official repository are included.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BinaryCrossEntropyWithLogits(nn.Module):
    def forward(self, logits, targets):
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)


class RGBLoss(nn.Module):
    """Reference-Guided Gradient Balancing Loss (ClassAsyDiff)."""

    def __init__(self, num_classes: int = 10, tn_gamma_neg: float = 3.0,
                 fp_gamma_neg: float = 3.0, ratio: float = 1.0,
                 eps: float = 1e-4):
        super().__init__()
        self.num_classes = num_classes
        self.tn_gamma_neg = tn_gamma_neg
        self.fp_gamma_neg = fp_gamma_neg
        self.ratio = ratio
        self.eps = eps

    def _pos_neg(self, x, y):
        x_i = x[y == 1]
        x_j = x[y == 0]
        if x_i.numel() == 0:
            return x.new_zeros(())
        p_neg_i = 1.0 - torch.sigmoid(x_i)
        if x_j.numel() == 0:
            log_term = -torch.log(torch.sigmoid(x_i).clamp(min=self.eps, max=1.0 - self.eps)) * (1.0 + self.ratio)
        else:
            diff = x_j.unsqueeze(0) - x_i.unsqueeze(1)
            exp_diff = torch.exp((1.0 - self.ratio) * diff)
            log_term = torch.log1p(exp_diff.sum(dim=1))
        return (log_term * p_neg_i).sum()

    def _fp_tp(self, x, y):
        x_i = x[y == 1]
        x_j = x[y == 0]
        if x_i.numel() == 0 or x_j.numel() == 0:
            zero = x.new_zeros(())
            return zero, zero
        p_j = torch.sigmoid(x_j)
        tn_mask = p_j < 0.5
        fp_mask = p_j >= 0.5
        tn_loss = x.new_zeros(())
        if tn_mask.any():
            tn_loss = (-torch.log((1.0 - p_j).clamp(min=self.eps, max=1.0 - self.eps))
                       * p_j ** self.tn_gamma_neg)[tn_mask].sum()
        fp_loss = x.new_zeros(())
        if fp_mask.any():
            diff = x_j[fp_mask].unsqueeze(0) - x_i.unsqueeze(1)
            exp_diff = torch.exp(-self.ratio * diff)
            log_term = torch.log1p(exp_diff.sum(dim=0))
            fp_loss = (log_term * (p_j[fp_mask] ** self.fp_gamma_neg)).sum()
        return tn_loss, fp_loss

    def _fp_tn(self, x):
        p = torch.sigmoid(x)
        tn_mask = p < 0.5
        fp_mask = p >= 0.5
        log_neg = torch.log((1.0 - p).clamp(min=self.eps, max=1.0 - self.eps))
        tn_loss = x.new_zeros(())
        fp_loss = x.new_zeros(())
        if not fp_mask.any():
            tn_loss = (-log_neg * (p ** self.tn_gamma_neg))[tn_mask].sum()
        elif not tn_mask.any():
            fp_loss = (-log_neg * (p ** self.fp_gamma_neg)).sum()
        else:
            tn_loss = (-log_neg * (p ** self.tn_gamma_neg))[tn_mask].sum()
            diff = x[fp_mask].unsqueeze(0) - x[tn_mask].unsqueeze(1)
            exp_diff = torch.exp(self.ratio * diff)
            log_term = torch.log1p(exp_diff.sum(dim=0))
            fp_loss = (log_term * (p[fp_mask] ** self.fp_gamma_neg)).sum()
        return tn_loss, fp_loss

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.float()
        batch = logits.shape[0]
        pos_neg = logits.new_zeros(())
        fp_tp = logits.new_zeros(())
        fp_tn = logits.new_zeros(())
        tn = logits.new_zeros(())
        for col in range(self.num_classes):
            x = logits[:, col]
            y = targets[:, col]
            if y.sum() > 0:
                pos_neg = pos_neg + self._pos_neg(x, y) / batch
                if (torch.sigmoid(x[y == 1]) >= 0.5).any():
                    z, f = self._fp_tp(x, y)
                    tn = tn + z / batch
                    fp_tp = fp_tp + f / batch
                else:
                    z, f = self._fp_tn(x)
                    tn = tn + z / batch
                    fp_tn = fp_tn + f / batch
            else:
                z, f = self._fp_tn(x)
                tn = tn + z / batch
                fp_tn = fp_tn + f / batch
        return pos_neg + fp_tn + fp_tp + tn


def build_loss(name: str = "BCE", num_classes: int = 10, **kwargs):
    """Create a loss by name. Default keeps the verified BCE baseline."""
    key = (name or "BCE").strip().upper()
    if key in ("BCE", "BCEWITHLOGITSLOSS"):
        return BinaryCrossEntropyWithLogits()
    if key in ("RGB", "RGBLOSS", "CLASSASYDIFF"):
        return RGBLoss(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unsupported loss: {name}")
