"""FP32 multi-label ASL adapted from Alibaba-MIIL/ASL (MIT; LICENSE.ASL.txt).

Source: https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
Defaults: gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, focal grad disabled.
Reduction is mean over batch x classes: upstream AsymmetricLoss sum / numel.
Focal weights are detached without changing the global torch grad flag.
"""

import torch


def asymmetric_loss(logits, targets, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
    """Return scalar mean ASL for logits and binary targets shaped [B, C]."""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float()
        targets = targets.float()
        anti_targets = 1.0 - targets

        pos = torch.sigmoid(logits)
        neg = 1.0 - pos
        if clip is not None and clip > 0:
            neg = (neg + clip).clamp(max=1.0)

        loss = targets * torch.log(pos.clamp(min=eps))
        loss = loss + anti_targets * torch.log(neg.clamp(min=eps))

        if gamma_neg > 0 or gamma_pos > 0:
            pt = pos * targets + neg * anti_targets
            gamma = gamma_pos * targets + gamma_neg * anti_targets
            weights = (1.0 - pt).pow(gamma).detach()
            loss = loss * weights

        return -loss.mean()
