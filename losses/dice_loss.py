"""
losses/dice_loss.py
===================

Differentiable soft Dice loss for volumetric masks.

    L_dice = 1 - (2 * Σ(p·g) + ε) / (Σ(p²) + Σ(g²) + ε)
"""

import torch


def soft_dice_loss(pred, target, eps=1e-5):
    """
    Parameters
    ----------
    pred : (B, 1, ...) — soft probability mask, values in [0, 1]
    target : (B, 1, ...) — binary GT mask

    Returns
    -------
    loss : scalar tensor in [0, 1]
    """
    pred = pred.float()
    target = target.float()

    # Flatten spatial dims
    pred_flat = pred.reshape(pred.shape[0], -1)
    tgt_flat = target.reshape(target.shape[0], -1)

    intersection = (pred_flat * tgt_flat).sum(dim=1)
    sum_sq = (pred_flat ** 2).sum(dim=1) + (tgt_flat ** 2).sum(dim=1)

    dice = (2.0 * intersection + eps) / (sum_sq + eps)
    return (1.0 - dice).mean()


def soft_dice_score(pred, target, eps=1e-5):
    """Same as above but returns the SCORE (higher is better)."""
    return 1.0 - soft_dice_loss(pred, target, eps)
