"""
losses/consistency_loss.py
===========================

Multi-view consistency: sag-derived and cor-derived axial
reconstructions should agree with each other.

    L_consistency = 1 - Dice(recon_sag, recon_cor)

This provides supervision even in regions where axial GT is
ambiguous (apex/base where slices are thick).
"""

import torch


def consistency_loss(recon_sag, recon_cor, eps=1e-5):
    """
    Parameters
    ----------
    recon_sag : (B, 1, ...) — sag predictions back-projected to axial
    recon_cor : (B, 1, ...) — cor predictions back-projected to axial

    Returns
    -------
    loss : scalar tensor
    """
    s = recon_sag.float().reshape(recon_sag.shape[0], -1)
    c = recon_cor.float().reshape(recon_cor.shape[0], -1)

    inter = (s * c).sum(dim=1)
    union = (s ** 2).sum(dim=1) + (c ** 2).sum(dim=1)

    dice = (2.0 * inter + eps) / (union + eps)
    return (1.0 - dice).mean()
