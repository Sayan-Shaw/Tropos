"""
losses/boundary_loss.py
========================

Signed Distance Map (SDM) weighted boundary loss.
(Kervadec et al., "Boundary loss for highly unbalanced segmentation", 2019)

    SDM_gt = EDT(GT) - EDT(1-GT)        (positive inside, negative outside)
    L_boundary = Σ(pred · SDM_gt) / Σ(|SDM_gt|)

Predictions far from the true boundary pay a large penalty proportional
to their distance from it. A prediction correct at the boundary has
near-zero boundary loss regardless of interior fill.
"""

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_sdm(binary_mask_np):
    """
    Compute signed distance map from a binary numpy mask.
    Positive inside, negative outside, zero at boundary.

    Parameters
    ----------
    binary_mask_np : ndarray, binary

    Returns
    -------
    sdm : ndarray, float32, same shape
    """
    mask = binary_mask_np > 0.5
    if mask.sum() == 0 or (~mask).sum() == 0:
        return np.zeros_like(binary_mask_np, dtype=np.float32)

    pos_dist = distance_transform_edt(mask).astype(np.float32)
    neg_dist = distance_transform_edt(~mask).astype(np.float32)
    return pos_dist - neg_dist


def boundary_loss(pred, target, precomputed_sdm=None):
    """
    Differentiable boundary loss.

    Parameters
    ----------
    pred : (B, 1, ...) torch tensor — soft prediction
    target : (B, 1, ...) torch tensor — binary GT
    precomputed_sdm : optional (B, 1, ...) torch tensor
        If provided, skip SDM computation (for efficiency when GT is fixed).

    Returns
    -------
    loss : scalar tensor
    """
    if precomputed_sdm is not None:
        sdm_tensor = precomputed_sdm.to(pred.device)
    else:
        # Compute SDM from target (detached, numpy)
        target_np = target.detach().cpu().numpy()
        sdm_list = []
        for b in range(target_np.shape[0]):
            sdm_b = compute_sdm(target_np[b, 0])
            sdm_list.append(sdm_b)
        sdm_np = np.stack(sdm_list)[:, np.newaxis]
        sdm_tensor = torch.from_numpy(sdm_np).to(pred.device)

    # Boundary loss = mean of (pred * sdm) / |sdm|
    abs_sdm = sdm_tensor.abs()
    denom = abs_sdm.sum() + 1e-8
    loss = (pred * sdm_tensor).sum() / denom

    return loss
