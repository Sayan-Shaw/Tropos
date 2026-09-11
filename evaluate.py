"""
evaluate.py
===========

Quantitative evaluation of round-trip segmentation quality.

Metrics (all computed on the axial grid):
    - Volume Dice
    - Per-slice Dice
    - Hausdorff distance 95th percentile
    - Boundary F1 (precision/recall at boundary)
    - Cross-view consistency Dice
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_score(pred, gt):
    """Volume-level Dice between two binary arrays."""
    p = (pred > 0.5).ravel()
    g = (gt > 0.5).ravel()
    inter = (p & g).sum()
    total = p.sum() + g.sum()
    return (2.0 * inter / total) if total > 0 else 1.0


def per_slice_dice(pred_vol, gt_vol, axis=0):
    """Dice per slice along given axis."""
    n = pred_vol.shape[axis]
    dices = []
    for i in range(n):
        p = np.take(pred_vol, i, axis=axis)
        g = np.take(gt_vol, i, axis=axis)
        dices.append(dice_score(p, g))
    return dices


def hausdorff_95(pred, gt):
    """95th percentile Hausdorff distance between two binary masks."""
    p = pred > 0.5
    g = gt > 0.5
    if p.sum() == 0 or g.sum() == 0:
        return float('inf')

    # Surface distances
    pred_border = p ^ (p & np.roll(p, 1, axis=0) & np.roll(p, 1, axis=1))
    gt_border = g ^ (g & np.roll(g, 1, axis=0) & np.roll(g, 1, axis=1))

    dt_gt = distance_transform_edt(~gt_border)
    dt_pred = distance_transform_edt(~pred_border)

    d_p2g = dt_gt[pred_border]
    d_g2p = dt_pred[gt_border]

    if len(d_p2g) == 0 or len(d_g2p) == 0:
        return float('inf')

    return max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95))


def boundary_f1(pred, gt, tolerance=2):
    """
    Boundary F1: precision and recall of boundary pixels within
    a tolerance distance.
    """
    p = pred > 0.5
    g = gt > 0.5
    if p.sum() == 0 and g.sum() == 0:
        return 1.0, 1.0, 1.0

    from scipy.ndimage import binary_erosion
    pred_contour = p & ~binary_erosion(p, iterations=1)
    gt_contour = g & ~binary_erosion(g, iterations=1)

    if pred_contour.sum() == 0 or gt_contour.sum() == 0:
        return 0.0, 0.0, 0.0

    dt_gt = distance_transform_edt(~gt_contour)
    dt_pred = distance_transform_edt(~pred_contour)

    precision = (dt_gt[pred_contour] <= tolerance).sum() / pred_contour.sum()
    recall = (dt_pred[gt_contour] <= tolerance).sum() / gt_contour.sum()

    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return float(precision), float(recall), float(f1)


def evaluate_round_trip(recon_vol, gt_vol, view_name=""):
    """
    Full evaluation of a round-trip reconstruction.

    Returns dict of metrics.
    """
    d = dice_score(recon_vol, gt_vol)
    hd95 = hausdorff_95(recon_vol, gt_vol)
    prec, rec, f1 = boundary_f1(recon_vol, gt_vol)

    metrics = {
        f'{view_name}dice': d,
        f'{view_name}hd95': hd95,
        f'{view_name}boundary_precision': prec,
        f'{view_name}boundary_recall': rec,
        f'{view_name}boundary_f1': f1,
    }
    return metrics
