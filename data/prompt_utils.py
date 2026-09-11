"""
data/prompt_utils.py
====================

Extract bounding boxes from coarse projected masks for SAM2 prompting.
"""

import numpy as np


def bbox_from_mask(mask_2d, pad=10):
    """
    Compute a padded bounding box [x0, y0, x1, y1] from a binary mask.
    Returns None if mask has no foreground.
    """
    ys, xs = np.where(mask_2d > 0.5)
    if len(xs) == 0:
        return None
    h, w = mask_2d.shape
    box = np.array([
        max(0, xs.min() - pad),
        max(0, ys.min() - pad),
        min(w - 1, xs.max() + pad),
        min(h - 1, ys.max() + pad),
    ], dtype=np.float32)
    return box


def masks_to_boxes(mask_list, pad=10, min_fg_px=15):
    """
    Convert a list of per-slice masks to a list of bounding boxes.
    Returns None for slices with < min_fg_px foreground pixels.
    """
    boxes = []
    for m in mask_list:
        if m.sum() < min_fg_px:
            boxes.append(None)
        else:
            boxes.append(bbox_from_mask(m, pad))
    return boxes
