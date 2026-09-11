"""
utils/visualization.py
=======================
Visualization helpers used by train.py and inference.py.
"""
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def auto_window(image, plo=1, phi=99):
    lo, hi = np.percentile(image, plo), np.percentile(image, phi)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return np.clip((image - lo) / (hi - lo), 0, 1)


def overlay(image_2d, mask_2d, alpha=0.35, colour=(1.0, 0.2, 0.2)):
    gray = auto_window(image_2d)
    rgb = np.stack([gray] * 3, axis=-1)
    region = mask_2d > 0.5
    if region.any():
        rgb[region] = (1 - alpha) * rgb[region] + alpha * np.array(colour)
    return np.clip(rgb, 0, 1)


def get_contour(mask_2d, thickness=2):
    m = mask_2d > 0.5
    if m.sum() == 0:
        return np.zeros_like(m, dtype=bool)
    inner = binary_erosion(m, iterations=1)
    contour = m & ~inner
    if thickness > 1:
        contour = binary_dilation(contour, iterations=thickness - 1)
    return contour


def overlay_with_contour(image_2d, pred_mask, gt_mask,
                          pred_colour=(0.2, 0.5, 1.0),
                          gt_colour=(0.0, 1.0, 0.0),
                          pred_alpha=0.35, contour_thickness=2):
    gray = auto_window(image_2d)
    rgb = np.stack([gray] * 3, axis=-1)
    pred_r = pred_mask > 0.5
    if pred_r.any():
        rgb[pred_r] = ((1 - pred_alpha) * rgb[pred_r]
                       + pred_alpha * np.array(pred_colour))
    gt_c = get_contour(gt_mask, contour_thickness)
    if gt_c.any():
        rgb[gt_c] = np.array(gt_colour)
    return np.clip(rgb, 0, 1)


def save_round_trip_figure(axial_img, gt_mask, recon_sag, recon_cor,
                            axial_idx, save_path, epoch=None):
    """3-column axial round-trip comparison."""
    rgb_gt  = overlay(axial_img, gt_mask)
    rgb_sag = overlay_with_contour(axial_img, recon_sag, gt_mask,
                                    pred_colour=(0.2, 0.5, 1.0))
    rgb_cor = overlay_with_contour(axial_img, recon_cor, gt_mask,
                                    pred_colour=(1.0, 0.5, 0.2))

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor='black')
    for ax, img, title, tc in zip(
        axes,
        [rgb_gt, rgb_sag, rgb_cor],
        [f"AXIAL {axial_idx} — GT (red)",
         f"SAG inv-proj (blue) + GT contour\nfg={(recon_sag>0.5).sum()}",
         f"COR inv-proj (orange) + GT contour\nfg={(recon_cor>0.5).sum()}"],
        ["white", "#66aaff", "#ff9933"]
    ):
        ax.imshow(img, interpolation='nearest', origin='upper')
        ax.set_title(title, color=tc, fontsize=11)
        ax.axis('off')
        ax.set_facecolor('black')

    ep_str = f"  epoch {epoch}" if epoch is not None else ""
    fig.suptitle(f"Round-trip validation{ep_str}", color='white', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)