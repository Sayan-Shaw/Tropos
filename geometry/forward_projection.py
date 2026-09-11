"""
geometry/forward_projection.py
==============================

Stage 1: Project axial GT mask (or image) onto native sag/cor grids
using DICOM affine geometry. Nearest-neighbor for masks, trilinear
for images.

Caching: results are saved as .npy under cache_dir so re-runs skip
the (expensive) per-slice world-coordinate computation.
"""

from pathlib import Path
import hashlib
import numpy as np
from scipy.ndimage import map_coordinates

from geometry.affine_utils import build_volume_affine


def _cache_key(patient_id, view_name, data_type):
    return f"{patient_id}_{view_name}_{data_type}"


def forward_project_mask(mask_vol, M_ax, origin_ax, M_ax_inv,
                          target_series, coord_mode='nifti',
                          cache_dir=None, cache_key=None):
    """
    Project a binary mask volume from the axial grid onto each slice
    of a target DICOM series (sagittal or coronal).

    Parameters
    ----------
    mask_vol : ndarray — axial mask volume
    M_ax, origin_ax, M_ax_inv : axial affine (LPS)
    target_series : DicomSeries
    coord_mode : 'nifti' or 'dicom' — axis ordering of mask_vol
    cache_dir : optional path to cache projected masks
    cache_key : optional string key for cache filename

    Returns
    -------
    list of 2D float32 arrays (one per target slice), binary {0, 1}
    """
    # Check cache
    if cache_dir and cache_key:
        cache_path = Path(cache_dir) / f"{cache_key}_fwd_mask.npy"
        if cache_path.exists():
            cached = np.load(str(cache_path))
            return [cached[i] for i in range(cached.shape[0])]

    M_tgt, origin_tgt, _ = build_volume_affine(target_series)
    nx_t, ny_t = target_series.cols, target_series.rows

    cc, rr = np.meshgrid(np.arange(nx_t, dtype=np.float64),
                          np.arange(ny_t, dtype=np.float64))
    inplane = (cc[:, :, None] * M_tgt[:, 0][None, None, :]
               + rr[:, :, None] * M_tgt[:, 1][None, None, :])

    projected = []
    for z_t in range(target_series.num_slices):
        world_lps = (origin_tgt + z_t * M_tgt[:, 2])[None, None, :] + inplane
        ax_vox = np.einsum('ij,...j->...i', M_ax_inv,
                           world_lps - origin_ax[None, None, :])

        if coord_mode == 'dicom':
            coords = np.array([ax_vox[..., 2].ravel(),
                               ax_vox[..., 1].ravel(),
                               ax_vox[..., 0].ravel()])
        else:
            coords = np.array([ax_vox[..., 0].ravel(),
                               ax_vox[..., 1].ravel(),
                               ax_vox[..., 2].ravel()])

        vals = map_coordinates(mask_vol, coords, order=0,
                               mode='constant', cval=0.0)
        projected.append((vals.reshape(ny_t, nx_t) > 0.5).astype(np.float32))

    # Save cache
    if cache_dir and cache_key:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), np.stack(projected))

    return projected


def forward_project_image(image_vol, M_ax, origin_ax, M_ax_inv,
                           target_series, coord_mode='nifti',
                           cache_dir=None, cache_key=None):
    """
    Same as forward_project_mask but for continuous image data.
    Uses trilinear interpolation (order=1).
    """
    if cache_dir and cache_key:
        cache_path = Path(cache_dir) / f"{cache_key}_fwd_img.npy"
        if cache_path.exists():
            cached = np.load(str(cache_path))
            return [cached[i] for i in range(cached.shape[0])]

    M_tgt, origin_tgt, _ = build_volume_affine(target_series)
    nx_t, ny_t = target_series.cols, target_series.rows

    cc, rr = np.meshgrid(np.arange(nx_t, dtype=np.float64),
                          np.arange(ny_t, dtype=np.float64))
    inplane = (cc[:, :, None] * M_tgt[:, 0][None, None, :]
               + rr[:, :, None] * M_tgt[:, 1][None, None, :])

    projected = []
    for z_t in range(target_series.num_slices):
        world_lps = (origin_tgt + z_t * M_tgt[:, 2])[None, None, :] + inplane
        ax_vox = np.einsum('ij,...j->...i', M_ax_inv,
                           world_lps - origin_ax[None, None, :])

        if coord_mode == 'dicom':
            coords = np.array([ax_vox[..., 2].ravel(),
                               ax_vox[..., 1].ravel(),
                               ax_vox[..., 0].ravel()])
        else:
            coords = np.array([ax_vox[..., 0].ravel(),
                               ax_vox[..., 1].ravel(),
                               ax_vox[..., 2].ravel()])

        vals = map_coordinates(image_vol, coords, order=1,
                               mode='constant', cval=0.0)
        projected.append(vals.reshape(ny_t, nx_t))

    if cache_dir and cache_key:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), np.stack(projected))

    return projected
