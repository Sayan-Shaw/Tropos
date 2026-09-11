"""
geometry/reverse_projection.py
===============================

Stage 3: Map refined sag/cor masks BACK onto the axial grid.

Two modes:
  - TRAINING:  `reverse_project_differentiable()` uses torch.grid_sample
               so gradients flow through the mask values back to the model.
  - EVAL/VIZ:  `reverse_project_numpy()` uses scipy.map_coordinates
               (faster, no GPU needed).

The coordinate grid is PRECOMPUTED once per patient per view and cached
as a .pt tensor. It encodes the fixed geometric mapping and is never
learned — only the mask values being sampled are trainable.
"""

from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates

import torch
import torch.nn.functional as F

from geometry.affine_utils import build_volume_affine


# ── Precompute coordinate grids ─────────────────────────

def precompute_reverse_grid(source_series, M_ax, origin_ax,
                             axial_shape, coord_mode='nifti',
                             cache_dir=None, cache_key=None):
    """
    For each axial voxel, compute its continuous (col, row, slice)
    coordinates in the source (sag/cor) series grid.

    Returns a torch tensor of shape (D_ax, H_ax, W_ax, 3) normalized
    to [-1, 1] for use with F.grid_sample, PLUS the raw unnormalized
    coords for the numpy fallback.

    The grid is a FIXED GEOMETRIC CONSTANT — never learned.

    Parameters
    ----------
    source_series : DicomSeries (sagittal or coronal)
    M_ax, origin_ax : axial affine in LPS
    axial_shape : tuple — shape of the axial volume
    coord_mode : 'nifti' or 'dicom'
    cache_dir, cache_key : optional caching

    Returns
    -------
    grid_normalized : torch.Tensor (1, D, H, W, 3), float32, for grid_sample
    grid_raw : np.ndarray (D, H, W, 3), float64, unnormalized voxel coords
    """
    tag = f"{cache_key}_rev_grid" if cache_key else None

    if cache_dir and tag:
        norm_path = Path(cache_dir) / f"{tag}_norm.pt"
        raw_path = Path(cache_dir) / f"{tag}_raw.npy"
        if norm_path.exists() and raw_path.exists():
            return torch.load(str(norm_path)), np.load(str(raw_path))

    M_src, origin_src, M_src_inv = build_volume_affine(source_series)

    # Source volume dimensions (for normalization)
    src_D = source_series.num_slices
    src_H = source_series.rows
    src_W = source_series.cols

    if coord_mode == 'nifti':
        ni, nj, nk = axial_shape
        grid_raw = np.zeros((nk, ni, nj, 3), dtype=np.float64)

        for k in range(nk):
            ii, jj = np.meshgrid(np.arange(ni, dtype=np.float64),
                                  np.arange(nj, dtype=np.float64),
                                  indexing='ij')
            vox = np.stack([ii, jj, np.full_like(ii, float(k))], axis=-1)
            world_lps = origin_ax[None, None, :] + np.einsum(
                'ij,...j->...i', M_ax, vox)
            src_vox = np.einsum('ij,...j->...i', M_src_inv,
                                world_lps - origin_src[None, None, :])
            grid_raw[k] = src_vox  # (ni, nj, 3) = (col, row, slice)
    else:
        nz, ny, nx = axial_shape
        grid_raw = np.zeros((nz, ny, nx, 3), dtype=np.float64)

        for z in range(nz):
            cc, rr = np.meshgrid(np.arange(nx, dtype=np.float64),
                                  np.arange(ny, dtype=np.float64))
            world_lps = (origin_ax[None, None, :]
                         + cc[:, :, None] * M_ax[:, 0][None, None, :]
                         + rr[:, :, None] * M_ax[:, 1][None, None, :]
                         + z * M_ax[:, 2][None, None, :])
            src_vox = np.einsum('ij,...j->...i', M_src_inv,
                                world_lps - origin_src[None, None, :])
            grid_raw[z] = src_vox

    # Normalize to [-1, 1] for F.grid_sample
    # grid_sample expects (x, y, z) in order (W, H, D) normalized
    # Source volume will be shaped (1, 1, D, H, W) = (1, 1, slices, rows, cols)
    # grid_sample grid: last dim is (x, y, z) mapping to (W, H, D)
    #   x → col  (W dimension), normalized: 2*col/(W-1) - 1
    #   y → row  (H dimension), normalized: 2*row/(H-1) - 1
    #   z → slice(D dimension), normalized: 2*slice/(D-1) - 1
    grid_norm = np.zeros_like(grid_raw, dtype=np.float32)
    # grid_raw[..., 0] = col, grid_raw[..., 1] = row, grid_raw[..., 2] = slice
    grid_norm[..., 0] = 2.0 * grid_raw[..., 0] / max(src_W - 1, 1) - 1.0  # x = col
    grid_norm[..., 1] = 2.0 * grid_raw[..., 1] / max(src_H - 1, 1) - 1.0  # y = row
    grid_norm[..., 2] = 2.0 * grid_raw[..., 2] / max(src_D - 1, 1) - 1.0  # z = slice

    grid_normalized = torch.from_numpy(grid_norm).unsqueeze(0)  # (1, D, H, W, 3)

    # Cache
    if cache_dir and tag:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        torch.save(grid_normalized, str(norm_path))
        np.save(str(raw_path), grid_raw)

    return grid_normalized, grid_raw


# ── Differentiable reverse projection (training) ────────

def reverse_project_differentiable(refined_mask_volume, grid_normalized):
    """
    Project a refined mask volume from source (sag/cor) grid back to
    the axial grid using differentiable bilinear sampling.

    Parameters
    ----------
    refined_mask_volume : torch.Tensor (1, 1, D_src, H_src, W_src)
        Soft probability mask on the source grid. Values in [0, 1].
        MUST have requires_grad or be connected to the computation graph.
    grid_normalized : torch.Tensor (1, D_ax, H_ax, W_ax, 3)
        Precomputed coordinate grid, normalized to [-1, 1].

    Returns
    -------
    recon_axial : torch.Tensor (1, 1, D_ax, H_ax, W_ax)
        Reconstructed soft mask on the axial grid.
    """
    grid = grid_normalized.to(refined_mask_volume.device)

    recon = F.grid_sample(
        refined_mask_volume,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )
    return recon


# ── Numpy reverse projection (eval/visualization) ───────

def reverse_project_numpy(pred_masks_list, source_series,
                           M_ax, origin_ax, M_ax_inv,
                           axial_shape, coord_mode='nifti',
                           grid_raw=None):
    """
    Non-differentiable reverse projection for evaluation.

    If grid_raw is provided (from precompute_reverse_grid), uses it
    directly. Otherwise computes on the fly.

    Parameters
    ----------
    pred_masks_list : list of 2D ndarray (per-slice masks)
    source_series : DicomSeries
    Others : same as precompute_reverse_grid

    Returns
    -------
    recon : ndarray, same shape as axial_shape, binary float32
    """
    # Build source volume
    src_vol = np.zeros(
        (source_series.num_slices, source_series.rows, source_series.cols),
        dtype=np.float32)
    for z, m in enumerate(pred_masks_list):
        if m.shape == (source_series.rows, source_series.cols):
            src_vol[z] = m
        else:
            from scipy.ndimage import zoom as spzoom
            zf = (source_series.rows / m.shape[0],
                  source_series.cols / m.shape[1])
            src_vol[z] = spzoom(m, zf, order=0)

    if grid_raw is not None:
        # Use precomputed grid
        recon = np.zeros(grid_raw.shape[:3], dtype=np.float32)
        for d in range(grid_raw.shape[0]):
            plane = grid_raw[d]  # (H, W, 3) = (col, row, slice)
            coords = np.array([
                plane[..., 2].ravel(),  # slice
                plane[..., 1].ravel(),  # row
                plane[..., 0].ravel(),  # col
            ])
            vals = map_coordinates(src_vol, coords, order=0,
                                   mode='constant', cval=0.0)
            recon[d] = vals.reshape(plane.shape[:2])
        return (recon > 0.5).astype(np.float32)

    # Fallback: compute on the fly
    M_src, origin_src, M_src_inv = build_volume_affine(source_series)
    recon = np.zeros(axial_shape, dtype=np.float32)

    if coord_mode == 'nifti':
        ni, nj, nk = axial_shape
        M_ax_mat = M_ax
        for k in range(nk):
            ii, jj = np.meshgrid(np.arange(ni, dtype=np.float64),
                                  np.arange(nj, dtype=np.float64), indexing='ij')
            vox = np.stack([ii, jj, np.full_like(ii, float(k))], axis=-1)
            world = origin_ax[None, None, :] + np.einsum(
                'ij,...j->...i', M_ax_mat, vox)
            sv = np.einsum('ij,...j->...i', M_src_inv,
                           world - origin_src[None, None, :])
            coords = np.array([sv[..., 2].ravel(), sv[..., 1].ravel(),
                               sv[..., 0].ravel()])
            recon[:, :, k] = map_coordinates(
                src_vol, coords, order=0, mode='constant', cval=0.0
            ).reshape(ni, nj)
    else:
        nz, ny, nx = axial_shape
        for z in range(nz):
            cc, rr = np.meshgrid(np.arange(nx, dtype=np.float64),
                                  np.arange(ny, dtype=np.float64))
            world = (origin_ax[None, None, :]
                     + cc[:, :, None] * M_ax[:, 0][None, None, :]
                     + rr[:, :, None] * M_ax[:, 1][None, None, :]
                     + z * M_ax[:, 2][None, None, :])
            sv = np.einsum('ij,...j->...i', M_src_inv,
                           world - origin_src[None, None, :])
            coords = np.array([sv[..., 2].ravel(), sv[..., 1].ravel(),
                               sv[..., 0].ravel()])
            recon[z] = map_coordinates(
                src_vol, coords, order=0, mode='constant', cval=0.0
            ).reshape(ny, nx)

    return (recon > 0.5).astype(np.float32)
