"""
data/standardize.py
====================

In-plane standardization for ProstateX multi-view data.

Targets (from dataset analysis of 204 patients):
    Axial:    384x384, sp=0.5 mm      (10 patients need fix)
    Sagittal: 320x320, sp=0.5625 mm   (all 204 match — no-op)
    Coronal:  320x320, sp=0.6 mm      (7 patients need fix)

Strategy:
    1. Resample in-plane to target spacing  (scipy.ndimage.zoom)
    2. Center-pad/crop to target dims
    3. Shift origin by pad_offset * spacing * direction  (exact geometry)
    4. Slice axis NEVER touched

Placement: Tropos/data/standardize.py
"""

import types
import numpy as np
from scipy.ndimage import zoom as spzoom

TARGETS = {
    "axial":    {"rows": 384, "cols": 384, "row_sp": 0.5,    "col_sp": 0.5},
    "sagittal": {"rows": 320, "cols": 320, "row_sp": 0.5625, "col_sp": 0.5625},
    "coronal":  {"rows": 320, "cols": 320, "row_sp": 0.6,    "col_sp": 0.6},
}


def _resample_slice(img, src_sp, tgt_sp, order=1):
    zf = (src_sp[0] / tgt_sp[0], src_sp[1] / tgt_sp[1])
    if abs(zf[0] - 1.0) < 1e-4 and abs(zf[1] - 1.0) < 1e-4:
        return img.copy()
    return spzoom(img, zf, order=order, mode='constant', cval=0.0)


def _center_pad_crop(img, tgt_h, tgt_w):
    h, w = img.shape
    pad_t = (tgt_h - h) // 2
    pad_b = tgt_h - h - pad_t
    pad_l = (tgt_w - w) // 2
    pad_r = tgt_w - w - pad_l

    r0 = max(-pad_t, 0)
    r1 = h - max(-pad_b, 0)
    c0 = max(-pad_l, 0)
    c1 = w - max(-pad_r, 0)
    cropped = img[r0:r1, c0:c1]

    pt = max(pad_t, 0)
    pb = max(pad_b, 0)
    pl = max(pad_l, 0)
    pr = max(pad_r, 0)
    result = np.pad(cropped, ((pt, pb), (pl, pr)),
                    mode='constant', constant_values=0)
    return result[:tgt_h, :tgt_w], pad_t, pad_l


def standardize_axial(image_vol, mask_vol, affine_4x4,
                       tgt_rows=384, tgt_cols=384, tgt_sp=0.5):
    ni, nj, nk = image_vol.shape
    affine = affine_4x4.copy()
    vox_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    sp_i, sp_j = vox_sizes[0], vox_sizes[1]

    if (ni == tgt_rows and nj == tgt_cols and
            abs(sp_i - tgt_sp) < 0.002 and abs(sp_j - tgt_sp) < 0.002):
        return image_vol, mask_vol, affine, False

    dir_i = affine[:3, 0] / sp_i
    dir_j = affine[:3, 1] / sp_j
    src_sp = (sp_i, sp_j)
    tgt_sp_pair = (tgt_sp, tgt_sp)

    new_img = np.zeros((tgt_rows, tgt_cols, nk), dtype=np.float32)
    new_msk = np.zeros((tgt_rows, tgt_cols, nk), dtype=np.float32)
    pad_top = pad_left = 0

    for k in range(nk):
        ri = _resample_slice(image_vol[:, :, k], src_sp, tgt_sp_pair, order=1)
        rm = _resample_slice(mask_vol[:, :, k], src_sp, tgt_sp_pair, order=0)
        ri, pt, pl = _center_pad_crop(ri, tgt_rows, tgt_cols)
        rm, _, _ = _center_pad_crop(rm, tgt_rows, tgt_cols)
        new_img[:, :, k] = ri
        new_msk[:, :, k] = (rm > 0.5).astype(np.float32)
        if k == 0:
            pad_top, pad_left = pt, pl

    new_aff = affine.copy()
    new_aff[:3, 0] = dir_i * tgt_sp
    new_aff[:3, 1] = dir_j * tgt_sp
    new_aff[:3, 3] = (affine[:3, 3]
                       - pad_left * tgt_sp * dir_i
                       - pad_top * tgt_sp * dir_j)

    print(f"    axial std: ({ni},{nj},{nk}) sp=({sp_i:.4f},{sp_j:.4f}) -> "
          f"({tgt_rows},{tgt_cols},{nk}) sp=({tgt_sp},{tgt_sp}) "
          f"pad=({pad_top},{pad_left})")
    return new_img, new_msk, new_aff, True


def standardize_dicom_series(series, tgt_rows, tgt_cols,
                              tgt_row_sp, tgt_col_sp):
    src_row_sp = float(series.pixel_spacing[0])
    src_col_sp = float(series.pixel_spacing[1])

    if (series.rows == tgt_rows and series.cols == tgt_cols and
            abs(src_row_sp - tgt_row_sp) < 0.002 and
            abs(src_col_sp - tgt_col_sp) < 0.002):
        return False

    src_sp = (src_row_sp, src_col_sp)
    tgt_sp = (tgt_row_sp, tgt_col_sp)
    pad_top = pad_left = 0

    for i, ds in enumerate(series.datasets):
        img = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        img = img * slope + intercept

        img_r = _resample_slice(img, src_sp, tgt_sp, order=1)
        img_f, pt, pl = _center_pad_crop(img_r, tgt_rows, tgt_cols)
        ds._std_array = img_f
        ds._std_done = True
        if i == 0:
            pad_top, pad_left = pt, pl

    old_r, old_c = series.rows, series.cols
    series.pixel_spacing = np.array([tgt_row_sp, tgt_col_sp], dtype=np.float64)
    series.rows = tgt_rows
    series.cols = tgt_cols

    shift = (- pad_left * tgt_col_sp * series.column_direction
             - pad_top * tgt_row_sp * series.row_direction)
    series.positions += shift[np.newaxis, :]

    def _get_slice_std(self, z):
        z = int(np.clip(z, 0, self.num_slices - 1))
        ds = self.datasets[z]
        if hasattr(ds, '_std_done') and ds._std_done:
            return ds._std_array.copy()
        img = ds.pixel_array.astype(np.float32)
        s = float(getattr(ds, "RescaleSlope", 1.0))
        ic = float(getattr(ds, "RescaleIntercept", 0.0))
        return img * s + ic

    series.get_slice = types.MethodType(_get_slice_std, series)

    def _get_volume_std(self):
        vol = np.zeros((self.num_slices, self.rows, self.cols), dtype=np.float32)
        for z in range(self.num_slices):
            vol[z] = self.get_slice(z)
        return vol

    series.get_volume = types.MethodType(_get_volume_std, series)

    print(f"    cor/sag std: ({old_r},{old_c}) sp=({src_row_sp:.4f},{src_col_sp:.4f}) -> "
          f"({tgt_rows},{tgt_cols}) sp=({tgt_row_sp},{tgt_col_sp}) "
          f"pad=({pad_top},{pad_left})")
    return True