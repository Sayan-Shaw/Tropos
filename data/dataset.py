"""
data/dataset.py  (clean rewrite)
=================================
Per-patient data loader with:
  - DICOM discovery + loading
  - Standardization to target dims BEFORE flag check
  - Intensity normalization
  - Forward projection + caching
  - Reverse grid precomputation + caching
"""

from pathlib import Path
import glob
import re
import types
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import map_coordinates

from geometry.affine_utils import (
    DicomSeries, build_volume_affine, build_nifti_affine,
    nifti_affine_to_lps, LPS_TO_RAS,
    normalize_intensity,
)
from geometry.forward_projection import forward_project_mask, forward_project_image
from geometry.reverse_projection import precompute_reverse_grid
from data.prompt_utils import masks_to_boxes
from data.standardize import standardize_axial, standardize_dicom_series, TARGETS


# ── Expected dims (after standardization) ──
EXPECTED = {
    "axial":    (384, 384),
    "sagittal": (320, 320),
    "coronal":  (320, 320),
}


class ProstateXPatient:
    """
    Loads and preprocesses one patient.
    Standardization runs FIRST, flag check runs AFTER.
    """

    def __init__(self, patient_id, axial_dir, sagittal_dir, coronal_dir,
                 mask_path, ref_nifti_path=None,
                 cache_dir=None, normalize='zscore', box_pad=10):

        self.patient_id = patient_id
        self.cache_dir = Path(cache_dir) / patient_id if cache_dir else None
        self.normalize_method = normalize
        self.box_pad = box_pad
        self.dim_flags = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # ── Load DICOM (headers only for geometry; pixels lazy) ──
        self.axial_series   = DicomSeries(axial_dir, "AXIAL")
        self.sag_series     = DicomSeries(sagittal_dir, "SAGITTAL")
        self.cor_series     = DicomSeries(coronal_dir, "CORONAL")

        # ── Load mask + ref NIfTI → axial volume + affine ──
        self._load_mask_and_ref(mask_path, ref_nifti_path)

        # ── Standardize BEFORE flag check ──
        self._standardize()

        # ── Validate dims (should all pass after standardization) ──
        for name, series in [("axial", self.axial_series),
                              ("sagittal", self.sag_series),
                              ("coronal", self.cor_series)]:
            exp = EXPECTED.get(name)
            if name == "axial":
                if self.coord_mode == 'nifti':
                    actual = (self.axial_vol.shape[0], self.axial_vol.shape[1])
                else:
                    actual = (series.rows, series.cols)
            else:
                actual = (series.rows, series.cols)
            ok = (actual == exp)
            self.dim_flags[name] = {"ok": ok, "actual": actual, "expected": exp}
            if not ok:
                print(f"  ⚠ FLAG (post-std): {name} {actual} != {exp}  "
                      f"patient={patient_id}")

        # ── Normalize image volumes ──
        self.axial_vol = normalize_intensity(self.axial_vol, self.normalize_method)
        self.sag_vol   = normalize_intensity(self.sag_series.get_volume(),
                                             self.normalize_method)
        self.cor_vol   = normalize_intensity(self.cor_series.get_volume(),
                                             self.normalize_method)

        # ── Forward projection (cached) ──
        ck = patient_id if self.cache_dir else None
        cd = str(self.cache_dir) if self.cache_dir else None

        self.sag_coarse_masks = forward_project_mask(
            self.mask_vol, self.M_ax, self.origin_ax, self.M_ax_inv,
            self.sag_series, self.coord_mode, cd,
            f"{ck}_sag" if ck else None)

        self.cor_coarse_masks = forward_project_mask(
            self.mask_vol, self.M_ax, self.origin_ax, self.M_ax_inv,
            self.cor_series, self.coord_mode, cd,
            f"{ck}_cor" if ck else None)

        # ── Bounding boxes ──
        self.sag_boxes = masks_to_boxes(self.sag_coarse_masks, pad=box_pad)
        self.cor_boxes = masks_to_boxes(self.cor_coarse_masks, pad=box_pad)

        # ── Reverse grids (cached) ──
        self.sag_grid_norm, self.sag_grid_raw = precompute_reverse_grid(
            self.sag_series, self.M_ax, self.origin_ax,
            self.axial_vol.shape, self.coord_mode, cd,
            f"{ck}_sag" if ck else None)

        self.cor_grid_norm, self.cor_grid_raw = precompute_reverse_grid(
            self.cor_series, self.M_ax, self.origin_ax,
            self.axial_vol.shape, self.coord_mode, cd,
            f"{ck}_cor" if ck else None)

    # ── Internal loaders ──────────────────────────────────

    def _load_mask_and_ref(self, mask_path, ref_nifti_path):
        import nibabel as nib

        nii = nib.load(str(mask_path))
        self.mask_vol = (np.asarray(nii.dataobj).astype(np.float32) > 0.5
                         ).astype(np.float32)
        mask_affine = nii.affine.astype(np.float64)

        if ref_nifti_path is not None:
            ref_data, ref_affine, _ = build_nifti_affine(ref_nifti_path)

            if ref_data.shape != self.mask_vol.shape:
                mask_inv = np.linalg.inv(mask_affine)
                ii, jj, kk = np.meshgrid(
                    *[np.arange(s) for s in ref_data.shape], indexing='ij')
                coords_ras = np.stack([ii.ravel(), jj.ravel(), kk.ravel(),
                                       np.ones(ii.size)], axis=0)
                coords_mask = mask_inv @ (ref_affine @ coords_ras)
                self.mask_vol = (map_coordinates(
                    self.mask_vol, coords_mask[:3], order=0,
                    mode='constant', cval=0.0
                ).reshape(ref_data.shape) > 0.5).astype(np.float32)

            M_lps, origin_lps, M_lps_inv = nifti_affine_to_lps(ref_affine)
            self.M_ax = M_lps
            self.origin_ax = origin_lps
            self.M_ax_inv = M_lps_inv
            self.axial_vol = ref_data.astype(np.float32)
            self.coord_mode = 'nifti'
            self._nifti_affine = ref_affine.copy()
        else:
            self.M_ax, self.origin_ax, self.M_ax_inv = build_volume_affine(
                self.axial_series)
            self.axial_vol = self.axial_series.get_volume()
            self.coord_mode = 'dicom'
            self._nifti_affine = None

    def _standardize(self):
        """
        Resample non-standard patients to targets in-place.
        Updates axial_vol, mask_vol, M_ax, origin_ax, M_ax_inv, sag/cor series.
        """
        # ── Axial ──
        ax_tgt = TARGETS["axial"]
        if self.coord_mode == 'nifti' and self._nifti_affine is not None:
            new_img, new_msk, new_aff, changed = standardize_axial(
                self.axial_vol, self.mask_vol, self._nifti_affine,
                ax_tgt["rows"], ax_tgt["cols"], ax_tgt["row_sp"])
            if changed:
                self.axial_vol = new_img
                self.mask_vol = new_msk
                self._nifti_affine = new_aff
                self.M_ax, self.origin_ax, self.M_ax_inv = nifti_affine_to_lps(new_aff)

        # ── Sagittal (all 204 already 320x320 sp=0.5625 — no-op) ──
        sag_tgt = TARGETS["sagittal"]
        standardize_dicom_series(
            self.sag_series,
            sag_tgt["rows"], sag_tgt["cols"],
            sag_tgt["row_sp"], sag_tgt["col_sp"])

        # ── Coronal ──
        cor_tgt = TARGETS["coronal"]
        standardize_dicom_series(
            self.cor_series,
            cor_tgt["rows"], cor_tgt["cols"],
            cor_tgt["row_sp"], cor_tgt["col_sp"])

    # ── Public helpers ────────────────────────────────────

    def get_axial_slice(self, idx):
        if self.coord_mode == 'nifti':
            idx = int(np.clip(idx, 0, self.axial_vol.shape[2] - 1))
            return self.axial_vol[:, :, idx], self.mask_vol[:, :, idx]
        else:
            idx = int(np.clip(idx, 0, self.axial_vol.shape[0] - 1))
            return self.axial_vol[idx], self.mask_vol[idx]

    @property
    def num_axial_slices(self):
        return (self.axial_vol.shape[2] if self.coord_mode == 'nifti'
                else self.axial_vol.shape[0])

    def has_dimension_issues(self):
        return any(not v["ok"] for v in self.dim_flags.values()
                   if v.get("expected") is not None)


# ═══════════════════════════════════════════════════════════
# Discovery + Dataset
# ═══════════════════════════════════════════════════════════

def discover_patients(dicom_root, masks_root,
                      mask_subdir="Files/prostate/mask_prostate"):
    dicom_root = Path(dicom_root)
    masks_root = Path(masks_root)
    pattern = re.compile(r"^ProstateX-\d{4}$")
    patients = []

    for d in sorted(dicom_root.iterdir()):
        if not (d.is_dir() and pattern.match(d.name)):
            continue
        pid = d.name

        subdirs = [s for s in d.iterdir() if s.is_dir()]
        if not subdirs:
            continue
        study = max(subdirs, key=lambda s: len(list(s.iterdir())))

        def find_series(kw):
            for sd in study.iterdir():
                if sd.is_dir() and kw in sd.name.lower():
                    return sd
            return None

        axial  = find_series("t2tsetra")
        sag    = find_series("t2tsesag")
        cor    = find_series("t2tsecor")
        mask   = masks_root / mask_subdir / f"{pid}.nii.gz"
        if not mask.exists():
            mask = None

        refs = sorted(glob.glob(str(
            masks_root / "Files" / "lesions" / "Images" / "T2" /
            f"{pid}_t2_tse_tra_*.nii.gz")))
        ref = refs[0] if refs else None

        missing = []
        if axial is None: missing.append("axial")
        if sag   is None: missing.append("sagittal")
        if cor   is None: missing.append("coronal")
        if mask  is None: missing.append("mask")
        if ref   is None: missing.append("ref_nifti (axis alignment unsafe)")

        patients.append({
            "patient_id": pid,
            "axial_dir": axial, "sagittal_dir": sag, "coronal_dir": cor,
            "mask_path": mask, "ref_nifti": ref, "missing": missing,
        })
    return patients


class ProstateXDataset(Dataset):
    def __init__(self, dicom_root, masks_root, cache_dir=None,
                 normalize='zscore',
                 mask_subdir="Files/prostate/mask_prostate",
                 box_pad=10):
        all_p = discover_patients(dicom_root, masks_root, mask_subdir)
        self.patients = [p for p in all_p if not p["missing"]]
        self.skipped  = [p for p in all_p if p["missing"]]
        self.cache_dir = cache_dir
        self.normalize = normalize
        self.box_pad = box_pad
        print(f"  ProstateXDataset: {len(self.patients)} runnable, "
              f"{len(self.skipped)} skipped")
        for p in self.skipped:
            print(f"    skip {p['patient_id']}: {p['missing']}")

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        info = self.patients[idx]
        return ProstateXPatient(
            patient_id=info["patient_id"],
            axial_dir=str(info["axial_dir"]),
            sagittal_dir=str(info["sagittal_dir"]),
            coronal_dir=str(info["coronal_dir"]),
            mask_path=str(info["mask_path"]),
            ref_nifti_path=str(info["ref_nifti"]) if info["ref_nifti"] else None,
            cache_dir=self.cache_dir,
            normalize=self.normalize,
            box_pad=self.box_pad,
        )

    def get_patient_ids(self):
        return [p["patient_id"] for p in self.patients]