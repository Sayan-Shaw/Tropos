"""
geometry/affine_utils.py
========================

Foundation module: DICOM series loader, affine construction,
LPS↔RAS conversion, and coordinate grid precomputation.

Everything in this module is numpy-only (no torch dependency)
so it can be used in preprocessing/caching scripts that don't
need GPU.
"""

from pathlib import Path
import numpy as np
import pydicom

# ── Coordinate system conversion ────────────────────────
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])
RAS_TO_LPS = LPS_TO_RAS  # self-inverse


def lps_to_ras(coords):
    """Convert (N,3) or (3,) from LPS to RAS."""
    return coords * np.array([-1, -1, 1], dtype=np.float64)


# ── DICOM series loader ─────────────────────────────────

class DicomSeries:
    """
    Load a directory of DICOM slices, sort by position along the
    slice normal, and expose the geometry (IPP, IOP, spacing, normal).

    Attributes after construction:
        rows, cols          : int — in-plane pixel dimensions
        num_slices          : int — number of slices
        pixel_spacing       : (2,) array [row_spacing, col_spacing]
        column_direction    : (3,) unit vector (IOP first triple)
        row_direction       : (3,) unit vector (IOP second triple)
        normal              : (3,) unit vector (cross product)
        positions           : (num_slices, 3) IPP per slice, sorted
        mean_slice_spacing  : float — median inter-slice distance
        datasets            : list of pydicom.Dataset, sorted
    """

    def __init__(self, directory, name="UNNAMED"):
        self.directory = Path(directory)
        self.name = name
        if not self.directory.exists():
            raise FileNotFoundError(f"{name}: {self.directory}")
        self.datasets = []
        self._load()

    def _load(self):
        files = sorted(p for p in self.directory.iterdir() if p.is_file())
        for path in files:
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=False)
                if not all(hasattr(ds, a) for a in
                           ("PixelData", "ImagePositionPatient",
                            "ImageOrientationPatient", "PixelSpacing")):
                    continue
                self.datasets.append(ds)
            except Exception:
                pass
        if not self.datasets:
            raise RuntimeError(f"No usable DICOM in {self.directory}")
        self._extract_geometry()

    def _extract_geometry(self):
        ds0 = self.datasets[0]
        self.rows = int(ds0.Rows)
        self.cols = int(ds0.Columns)
        self.pixel_spacing = np.asarray(ds0.PixelSpacing, dtype=np.float64)

        iop = np.asarray(ds0.ImageOrientationPatient, dtype=np.float64)
        self.column_direction = iop[:3] / np.linalg.norm(iop[:3])
        self.row_direction = iop[3:] / np.linalg.norm(iop[3:])
        self.normal = np.cross(self.column_direction, self.row_direction)
        self.normal /= np.linalg.norm(self.normal)

        records = []
        for ds in self.datasets:
            ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
            records.append((float(np.dot(ipp, self.normal)), ipp, ds))
        records.sort(key=lambda r: r[0])

        self.positions = np.array([r[1] for r in records], dtype=np.float64)
        self.datasets = [r[2] for r in records]
        self.num_slices = len(self.datasets)

        if self.num_slices > 1:
            self.mean_slice_spacing = float(
                np.median(np.linalg.norm(
                    np.diff(self.positions, axis=0), axis=1)))
        else:
            self.mean_slice_spacing = float(
                getattr(ds0, "SpacingBetweenSlices",
                        getattr(ds0, "SliceThickness", 1.0)))

    @property
    def spacing_xyz(self):
        """(col_spacing, row_spacing, slice_spacing)"""
        return np.array([self.pixel_spacing[1],
                         self.pixel_spacing[0],
                         self.mean_slice_spacing])

    @property
    def frame_of_reference_uid(self):
        return getattr(self.datasets[0], "FrameOfReferenceUID", "?")

    def get_slice(self, z):
        z = int(np.clip(z, 0, self.num_slices - 1))
        ds = self.datasets[z]
        img = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        return img * slope + intercept

    def get_volume(self):
        """Load full volume as (num_slices, rows, cols) float32."""
        vol = np.zeros((self.num_slices, self.rows, self.cols),
                       dtype=np.float32)
        for z in range(self.num_slices):
            vol[z] = self.get_slice(z)
        return vol


# ── Affine construction ─────────────────────────────────

def build_volume_affine(series):
    """
    Build (M, origin, M_inv) from a DicomSeries such that:
        world_LPS = origin + M @ [col, row, slice]^T

    M is 3×3, mapping (col, row, slice) indices to world LPS.
    """
    origin = series.positions[0].copy()
    M = np.column_stack([
        series.column_direction * series.pixel_spacing[1],
        series.row_direction * series.pixel_spacing[0],
        series.normal * series.mean_slice_spacing,
    ])
    return M, origin, np.linalg.inv(M)


def build_nifti_affine(nifti_path):
    """
    Load NIfTI, return (data, affine_4x4, inv_affine_4x4).
    affine maps voxel (i,j,k) → RAS.
    """
    import nibabel as nib
    nii = nib.load(str(nifti_path))
    data = np.asarray(nii.dataobj).astype(np.float32)
    affine = nii.affine.astype(np.float64)
    return data, affine, np.linalg.inv(affine)


def nifti_affine_to_lps(nifti_affine_4x4):
    """
    Convert a NIfTI 4×4 affine (voxel→RAS) into (M_lps_3x3, origin_lps).
    Returns M_lps, origin_lps, M_lps_inv.
    """
    M_ras = nifti_affine_4x4[:3, :3]
    t_ras = nifti_affine_4x4[:3, 3]
    M_lps = LPS_TO_RAS @ M_ras
    origin_lps = LPS_TO_RAS @ t_ras
    return M_lps, origin_lps, np.linalg.inv(M_lps)


# ── Coordinate grid precomputation ──────────────────────

def precompute_world_coords_for_series(series):
    """
    Precompute world LPS coordinates for every pixel in every slice
    of a DICOM series.

    Returns: (num_slices, rows, cols, 3) float64 array of world_LPS.
    """
    M, origin, _ = build_volume_affine(series)
    nx, ny = series.cols, series.rows

    cc, rr = np.meshgrid(np.arange(nx, dtype=np.float64),
                          np.arange(ny, dtype=np.float64))
    # In-plane contribution (same for every slice)
    inplane = (cc[:, :, None] * M[:, 0][None, None, :]
               + rr[:, :, None] * M[:, 1][None, None, :])

    world = np.zeros((series.num_slices, ny, nx, 3), dtype=np.float64)
    for z in range(series.num_slices):
        world[z] = (origin + z * M[:, 2])[None, None, :] + inplane

    return world


def precompute_source_voxel_coords(target_world_coords, M_src_inv, origin_src):
    """
    Given target world LPS coordinates (..., 3), compute continuous
    voxel coordinates in the source grid.

    Returns: (..., 3) float64 of (v0, v1, v2) in source voxel space.
    """
    delta = target_world_coords - origin_src
    return np.einsum('ij,...j->...i', M_src_inv, delta)


# ── Dimension validation ────────────────────────────────

EXPECTED_DIMS = {
    "axial":    (384, 384),
    "sagittal": (320, 320),
    "coronal":  (320, 320),
}


def validate_inplane_dims(series, view_name, expected=None):
    """
    Check that a series has the expected in-plane dimensions.
    Returns (is_ok, actual_dims, expected_dims).
    """
    if expected is None:
        expected = EXPECTED_DIMS.get(view_name.lower(), None)
    actual = (series.rows, series.cols)
    if expected is None:
        return True, actual, None
    is_ok = (actual == expected)
    if not is_ok:
        print(f"  ⚠ FLAG: {view_name} in-plane dims {actual} != expected {expected}")
    return is_ok, actual, expected


# ── Intensity normalization ─────────────────────────────

def normalize_intensity(volume, method="zscore", clip_pct=(1, 99)):
    """
    Normalize a volume's intensity. Does NOT change shape or affine.

    Methods:
        'zscore'   — clip to [p1, p99], then (x - mean) / std
        'minmax'   — clip to [p1, p99], then scale to [0, 1]
        'none'     — return as-is

    Parameters
    ----------
    volume : ndarray, any shape
    method : str
    clip_pct : tuple (lo_percentile, hi_percentile)

    Returns
    -------
    normalized : ndarray, same shape, float32
    """
    vol = volume.astype(np.float32)
    if method == "none":
        return vol

    lo = np.percentile(vol, clip_pct[0])
    hi = np.percentile(vol, clip_pct[1])
    vol = np.clip(vol, lo, hi)

    if method == "zscore":
        mu = vol.mean()
        sigma = vol.std()
        if sigma < 1e-8:
            sigma = 1.0
        return (vol - mu) / sigma
    elif method == "minmax":
        rng = hi - lo
        if rng < 1e-8:
            rng = 1.0
        return (vol - lo) / rng
    else:
        raise ValueError(f"Unknown normalization: {method}")
