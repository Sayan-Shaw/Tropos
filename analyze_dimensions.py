"""
analyze_dimensions.py
======================

Standalone script: scan all ProstateX patients, report per-patient
dimensions, spacings, and slice counts for axial/sagittal/coronal.

Outputs:
    patient_dimensions.csv — one row per patient, columns:
        patient_id,
        ax_rows, ax_cols, ax_slices, ax_row_sp, ax_col_sp, ax_slice_sp,
        sag_rows, sag_cols, sag_slices, sag_row_sp, sag_col_sp, sag_slice_sp,
        cor_rows, cor_cols, cor_slices, cor_row_sp, cor_col_sp, cor_slice_sp,
        ax_flag, sag_flag, cor_flag,
        ref_nifti_shape, mask_shape, mask_fg_voxels

Usage:
    python analyze_dimensions.py \
      --dicom_root  "/mnt/c/Users/ANT-PC/Desktop/sayan/ProstateX/PROSTATEx_Train/PROSTATEx" \
      --masks_root  "/mnt/c/Users/ANT-PC/Desktop/sayan/ProstateX/PROSTATEx_github/PROSTATEx_masks-master" \
      --output      patient_dimensions.csv
"""

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pydicom


# ── Minimal DICOM geometry extraction (no pixel loading) ──

def extract_series_info(series_dir):
    """
    Read DICOM headers from a series directory.
    Returns dict with rows, cols, num_slices, pixel_spacing, slice_spacing.
    Does NOT load pixel data (fast).
    """
    series_dir = Path(series_dir)
    datasets = []
    for f in sorted(series_dir.iterdir()):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            if all(hasattr(ds, a) for a in
                   ("ImagePositionPatient", "ImageOrientationPatient",
                    "PixelSpacing", "Rows", "Columns")):
                datasets.append(ds)
        except Exception:
            pass

    if not datasets:
        return None

    ds0 = datasets[0]
    rows = int(ds0.Rows)
    cols = int(ds0.Columns)
    ps = [float(x) for x in ds0.PixelSpacing]  # [row_sp, col_sp]
    num_slices = len(datasets)

    # Compute slice spacing from IPP
    if num_slices > 1:
        iop = np.asarray(ds0.ImageOrientationPatient, dtype=np.float64)
        col_dir = iop[:3] / np.linalg.norm(iop[:3])
        row_dir = iop[3:] / np.linalg.norm(iop[3:])
        normal = np.cross(col_dir, row_dir)
        normal /= np.linalg.norm(normal)

        positions = []
        for ds in datasets:
            ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
            positions.append(float(np.dot(ipp, normal)))
        positions.sort()
        spacings = np.diff(positions)
        slice_sp = float(np.median(spacings)) if len(spacings) > 0 else 0.0
    else:
        slice_sp = float(getattr(ds0, "SpacingBetweenSlices",
                                  getattr(ds0, "SliceThickness", 0.0)))

    return {
        "rows": rows,
        "cols": cols,
        "num_slices": num_slices,
        "row_spacing": ps[0],
        "col_spacing": ps[1],
        "slice_spacing": round(slice_sp, 4),
    }


# ── Patient discovery ──

def find_series_dir(study_dir, keyword):
    for d in study_dir.iterdir():
        if d.is_dir() and keyword.lower() in d.name.lower():
            return d
    return None


def find_study_dir(patient_dir):
    subdirs = [d for d in patient_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    if len(subdirs) == 1:
        return subdirs[0]
    return max(subdirs, key=lambda s: len(list(s.iterdir())))


def main():
    p = argparse.ArgumentParser(
        description="Analyze ProstateX dataset dimensions")
    p.add_argument("--dicom_root", required=True)
    p.add_argument("--masks_root", required=True)
    p.add_argument("--mask_subdir", default="Files/prostate/mask_prostate")
    p.add_argument("--output", default="patient_dimensions.csv")
    p.add_argument("--expected_ax", default="384,384",
                   help="Expected axial dims (rows,cols)")
    p.add_argument("--expected_sag", default="320,320")
    p.add_argument("--expected_cor", default="320,320")
    args = p.parse_args()

    exp_ax = tuple(int(x) for x in args.expected_ax.split(","))
    exp_sag = tuple(int(x) for x in args.expected_sag.split(","))
    exp_cor = tuple(int(x) for x in args.expected_cor.split(","))

    dicom_root = Path(args.dicom_root)
    masks_root = Path(args.masks_root)

    # Discover patients
    pattern = re.compile(r"^ProstateX-\d{4}$")
    patient_dirs = sorted(
        d for d in dicom_root.iterdir()
        if d.is_dir() and pattern.match(d.name))

    print(f"Found {len(patient_dirs)} patient folders in {dicom_root}\n")

    # CSV setup
    fieldnames = [
        "patient_id",
        "ax_rows", "ax_cols", "ax_slices",
        "ax_row_sp", "ax_col_sp", "ax_slice_sp",
        "sag_rows", "sag_cols", "sag_slices",
        "sag_row_sp", "sag_col_sp", "sag_slice_sp",
        "cor_rows", "cor_cols", "cor_slices",
        "cor_row_sp", "cor_col_sp", "cor_slice_sp",
        "ax_flag", "sag_flag", "cor_flag",
        "ref_nifti_shape", "mask_shape", "mask_fg_voxels",
        "notes",
    ]

    rows_out = []
    flag_counts = {"axial": 0, "sagittal": 0, "coronal": 0}
    unique_dims = {"axial": set(), "sagittal": set(), "coronal": set()}

    for pd in patient_dirs:
        pid = pd.name
        study = find_study_dir(pd)
        if study is None:
            rows_out.append({"patient_id": pid, "notes": "no study folder"})
            continue

        row = {"patient_id": pid, "notes": ""}

        # ── DICOM series ──
        for view, keyword, exp in [
            ("ax", "t2tsetra", exp_ax),
            ("sag", "t2tsesag", exp_sag),
            ("cor", "t2tsecor", exp_cor),
        ]:
            sdir = find_series_dir(study, keyword)
            if sdir is None:
                row[f"{view}_rows"] = ""
                row[f"{view}_cols"] = ""
                row[f"{view}_slices"] = ""
                row[f"{view}_row_sp"] = ""
                row[f"{view}_col_sp"] = ""
                row[f"{view}_slice_sp"] = ""
                row[f"{view}_flag"] = "MISSING"
                row["notes"] += f"{view} missing; "
                continue

            info = extract_series_info(sdir)
            if info is None:
                row[f"{view}_flag"] = "UNREADABLE"
                row["notes"] += f"{view} unreadable; "
                continue

            row[f"{view}_rows"] = info["rows"]
            row[f"{view}_cols"] = info["cols"]
            row[f"{view}_slices"] = info["num_slices"]
            row[f"{view}_row_sp"] = info["row_spacing"]
            row[f"{view}_col_sp"] = info["col_spacing"]
            row[f"{view}_slice_sp"] = info["slice_spacing"]

            actual = (info["rows"], info["cols"])
            view_name = {"ax": "axial", "sag": "sagittal", "cor": "coronal"}[view]
            unique_dims[view_name].add(actual)

            if actual != exp:
                row[f"{view}_flag"] = f"⚠ {actual} != {exp}"
                flag_counts[view_name] += 1
            else:
                row[f"{view}_flag"] = "ok"

        # ── Mask ──
        mask_path = masks_root / args.mask_subdir / f"{pid}.nii.gz"
        if mask_path.exists():
            try:
                import nibabel as nib
                nii = nib.load(str(mask_path))
                mdata = np.asarray(nii.dataobj)
                row["mask_shape"] = str(mdata.shape)
                row["mask_fg_voxels"] = int((mdata > 0.5).sum())
            except Exception as e:
                row["mask_shape"] = f"error: {e}"
                row["mask_fg_voxels"] = ""
        else:
            row["mask_shape"] = "MISSING"
            row["mask_fg_voxels"] = ""

        # ── Ref NIfTI ──
        ref_pat = str(masks_root / "Files" / "lesions" / "Images" / "T2" /
                       f"{pid}_t2_tse_tra_*.nii.gz")
        refs = sorted(glob.glob(ref_pat))
        if refs:
            try:
                import nibabel as nib
                nii = nib.load(refs[0])
                row["ref_nifti_shape"] = str(nii.shape)
            except Exception as e:
                row["ref_nifti_shape"] = f"error: {e}"
        else:
            row["ref_nifti_shape"] = "MISSING"

        rows_out.append(row)

        # Progress
        if (len(rows_out)) % 20 == 0:
            print(f"  processed {len(rows_out)}/{len(patient_dirs)}...")

    # ── Write CSV ──
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\n{'='*60}")
    print(f"  RESULTS: {args.output}")
    print(f"  Total patients: {len(rows_out)}")
    print(f"{'='*60}")

    # ── Summary ──
    print(f"\n  Unique in-plane dimensions:")
    for view in ["axial", "sagittal", "coronal"]:
        dims = unique_dims[view]
        flags = flag_counts[view]
        print(f"    {view:>10s}: {sorted(dims)}  "
              f"({flags} flagged out of {len(rows_out)})")

    # Slice count ranges
    for view, key in [("axial", "ax"), ("sagittal", "sag"), ("coronal", "cor")]:
        slices = [r.get(f"{key}_slices", "") for r in rows_out
                  if r.get(f"{key}_slices", "") != ""]
        if slices:
            slices = [int(s) for s in slices]
            print(f"    {view:>10s} slices: min={min(slices)} max={max(slices)} "
                  f"median={int(np.median(slices))} "
                  f"unique={sorted(set(slices))}")

    # Spacing ranges
    print(f"\n  Pixel spacing ranges:")
    for view, key in [("axial", "ax"), ("sagittal", "sag"), ("coronal", "cor")]:
        row_sps = [float(r[f"{key}_row_sp"]) for r in rows_out
                   if r.get(f"{key}_row_sp", "") != ""]
        slice_sps = [float(r[f"{key}_slice_sp"]) for r in rows_out
                     if r.get(f"{key}_slice_sp", "") != ""]
        if row_sps:
            print(f"    {view:>10s} in-plane: "
                  f"[{min(row_sps):.4f} — {max(row_sps):.4f}] mm   "
                  f"slice: [{min(slice_sps):.4f} — {max(slice_sps):.4f}] mm")

    print(f"\n  CSV saved to: {args.output}\n")


if __name__ == "__main__":
    main()