"""
diagnose_shapes.py — run once to see exact shapes at every step.
Place in Tropos root, run with PYTHONPATH as usual.
"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
os.environ["TQDM_DISABLE"] = "1"

import numpy as np
from data.dataset import ProstateXDataset

# Load first patient
ds = ProstateXDataset(
    dicom_root="/mnt/c/Users/ANT-PC/Desktop/sayan/ProstateX/PROSTATEx_Train/PROSTATEx",
    masks_root="/mnt/c/Users/ANT-PC/Desktop/sayan/ProstateX/PROSTATEx_github/PROSTATEx_masks-master",
    cache_dir="./cache", normalize="zscore")

patient = ds[0]
pid = patient.patient_id

print(f"\n{'='*60}")
print(f"  PATIENT: {pid}")
print(f"  coord_mode: {patient.coord_mode}")
print(f"  mask_vol.shape: {patient.mask_vol.shape}")
print(f"  axial_vol.shape: {patient.axial_vol.shape}")
print(f"  num_axial_slices: {patient.num_axial_slices}")
print(f"  sag_series: rows={patient.sag_series.rows} cols={patient.sag_series.cols} slices={patient.sag_series.num_slices}")
print(f"  cor_series: rows={patient.cor_series.rows} cols={patient.cor_series.cols} slices={patient.cor_series.num_slices}")
print(f"{'='*60}")

# GT masks as list
if patient.coord_mode == "nifti":
    gt_masks = [patient.mask_vol[:, :, k].astype(np.float32)
                for k in range(patient.mask_vol.shape[2])]
else:
    gt_masks = [patient.mask_vol[z].astype(np.float32)
                for z in range(patient.mask_vol.shape[0])]
print(f"\n  gt_masks: {len(gt_masks)} items, each shape {gt_masks[0].shape}")

# Axial vol transposed
if patient.coord_mode == "nifti":
    ax_vol = np.transpose(patient.axial_vol, (2, 0, 1)).copy()
else:
    ax_vol = patient.axial_vol.copy()
print(f"  ax_vol (series-like): {ax_vol.shape}")
print(f"  ax_vol[0].shape (one frame): {ax_vol[0].shape}")

# Align
n_frames = ax_vol.shape[0]
if len(gt_masks) > n_frames:
    gt_masks = gt_masks[:n_frames]
elif len(gt_masks) < n_frames:
    h, w = gt_masks[0].shape
    gt_masks += [np.zeros((h, w), dtype=np.float32)] * (n_frames - len(gt_masks))
print(f"  gt_masks aligned: {len(gt_masks)} items")

# Load model
from models.adapted_medsam2 import AdaptedMedSAM2
model = AdaptedMedSAM2(
    checkpoint_path="/mnt/c/Users/ANT-PC/Desktop/sayan/code/CorssViewVisualization/MedSam2/MedSAM2/checkpoints/MedSAM2_latest.pt",
    config_path="configs/sam2.1_hiera_t512.yaml",
    device="cuda", train_mode=False)

# Predict axial
print(f"\n  Running predict_axial_from_points...")
pred_dict = model.predict_axial_from_points(ax_vol, gt_masks, min_fg_px=15)
print(f"  pred_dict keys: {sorted(pred_dict.keys())}")
print(f"  pred_dict count: {len(pred_dict)}")

if pred_dict:
    first_key = list(pred_dict.keys())[0]
    first_mask = pred_dict[first_key]
    print(f"  pred_dict[{first_key}].shape: {first_mask.shape}")
    print(f"  pred_dict[{first_key}].dtype: {first_mask.dtype}")
    print(f"  pred_dict[{first_key}].min/max: {first_mask.min()}/{first_mask.max()}")

    # Check ALL shapes
    shapes = set(pred_dict[k].shape for k in pred_dict)
    print(f"  unique mask shapes across all slices: {shapes}")
else:
    print(f"  ⚠ pred_dict is EMPTY!")

# Build volume
from scipy.ndimage import zoom as spzoom
n_slices = patient.num_axial_slices
if patient.coord_mode == "nifti":
    h, w = patient.mask_vol.shape[0], patient.mask_vol.shape[1]
else:
    h, w = patient.mask_vol.shape[1], patient.mask_vol.shape[2]
print(f"\n  Building pred volume: n_slices={n_slices} h={h} w={w}")

vol = np.zeros((n_slices, h, w), dtype=np.float32)
for z in range(n_slices):
    if z not in pred_dict:
        continue
    m = pred_dict[z].astype(np.float32)
    print(f"    slice {z}: raw mask shape = {m.shape}", end="")
    if m.shape != (h, w):
        zf = (h / m.shape[0], w / m.shape[1])
        m = (spzoom(m, zf, order=0) > 0.5).astype(np.float32)
        print(f" → resized to {m.shape}", end="")
    vol[z] = m
    print()

if patient.coord_mode == "nifti":
    pred_vol = np.transpose(vol, (1, 2, 0))
else:
    pred_vol = vol

print(f"\n  FINAL pred_vol.shape: {pred_vol.shape}")
print(f"  mask_vol.shape:       {patient.mask_vol.shape}")
print(f"  MATCH: {pred_vol.shape == patient.mask_vol.shape}")

# Also test sag/cor predict_view
print(f"\n  Running predict_view on sagittal...")
sag_segs = model.predict_view(
    patient.sag_series, patient.sag_coarse_masks, patient.sag_boxes,
    prompt_mode='box')
if sag_segs:
    fk = list(sag_segs.keys())[0]
    print(f"  sag_segs[{fk}].shape: {sag_segs[fk].shape}")
    sag_shapes = set(sag_segs[k].shape for k in sag_segs)
    print(f"  unique sag mask shapes: {sag_shapes}")
else:
    print(f"  ⚠ sag_segs is EMPTY!")

print(f"\n{'='*60}")
print(f"  DONE. If any shape != ({h},{w}), that's the bug source.")
print(f"{'='*60}\n")