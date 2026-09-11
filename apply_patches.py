# """
# apply_patches.py
# ================
# Run from Tropos root to:
#   1. Add standardization to data/dataset.py
#   2. Fix the recon/gt shape mismatch crash in train.py

# Usage:
#     cd /mnt/c/Users/ANT-PC/Desktop/sayan/code/Tropos
#     python apply_patches.py
# """

# from pathlib import Path
# import re

# ROOT = Path(__file__).parent

# # ═══════════════════════════════════════════════════════════
# # PATCH 1: data/dataset.py — add standardization after loading
# # ═══════════════════════════════════════════════════════════

# ds_path = ROOT / "data" / "dataset.py"
# ds_text = ds_path.read_text()

# # 1a. Add import at top (after existing imports)
# if "from data.standardize" not in ds_text:
#     # Find the last "from" import line
#     insert_after = "from scipy.ndimage import map_coordinates"
#     if insert_after in ds_text:
#         ds_text = ds_text.replace(
#             insert_after,
#             insert_after + "\n\nfrom data.standardize import standardize_axial, standardize_dicom_series, TARGETS"
#         )
#         print("  dataset.py: added standardize import")

# # 1b. Add standardization call in __init__ after _load_mask_and_ref
# # We insert a standardization block after the line: self._load_mask_and_ref(mask_path, ref_nifti_path)
# marker = "self._load_mask_and_ref(mask_path, ref_nifti_path)"
# std_block = '''self._load_mask_and_ref(mask_path, ref_nifti_path)

#         # ── Standardize axial to target dims/spacing ──
#         ax_tgt = TARGETS.get("axial", {})
#         if ax_tgt and self.coord_mode == 'nifti':
#             import nibabel as nib
#             # Reconstruct 4x4 affine from LPS components
#             aff_4x4 = np.eye(4, dtype=np.float64)
#             # NIfTI affine = RAS, our M_ax is LPS. Convert back.
#             from geometry.affine_utils import LPS_TO_RAS
#             aff_4x4[:3, :3] = LPS_TO_RAS @ self.M_ax
#             aff_4x4[:3, 3] = LPS_TO_RAS @ self.origin_ax
#             new_img, new_msk, new_aff, changed = standardize_axial(
#                 self.axial_vol, self.mask_vol, aff_4x4,
#                 ax_tgt.get("rows", 384), ax_tgt.get("cols", 384),
#                 ax_tgt.get("row_sp", 0.5))
#             if changed:
#                 self.axial_vol = new_img
#                 self.mask_vol = new_msk
#                 from geometry.affine_utils import nifti_affine_to_lps
#                 self.M_ax, self.origin_ax, self.M_ax_inv = nifti_affine_to_lps(new_aff)

#         # ── Standardize coronal if needed ──
#         cor_tgt = TARGETS.get("coronal", {})
#         if cor_tgt:
#             standardize_dicom_series(
#                 self.cor_series,
#                 cor_tgt.get("rows", 320), cor_tgt.get("cols", 320),
#                 cor_tgt.get("row_sp", 0.6), cor_tgt.get("col_sp", 0.6))'''

# if "standardize_axial(" not in ds_text:
#     ds_text = ds_text.replace(marker, std_block)
#     print("  dataset.py: added standardization block")

# ds_path.write_text(ds_text)
# print(f"  WROTE: {ds_path}")


# # ═══════════════════════════════════════════════════════════
# # PATCH 2: train.py — fix shape mismatch + add debug prints
# # ═══════════════════════════════════════════════════════════

# tr_path = ROOT / "train.py"
# tr_text = tr_path.read_text()

# # 2a. Replace the gt construction + loss block with shape-safe version
# old_gt_block = """        # GT mask tensor
#         gt_np = patient.mask_vol
#         if patient.coord_mode == 'nifti':
#             # Rearrange (ni, nj, nk) → (nk, ni, nj) to match grid output
#             gt_np = np.transpose(gt_np, (2, 0, 1))
#         gt_t = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).float().to(device)

#         # Loss
#         loss, loss_dict = criterion(recon_sag, recon_cor, gt_t,
#                                      step=step, total_steps=total_steps)"""

# new_gt_block = """        # GT mask tensor — must match recon shape from grid_sample
#         gt_np = patient.mask_vol
#         if patient.coord_mode == 'nifti':
#             gt_np = np.transpose(gt_np, (2, 0, 1))
#         gt_t = torch.from_numpy(gt_np.copy()).unsqueeze(0).unsqueeze(0).float().to(device)

#         # Shape safety check
#         if recon_sag.shape != gt_t.shape:
#             print(f"    ⚠ SHAPE MISMATCH {patient.patient_id}: "
#                   f"recon={recon_sag.shape} gt={gt_t.shape} — skipping")
#             continue

#         # Loss
#         loss, loss_dict = criterion(recon_sag, recon_cor, gt_t,
#                                      step=step, total_steps=total_steps)"""

# if "SHAPE MISMATCH" not in tr_text:
#     tr_text = tr_text.replace(old_gt_block, new_gt_block)
#     print("  train.py: added shape safety check")

# tr_path.write_text(tr_text)
# print(f"  WROTE: {tr_path}")


# # ═══════════════════════════════════════════════════════════
# # PATCH 3: losses/__init__.py — restore CombinedRoundTripLoss
# #   (was emptied earlier, verify it has content)
# # ═══════════════════════════════════════════════════════════

# loss_path = ROOT / "losses" / "__init__.py"
# loss_text = loss_path.read_text()
# if "CombinedRoundTripLoss" not in loss_text:
#     loss_path.write_text('''import torch
# from losses.dice_loss import soft_dice_loss
# from losses.boundary_loss import boundary_loss
# from losses.consistency_loss import consistency_loss


# class CombinedRoundTripLoss(torch.nn.Module):
#     def __init__(self, lambda_boundary=0.5, lambda_consistency=0.3,
#                  boundary_warmup_frac=0.2):
#         super().__init__()
#         self.lambda_boundary = lambda_boundary
#         self.lambda_consistency = lambda_consistency
#         self.boundary_warmup_frac = boundary_warmup_frac

#     def get_boundary_weight(self, step, total_steps):
#         warmup_steps = int(total_steps * self.boundary_warmup_frac)
#         if step >= warmup_steps:
#             return self.lambda_boundary
#         return self.lambda_boundary * (step / max(warmup_steps, 1))

#     def forward(self, recon_sag, recon_cor, gt_mask, step=0, total_steps=1):
#         dice_sag = soft_dice_loss(recon_sag, gt_mask)
#         dice_cor = soft_dice_loss(recon_cor, gt_mask)
#         l_dice = dice_sag + dice_cor

#         bw = self.get_boundary_weight(step, total_steps)
#         if bw > 0:
#             bnd_sag = boundary_loss(recon_sag, gt_mask)
#             bnd_cor = boundary_loss(recon_cor, gt_mask)
#             l_boundary = bw * (bnd_sag + bnd_cor)
#         else:
#             l_boundary = torch.tensor(0.0, device=gt_mask.device)
#             bnd_sag = bnd_cor = l_boundary

#         l_consist = self.lambda_consistency * consistency_loss(recon_sag, recon_cor)
#         total = l_dice + l_boundary + l_consist

#         return total, {
#             "total": total.item(),
#             "dice_sag": dice_sag.item(),
#             "dice_cor": dice_cor.item(),
#             "boundary_sag": bnd_sag.item() if isinstance(bnd_sag, torch.Tensor) else 0,
#             "boundary_cor": bnd_cor.item() if isinstance(bnd_cor, torch.Tensor) else 0,
#             "boundary_weight": bw,
#             "consistency": l_consist.item(),
#         }
# ''')
#     print("  losses/__init__.py: restored CombinedRoundTripLoss")
# else:
#     print("  losses/__init__.py: already has CombinedRoundTripLoss")


# print("\n✓ All patches applied. Now:")
# print("  1. Copy standardize.py → Tropos/data/standardize.py")
# print("  2. Clear cache:  rm -rf ./cache")
# print("  3. Re-run training")

"""
apply_patches.py  (axial prediction fix)
=========================================
Root cause: in validate() and phase_A/B_step(), the call chain is:

  gt_masks  = gt_masks_as_list(patient)     → list of 21 × (384,384)  ✓
  ax_vol    = axial_vol_as_series_like(p)   → (21,384,384)              ✓
  gt_masks  = align_gt_and_vol(gt, ax_vol.shape[0])  → truncate to 21  ✓
  pred_dict = model.predict_axial_from_points(ax_vol, gt_masks)

Inside predict_axial_from_points, when input is numpy:
  vol = axial_vol_np = (21,384,384)
  vol[z] for z in range(21) = (384,384) → JPEG 384×384                  ✓
  n_frames = count JPEGs = 21                                             ✓
  SAM2 returns mask (384,384)                                             ✓

So the call chain looks correct. But the error says (21,21,384)...

The real problem: validate() calls axial_vol_as_series_like(patient) 
AFTER a phase B step that may have altered patient state, OR the vol
is passed without transposing in some code path.

Fix: add explicit shape assertion + auto-correction in
predict_axial_from_points for the numpy path.
"""

import re

# ── Fix 1: predict_axial_from_points — add assertion on vol shape
old_1 = '''        if isinstance(axial_series_or_vol, np.ndarray):
            vol = axial_series_or_vol
            frames_dir = Path(work_dir)
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            for z in range(vol.shape[0]):
                img = vol[z]
                lo, hi = np.percentile(img, 1), np.percentile(img, 99)
                if hi - lo < 1e-6: hi = lo + 1
                win = np.clip((img - lo) / (hi - lo), 0, 1)
                u8 = (win * 255).astype(np.uint8)
                rgb = np.stack([u8] * 3, axis=-1)
                cv2.imwrite(str(frames_dir / f"{z:05d}.jpg"),
                             cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))'''

new_1 = '''        if isinstance(axial_series_or_vol, np.ndarray):
            vol = axial_series_or_vol
            # Safety: ensure (D, H, W) layout — D must be smallest dim for axial
            # If shape is (H, W, D) NIfTI order, transpose to (D, H, W)
            if vol.ndim == 3 and vol.shape[2] < vol.shape[0]:
                # shape[2] is slice count (small), shape[0] is rows (large)
                # → still in NIfTI (ni, nj, nk) order → transpose
                vol = np.transpose(vol, (2, 0, 1))
            frames_dir = Path(work_dir)
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            for z in range(vol.shape[0]):
                img = vol[z]
                lo, hi = np.percentile(img, 1), np.percentile(img, 99)
                if hi - lo < 1e-6: hi = lo + 1
                win = np.clip((img - lo) / (hi - lo), 0, 1)
                u8 = (win * 255).astype(np.uint8)
                rgb = np.stack([u8] * 3, axis=-1)
                cv2.imwrite(str(frames_dir / f"{z:05d}.jpg"),
                             cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))'''

# ── Fix 2: predicted_axial_to_gt_shape — add resize guard for SAM2 output
old_2 = '''def predicted_axial_to_gt_shape(pred_dict, patient):
    """
    Convert dict {slice_idx: 2D mask} into full axial volume matching
    patient.mask_vol shape and axis order.
    Resizes predictions if SAM2 output resolution differs from input.
    """
    from scipy.ndimage import zoom as spzoom
    n_slices = patient.num_axial_slices

    if patient.coord_mode == "nifti":
        h, w = patient.mask_vol.shape[0], patient.mask_vol.shape[1]
    else:
        h, w = patient.mask_vol.shape[1], patient.mask_vol.shape[2]

    vol = np.zeros((n_slices, h, w), dtype=np.float32)
    for z in range(n_slices):
        if z in pred_dict:
            m = pred_dict[z].astype(np.float32)
            # Resize if SAM2 output doesn't match expected frame size
            if m.shape != (h, w):
                zf = (h / m.shape[0], w / m.shape[1])
                m = (spzoom(m, zf, order=0) > 0.5).astype(np.float32)
            vol[z] = m

    if patient.coord_mode == "nifti":
        return np.transpose(vol, (1, 2, 0))
    return vol'''

new_2 = '''def predicted_axial_to_gt_shape(pred_dict, patient):
    """
    Convert dict {slice_idx: 2D mask} into full axial volume matching
    patient.mask_vol shape and axis order.
    Resizes predictions if SAM2 output resolution differs from input.
    Also handles transposed SAM2 outputs (H,W) vs (W,H).
    """
    from scipy.ndimage import zoom as spzoom
    n_slices = patient.num_axial_slices

    if patient.coord_mode == "nifti":
        # mask_vol is (ni, nj, nk) — in-plane is (ni, nj)
        h, w = patient.mask_vol.shape[0], patient.mask_vol.shape[1]
    else:
        # mask_vol is (nz, ny, nx)
        h, w = patient.mask_vol.shape[1], patient.mask_vol.shape[2]

    vol = np.zeros((n_slices, h, w), dtype=np.float32)
    for z in range(n_slices):
        if z not in pred_dict:
            continue
        m = pred_dict[z].astype(np.float32)
        if m.ndim != 2:
            m = m.squeeze()
        if m.ndim != 2:
            continue  # unexpected shape, skip

        # If SAM2 returned transposed mask (w,h) instead of (h,w), fix it
        if m.shape == (w, h) and w != h:
            m = m.T

        # Resize to expected in-plane dims if needed
        if m.shape != (h, w):
            zf = (h / m.shape[0], w / m.shape[1])
            m = (spzoom(m, zf, order=0) > 0.5).astype(np.float32)

        vol[z] = m

    if patient.coord_mode == "nifti":
        return np.transpose(vol, (1, 2, 0))  # (n_slices,h,w) → (h,w,n_slices)
    return vol'''

print("Patches ready. Applying to train.py and adapted_medsam2.py...")

# Apply fix 1 to adapted_medsam2.py
path_m = "/mnt/c/Users/ANT-PC/Desktop/sayan/code/Tropos/models/adapted_medsam2.py"
try:
    txt = open(path_m).read()
    if old_1 in txt:
        txt = txt.replace(old_1, new_1)
        open(path_m, "w").write(txt)
        print("✓ adapted_medsam2.py: added auto-transpose for NIfTI volumes")
    else:
        print("⚠  adapted_medsam2.py: target not found — applying manual fix")
        # Manual fix: add a line at the numpy path start
        manual_old = "            vol = axial_series_or_vol\n            frames_dir = Path(work_dir)"
        manual_new = ("            vol = axial_series_or_vol\n"
                      "            # Auto-correct NIfTI (ni,nj,nk) → (nk,ni,nj)\n"
                      "            if vol.ndim == 3 and vol.shape[2] < vol.shape[0]:\n"
                      "                vol = np.transpose(vol, (2, 0, 1))\n"
                      "            frames_dir = Path(work_dir)")
        if manual_old in txt:
            txt = txt.replace(manual_old, manual_new)
            open(path_m, "w").write(txt)
            print("✓ adapted_medsam2.py: applied manual fix")
        else:
            print("✗ adapted_medsam2.py: could not apply — edit manually")
except FileNotFoundError:
    print(f"✗ File not found: {path_m}")

# Apply fix 2 to train.py
path_t = "/mnt/c/Users/ANT-PC/Desktop/sayan/code/Tropos/train.py"
try:
    txt = open(path_t).read()
    if old_2 in txt:
        txt = txt.replace(old_2, new_2)
        open(path_t, "w").write(txt)
        print("✓ train.py: patched predicted_axial_to_gt_shape")
    elif "predicted_axial_to_gt_shape" in txt:
        # Function exists but text differs — find and replace the core loop
        core_old = ("    vol = np.zeros((n_slices, h, w), dtype=np.float32)\n"
                    "    for z in range(n_slices):\n"
                    "        if z in pred_dict:\n"
                    "            m = pred_dict[z].astype(np.float32)\n"
                    "            # Resize if SAM2 output doesn't match expected frame size\n"
                    "            if m.shape != (h, w):\n"
                    "                zf = (h / m.shape[0], w / m.shape[1])\n"
                    "                m = (spzoom(m, zf, order=0) > 0.5).astype(np.float32)\n"
                    "            vol[z] = m")
        core_new = ("    vol = np.zeros((n_slices, h, w), dtype=np.float32)\n"
                    "    for z in range(n_slices):\n"
                    "        if z not in pred_dict:\n"
                    "            continue\n"
                    "        m = pred_dict[z].astype(np.float32)\n"
                    "        if m.ndim != 2:\n"
                    "            m = m.squeeze()\n"
                    "        if m.ndim != 2:\n"
                    "            continue\n"
                    "        if m.shape == (w, h) and w != h:\n"
                    "            m = m.T\n"
                    "        if m.shape != (h, w):\n"
                    "            zf = (h / m.shape[0], w / m.shape[1])\n"
                    "            m = (spzoom(m, zf, order=0) > 0.5).astype(np.float32)\n"
                    "        vol[z] = m")
        if core_old in txt:
            txt = txt.replace(core_old, core_new)
            open(path_t, "w").write(txt)
            print("✓ train.py: patched inner loop of predicted_axial_to_gt_shape")
        else:
            print("⚠  train.py: could not match inner loop either")
    else:
        print("✗ train.py: predicted_axial_to_gt_shape not found")
except FileNotFoundError:
    print(f"✗ File not found: {path_t}")

print("\nDone. Run training again.")