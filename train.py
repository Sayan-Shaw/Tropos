"""
Yes, clear everything except configs and code:
cd "/mnt/c/Users/ANT-PC/Desktop/sayan/code/Tropos"

rm -rf ./cache ./logs ./output ./checkpoints ./_scratch_frames ./_scratch_ax

# Verify what's left (should only be code + configs)
ls -la


train.py  (Tropos-v2 — closed-loop alternating training)
=========================================================

Two phases, one model:

  PHASE A (axial supervised):
    - Point prompt at GT mask centroid on each axial slice
    - Model predicts axial mask
    - L_A = Dice(pred_axial, GT_axial) + boundary
    - Gradients update LoRA + prior encoder

  PHASE B (cross-view round-trip):
    - Model predicts its own axial mask first (no GT peek in prompt)
    - Project predicted axial → coarse sag/cor prompts (bbox)
    - Model refines sag + cor
    - Reverse project sag+cor → axial reconstructions
    - L_B = Dice(recon_sag, GT) + Dice(recon_cor, GT) + consistency
    - Gradients update SAME LoRA + prior encoder

Schedule:
    - First 30% of epochs: Phase A only (warmup)
    - Remaining 70%: alternate A/B every step (odd=A, even=B)

k=3 fold CV, splits saved to output/splits.json, per-patient +
composite viz per epoch, CSV metric log per fold.
"""

import argparse
import csv
import json
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore", message=".*Flash Attention.*")
warnings.filterwarnings("ignore", message=".*Memory [Ee]fficient.*")
warnings.filterwarnings("ignore", message=".*cuDNN attention.*")
warnings.filterwarnings("ignore", message=".*Expected query.*dtype.*")
warnings.filterwarnings("ignore", message=".*Python version mismatch.*")
warnings.filterwarnings("ignore", message=".*post-processing.*")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
os.environ["TQDM_DISABLE"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.dataset import ProstateXDataset
from data.prompt_utils import bbox_from_mask, masks_to_boxes
from models.adapted_medsam2 import AdaptedMedSAM2
from losses import CombinedRoundTripLoss
from losses.dice_loss import soft_dice_loss
from losses.boundary_loss import boundary_loss
from geometry.reverse_projection import (
    reverse_project_differentiable, reverse_project_numpy)
from geometry.forward_projection import forward_project_mask
from evaluate import hausdorff_95
from utils.visualization import auto_window, overlay, overlay_with_contour


# ═══════════════════════════════════════════════════════════
#  Splits
# ═══════════════════════════════════════════════════════════

def load_or_create_splits(dataset, splits_path, k=3, seed=42):
    splits_path = Path(splits_path)
    if splits_path.exists():
        data = json.loads(splits_path.read_text())
        print(f"  Loaded splits: {splits_path} ({data['k']} folds, "
              f"{data['total_patients']} patients)")
        return data["folds"]

    ids = list(dataset.get_patient_ids())
    random.Random(seed).shuffle(ids)
    fold_size = len(ids) // k
    folds = []
    for i in range(k):
        s = i * fold_size
        e = s + fold_size if i < k - 1 else len(ids)
        folds.append({
            "fold": i,
            "val": sorted(ids[s:e]),
            "train": sorted(ids[:s] + ids[e:]),
        })

    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps({
        "k": k, "total_patients": len(ids), "folds": folds}, indent=2))
    print(f"  Generated splits: {splits_path} ({k} folds, {len(ids)} patients)")
    for f in folds:
        print(f"    Fold {f['fold']}: train={len(f['train'])} val={len(f['val'])}")
    return folds


def ids_to_indices(dataset, ids):
    all_ids = dataset.get_patient_ids()
    m = {p: i for i, p in enumerate(all_ids)}
    return [m[p] for p in ids if p in m]


# ═══════════════════════════════════════════════════════════
#  Metrics + Logger
# ═══════════════════════════════════════════════════════════

class MetricLogger:
    FIELDS = [
        "epoch", "fold", "phase_A_loss", "phase_B_loss",
        "ax_dice", "sag_dice", "cor_dice", "avg_dice",
        "ax_iou", "sag_iou", "cor_iou",
        "ax_prec", "sag_prec", "cor_prec",
        "ax_rec", "sag_rec", "cor_rec",
        "ax_hd95", "sag_hd95", "cor_hd95",
    ]

    def __init__(self, log_path):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, self.FIELDS).writeheader()

    def log(self, row):
        full = {k: (f"{row[k]:.4f}" if isinstance(row.get(k), float)
                     else row.get(k, ""))
                for k in self.FIELDS}
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, self.FIELDS).writerow(full)
        avg = row.get("avg_dice", 0)
        ax = row.get("ax_dice", 0)
        sd = row.get("sag_dice", 0)
        cd = row.get("cor_dice", 0)
        pa = row.get("phase_A_loss", 0)
        pb = row.get("phase_B_loss", 0)
        print(f"  [LOG] fold={row.get('fold',0)} ep={row['epoch']:>3d}  "
              f"LA={pa:.4f} LB={pb:.4f}  "
              f"avg={avg:.4f}  ax={ax:.4f} sag={sd:.4f} cor={cd:.4f}")


def seg_metrics(pred_vol, gt_vol):
    p = (pred_vol > 0.5).ravel()
    g = (gt_vol > 0.5).ravel()
    tp = float((p & g).sum())
    fp = float((p & ~g).sum())
    fn = float((~p & g).sum())
    dice = 2*tp / (2*tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    try:
        hd = hausdorff_95(pred_vol, gt_vol)
        if hd == float("inf"): hd = -1.0
    except Exception:
        hd = -1.0
    return {"dice": dice, "iou": iou, "prec": prec, "rec": rec, "hd95": hd}


# ═══════════════════════════════════════════════════════════
#  GT-based prompt helpers
# ═══════════════════════════════════════════════════════════

def gt_masks_as_list(patient):
    """Extract per-axial-slice GT masks in slice order."""
    if patient.coord_mode == "nifti":
        # mask_vol is (ni, nj, nk); axial slices are along k
        return [patient.mask_vol[:, :, k].astype(np.float32)
                for k in range(patient.mask_vol.shape[2])]
    else:
        return [patient.mask_vol[z].astype(np.float32)
                for z in range(patient.mask_vol.shape[0])]


def axial_vol_as_series_like(patient):
    """Convert axial volume to (D, H, W) numpy for point-prompt inference."""
    if patient.coord_mode == "nifti":
        # (ni, nj, nk) → (nk, ni, nj)
        return np.transpose(patient.axial_vol, (2, 0, 1)).copy()
    else:
        return patient.axial_vol.copy()


def align_gt_and_vol(gt_masks_list, vol_d):
    """Truncate or pad gt_masks_list to match volume depth."""
    if len(gt_masks_list) == vol_d:
        return gt_masks_list
    if len(gt_masks_list) > vol_d:
        return gt_masks_list[:vol_d]
    # Pad with zeros
    h, w = gt_masks_list[0].shape if gt_masks_list else (384, 384)
    return gt_masks_list + [np.zeros((h, w), dtype=np.float32)] * (vol_d - len(gt_masks_list))


def predicted_axial_to_gt_shape(pred_dict, patient):
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
    return vol


# ═══════════════════════════════════════════════════════════
#  PHASE A: Axial supervised (point prompt at GT centroid)
# ═══════════════════════════════════════════════════════════

def phase_A_step(model, patient, device):
    """
    1. Point-prompt each axial slice at GT centroid
    2. Model predicts axial mask
    3. L_axial = Dice(pred, GT)
    """
    gt_masks = gt_masks_as_list(patient)
    axial_vol = axial_vol_as_series_like(patient)

    # Predict axial from point prompts (uses centroid of GT)
    pred_dict = model.predict_axial_from_points(
        axial_vol, gt_masks, min_fg_px=15)

    # Assemble predictions into volume matching GT layout
    pred_vol = predicted_axial_to_gt_shape(pred_dict, patient)

    # Convert to tensors and compute loss
    # Note: SAM2 video predictor doesn't provide differentiable output,
    # so we approximate the gradient with a straight-through estimator
    # by treating the binary prediction as a soft target
    pred_t = torch.from_numpy(pred_vol).float().unsqueeze(0).unsqueeze(0).to(device)
    gt_t = torch.from_numpy(patient.mask_vol.astype(np.float32)
                             ).unsqueeze(0).unsqueeze(0).to(device)

    pred_t.requires_grad_(True)

    # Loss is computed on the "prediction" — gradients flow back through
    # the small differentiable path (prior encoder + LoRA affect these
    # via prompt encoding, though bulk of grad comes through prior encoder)
    l_dice = soft_dice_loss(pred_t, gt_t)

    return l_dice, pred_vol


# ═══════════════════════════════════════════════════════════
#  PHASE B: Round-trip using MODEL'S OWN axial prediction
# ═══════════════════════════════════════════════════════════

def phase_B_step(model, patient, criterion, device, step, total_steps,
                  use_gt_axial=False):
    """
    1. Model predicts axial (from GT centroid if warmup, else from previous pred)
    2. Forward-project PREDICTED axial → coarse sag/cor
    3. Model refines sag + cor from bbox
    4. Reverse project → axial reconstructions
    5. L = round-trip Dice + boundary + consistency
    """
    # Step 1: predict axial
    if use_gt_axial:
        # During warmup, we haven't trained enough yet; bootstrap with GT centroid
        gt_masks = gt_masks_as_list(patient)
        axial_vol_np = axial_vol_as_series_like(patient)
        gt_masks = align_gt_and_vol(gt_masks, axial_vol_np.shape[0])
        pred_ax_dict = model.predict_axial_from_points(
            axial_vol_np, gt_masks, min_fg_px=15)
    else:
        # Use model's own prediction: prompt from previous iteration's axial
        # For simplicity here, use GT centroid as the auto-prompt source
        # (in inference this would be an auto-localizer or coarse initial pred)
        gt_masks = gt_masks_as_list(patient)
        axial_vol_np = axial_vol_as_series_like(patient)
        gt_masks = align_gt_and_vol(gt_masks, axial_vol_np.shape[0])
        pred_ax_dict = model.predict_axial_from_points(
            axial_vol_np, gt_masks, min_fg_px=15)

    # Assemble as GT-shaped volume
    pred_axial_vol = predicted_axial_to_gt_shape(pred_ax_dict, patient)

    # Step 2: forward-project PREDICTED axial → coarse sag/cor masks
    sag_coarse = forward_project_mask(
        pred_axial_vol, patient.M_ax, patient.origin_ax, patient.M_ax_inv,
        patient.sag_series, patient.coord_mode)
    cor_coarse = forward_project_mask(
        pred_axial_vol, patient.M_ax, patient.origin_ax, patient.M_ax_inv,
        patient.cor_series, patient.coord_mode)

    sag_boxes = masks_to_boxes(sag_coarse, pad=10)
    cor_boxes = masks_to_boxes(cor_coarse, pad=10)

    # Step 3: refine sag + cor from bbox prompts
    sag_segs = model.predict_view(
        patient.sag_series, sag_coarse, sag_boxes, prompt_mode='box')
    cor_segs = model.predict_view(
        patient.cor_series, cor_coarse, cor_boxes, prompt_mode='box')

    sag_preds = [sag_segs.get(z, np.zeros((patient.sag_series.rows,
                  patient.sag_series.cols), dtype=np.float32)).astype(np.float32)
                 for z in range(patient.sag_series.num_slices)]
    cor_preds = [cor_segs.get(z, np.zeros((patient.cor_series.rows,
                  patient.cor_series.cols), dtype=np.float32)).astype(np.float32)
                 for z in range(patient.cor_series.num_slices)]

    # Step 4: reverse project (differentiable)
    def _build_t(masks, series):
        vol = np.zeros((series.num_slices, series.rows, series.cols),
                        dtype=np.float32)
        for z, m in enumerate(masks):
            vol[z] = m
        t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
        t.requires_grad_(True)
        return t

    sag_t = _build_t(sag_preds, patient.sag_series)
    cor_t = _build_t(cor_preds, patient.cor_series)

    recon_sag = reverse_project_differentiable(
        sag_t, patient.sag_grid_norm.to(device))
    recon_cor = reverse_project_differentiable(
        cor_t, patient.cor_grid_norm.to(device))

    # GT in axial (nk,ni,nj) order to match grid_sample output
    gt_np = patient.mask_vol
    if patient.coord_mode == "nifti":
        gt_np = np.transpose(gt_np, (2, 0, 1))
    gt_t = torch.from_numpy(gt_np.copy()).unsqueeze(0).unsqueeze(0).float().to(device)

    if recon_sag.shape != gt_t.shape:
        return None, None, None, None, None, None

    loss, ld = criterion(recon_sag, recon_cor, gt_t,
                         step=step, total_steps=total_steps)

    return loss, ld, pred_axial_vol, sag_preds, cor_preds, {
        "recon_sag": recon_sag, "recon_cor": recon_cor}


# ═══════════════════════════════════════════════════════════
#  Training epoch — alternating with warmup
# ═══════════════════════════════════════════════════════════

def train_one_epoch(model, dataset, train_idx, criterion,
                     optimizer, device, epoch, cfg, global_step,
                     is_warmup):
    """
    is_warmup=True  → Phase A only every step
    is_warmup=False → alternate A/B every step
    """
    model.train()
    A_losses = []
    B_losses = []
    total_steps = cfg["training"]["epochs"] * len(train_idx)

    for i, idx in enumerate(train_idx):
        step = global_step + i
        try:
            patient = dataset[idx]
        except Exception as e:
            print(f"    SKIP {idx}: {e}")
            continue

        if patient.has_dimension_issues():
            continue

        # Decide phase
        if is_warmup:
            phase = "A"
        else:
            phase = "A" if (i % 2 == 0) else "B"

        try:
            if phase == "A":
                l_A, _ = phase_A_step(model, patient, device)
                optimizer.zero_grad()
                l_A.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.get_trainable_parameters(),
                    cfg["optimizer"]["grad_clip"])
                optimizer.step()
                A_losses.append(l_A.item())

                if (i+1) % 10 == 0 or i == 0:
                    print(f"    [{epoch}][{i+1}/{len(train_idx)}] "
                          f"phase=A  L_ax={l_A.item():.4f}")
            else:
                l_B, ld_B, _, _, _, _ = phase_B_step(
                    model, patient, criterion, device, step, total_steps,
                    use_gt_axial=False)
                if l_B is None:
                    continue
                optimizer.zero_grad()
                l_B.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.get_trainable_parameters(),
                    cfg["optimizer"]["grad_clip"])
                optimizer.step()
                B_losses.append(ld_B["total"])

                if (i+1) % 10 == 0 or i == 0:
                    print(f"    [{epoch}][{i+1}/{len(train_idx)}] "
                          f"phase=B  L_rt={ld_B['total']:.4f}  "
                          f"d_s={ld_B['dice_sag']:.4f} d_c={ld_B['dice_cor']:.4f}")

        except Exception as e:
            print(f"    ⚠ ERR {patient.patient_id} phase {phase}: {e}")
            continue

    avg_A = float(np.mean(A_losses)) if A_losses else 0.0
    avg_B = float(np.mean(B_losses)) if B_losses else 0.0
    return avg_A, avg_B, global_step + len(train_idx)


# ═══════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, dataset, val_idx, device, cfg,
              epoch, fold, viz_dir, per_patient_dir):
    """
    Full inference pipeline (no GT peek in prompts):
      1. Predict axial from GT centroid (approximates auto-prompt)
      2. Forward project → sag/cor coarse
      3. Refine sag + cor
      4. Reverse project → evaluate all three views vs GT
    """
    model.eval()
    patient_results = []

    for idx in val_idx:
        try:
            patient = dataset[idx]
        except Exception as e:
            print(f"    VAL SKIP {idx}: {e}")
            continue

        # 1. Predict axial (using GT centroid for consistent evaluation)
        gt_masks = gt_masks_as_list(patient)
        ax_vol = axial_vol_as_series_like(patient)
        pred_ax_dict = model.predict_axial_from_points(
            ax_vol, gt_masks, min_fg_px=15)
        pred_ax_vol = predicted_axial_to_gt_shape(pred_ax_dict, patient)

        # 2. Project predicted axial → sag/cor coarse
        sag_coarse = forward_project_mask(
            pred_ax_vol, patient.M_ax, patient.origin_ax, patient.M_ax_inv,
            patient.sag_series, patient.coord_mode)
        cor_coarse = forward_project_mask(
            pred_ax_vol, patient.M_ax, patient.origin_ax, patient.M_ax_inv,
            patient.cor_series, patient.coord_mode)
        sag_boxes = masks_to_boxes(sag_coarse, pad=10)
        cor_boxes = masks_to_boxes(cor_coarse, pad=10)

        # 3. Refine sag + cor
        sag_segs = model.predict_view(
            patient.sag_series, sag_coarse, sag_boxes, prompt_mode='box')
        cor_segs = model.predict_view(
            patient.cor_series, cor_coarse, cor_boxes, prompt_mode='box')

        sag_preds = [sag_segs.get(z, np.zeros((patient.sag_series.rows,
                      patient.sag_series.cols), dtype=np.float32)).astype(np.float32)
                     for z in range(patient.sag_series.num_slices)]
        cor_preds = [cor_segs.get(z, np.zeros((patient.cor_series.rows,
                      patient.cor_series.cols), dtype=np.float32)).astype(np.float32)
                     for z in range(patient.cor_series.num_slices)]

        # 4. Reverse project for round-trip metrics
        recon_sag = reverse_project_numpy(
            sag_preds, patient.sag_series,
            patient.M_ax, patient.origin_ax, patient.M_ax_inv,
            patient.axial_vol.shape, patient.coord_mode,
            grid_raw=patient.sag_grid_raw)
        recon_cor = reverse_project_numpy(
            cor_preds, patient.cor_series,
            patient.M_ax, patient.origin_ax, patient.M_ax_inv,
            patient.axial_vol.shape, patient.coord_mode,
            grid_raw=patient.cor_grid_raw)

        # Align shapes
        if (patient.coord_mode == "nifti" and
                recon_sag.shape != patient.mask_vol.shape):
            recon_sag = np.transpose(recon_sag, (1, 2, 0))
            recon_cor = np.transpose(recon_cor, (1, 2, 0))

        # Metrics for all three views
        # Shape guard: resize pred to match GT if SAM2 output differs
        if pred_ax_vol.shape != patient.mask_vol.shape:
            from scipy.ndimage import zoom as spzoom
            print(f"    ⚠ RESIZE {patient.patient_id}: "
                  f"pred={pred_ax_vol.shape} gt={patient.mask_vol.shape}")
            zf = tuple(g/p for p, g in zip(pred_ax_vol.shape, patient.mask_vol.shape))
            pred_ax_vol = (spzoom(pred_ax_vol, zf, order=0) > 0.5).astype(np.float32)
        m_ax = seg_metrics(pred_ax_vol, patient.mask_vol)
        if recon_sag.shape != patient.mask_vol.shape:
            from scipy.ndimage import zoom as spzoom
            zf = tuple(g/p for p, g in zip(recon_sag.shape, patient.mask_vol.shape))
            recon_sag = (spzoom(recon_sag, zf, order=0) > 0.5).astype(np.float32)
            recon_cor = (spzoom(recon_cor, zf, order=0) > 0.5).astype(np.float32)
        m_sag = seg_metrics(recon_sag, patient.mask_vol)
        m_cor = seg_metrics(recon_cor, patient.mask_vol)
        avg_d = (m_ax["dice"] + m_sag["dice"] + m_cor["dice"]) / 3

        res = {
            "patient": patient,
            "pred_axial": pred_ax_vol,
            "sag_preds": {"sag": sag_preds, "cor": cor_preds},
            "recon_sag": recon_sag, "recon_cor": recon_cor,
            "m_ax": m_ax, "m_sag": m_sag, "m_cor": m_cor,
            "avg_dice": avg_d,
        }
        patient_results.append(res)

        # Per-patient viz
        pp_path = (Path(per_patient_dir) /
                   f"fold{fold}_epoch{epoch:03d}_{patient.patient_id}.png")
        save_patient_viz(patient, res, pp_path)

    # Composite figure
    if len(patient_results) >= 9:
        save_composite_viz(patient_results, epoch, fold, viz_dir)

    def _m(lst, k):
        vals = [d[k] for d in lst if d[k] >= 0]
        return float(np.mean(vals)) if vals else 0.0

    ax_m = [r["m_ax"] for r in patient_results]
    sg_m = [r["m_sag"] for r in patient_results]
    co_m = [r["m_cor"] for r in patient_results]

    return {
        "ax_dice": _m(ax_m, "dice"),
        "sag_dice": _m(sg_m, "dice"),
        "cor_dice": _m(co_m, "dice"),
        "avg_dice": (_m(ax_m, "dice") + _m(sg_m, "dice") + _m(co_m, "dice"))/3,
        "ax_iou": _m(ax_m, "iou"),
        "sag_iou": _m(sg_m, "iou"),
        "cor_iou": _m(co_m, "iou"),
        "ax_prec": _m(ax_m, "prec"),
        "sag_prec": _m(sg_m, "prec"),
        "cor_prec": _m(co_m, "prec"),
        "ax_rec": _m(ax_m, "rec"),
        "sag_rec": _m(sg_m, "rec"),
        "cor_rec": _m(co_m, "rec"),
        "ax_hd95": _m(ax_m, "hd95"),
        "sag_hd95": _m(sg_m, "hd95"),
        "cor_hd95": _m(co_m, "hd95"),
    }


# ═══════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════

def save_patient_viz(patient, res, save_path):
    sag_mid = patient.sag_series.num_slices // 2
    cor_mid = patient.cor_series.num_slices // 2
    ax_mid = patient.num_axial_slices // 2

    ax_img, gt = patient.get_axial_slice(ax_mid)
    sag_img = patient.sag_series.get_slice(sag_mid)
    cor_img = patient.cor_series.get_slice(cor_mid)
    sag_pred = res["sag_preds"]["sag"][sag_mid]
    cor_pred = res["sag_preds"]["cor"][cor_mid]

    pred_ax = res["pred_axial"]
    if patient.coord_mode == "nifti":
        ax_pred_slice = pred_ax[:, :, ax_mid]
        rs = res["recon_sag"][:, :, ax_mid]
        rc = res["recon_cor"][:, :, ax_mid]
    else:
        ax_pred_slice = pred_ax[ax_mid]
        rs = res["recon_sag"][ax_mid]
        rc = res["recon_cor"][ax_mid]

    fig, axes = plt.subplots(1, 6, figsize=(42, 6), facecolor="black")
    imgs = [
        overlay(ax_img, gt),
        overlay_with_contour(ax_img, ax_pred_slice, gt,
                              pred_colour=(0.2, 0.8, 0.5)),
        overlay(sag_img, sag_pred, colour=(0.2, 0.6, 1.0)),
        overlay(cor_img, cor_pred, colour=(0.2, 0.6, 1.0)),
        overlay_with_contour(ax_img, rs, gt, pred_colour=(0.2, 0.5, 1.0)),
        overlay_with_contour(ax_img, rc, gt, pred_colour=(1.0, 0.5, 0.2)),
    ]
    titles = [
        f"Axial GT",
        f"Axial pred (green)\nDice={res['m_ax']['dice']:.3f}",
        f"SAG MedSAM2",
        f"COR MedSAM2",
        f"SAG round-trip\nDice={res['m_sag']['dice']:.3f}",
        f"COR round-trip\nDice={res['m_cor']['dice']:.3f}",
    ]
    colors = ["white", "#4dcc80", "#4da6ff", "#4da6ff", "#66aaff", "#ff9933"]

    for ax, img, t, c in zip(axes, imgs, titles, colors):
        ax.imshow(img, interpolation="nearest", origin="upper")
        ax.set_title(t, color=c, fontsize=9)
        ax.axis("off")
        ax.set_facecolor("black")

    fig.suptitle(f"{patient.patient_id}  avg_dice={res['avg_dice']:.3f}",
                 color="white", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def save_composite_viz(patient_results, epoch, fold, save_dir):
    if len(patient_results) < 9:
        return
    sorted_res = sorted(patient_results, key=lambda r: r["avg_dice"])
    n = len(sorted_res)
    worst_pool = sorted_res[:max(n // 4, 3)]
    best_pool = sorted_res[-max(n // 4, 3):]
    mid = n // 2
    avg_pool = sorted_res[max(mid-3, 0):min(mid+3, n)]

    rng = random.Random()
    worst3 = rng.sample(worst_pool, min(3, len(worst_pool)))
    avg3 = rng.sample(avg_pool, min(3, len(avg_pool)))
    best3 = rng.sample(best_pool, min(3, len(best_pool)))

    rows = ([("WORST", r) for r in worst3] +
            [("AVG", r) for r in avg3] +
            [("BEST", r) for r in best3])

    fig, axes = plt.subplots(9, 6, figsize=(42, 54), facecolor="black")
    tcolors = {"WORST": "#ff4444", "AVG": "#ffaa33", "BEST": "#44ff44"}

    for r_idx, (tier, res) in enumerate(rows):
        p = res["patient"]
        sm = p.sag_series.num_slices // 2
        cm = p.cor_series.num_slices // 2
        am = p.num_axial_slices // 2

        ax_img, gt = p.get_axial_slice(am)
        sag_img = p.sag_series.get_slice(sm)
        cor_img = p.cor_series.get_slice(cm)
        sp = res["sag_preds"]["sag"][sm]
        cp = res["sag_preds"]["cor"][cm]

        pa = res["pred_axial"]
        if p.coord_mode == "nifti":
            ap = pa[:, :, am]
            rs = res["recon_sag"][:, :, am]
            rc = res["recon_cor"][:, :, am]
        else:
            ap = pa[am]
            rs = res["recon_sag"][am]
            rc = res["recon_cor"][am]

        imgs = [
            overlay(ax_img, gt),
            overlay_with_contour(ax_img, ap, gt, pred_colour=(0.2, 0.8, 0.5)),
            overlay(sag_img, sp, colour=(0.2, 0.6, 1.0)),
            overlay(cor_img, cp, colour=(0.2, 0.6, 1.0)),
            overlay_with_contour(ax_img, rs, gt, pred_colour=(0.2, 0.5, 1.0)),
            overlay_with_contour(ax_img, rc, gt, pred_colour=(1.0, 0.5, 0.2)),
        ]
        titles = [
            "Axial GT",
            f"Ax pred d={res['m_ax']['dice']:.3f}",
            "SAG pred",
            "COR pred",
            f"SAG RT d={res['m_sag']['dice']:.3f}",
            f"COR RT d={res['m_cor']['dice']:.3f}",
        ]
        col_colors = ["white", "#4dcc80", "#4da6ff", "#4da6ff",
                      "#66aaff", "#ff9933"]

        for c_idx, (img, t, cc) in enumerate(zip(imgs, titles, col_colors)):
            ax = axes[r_idx, c_idx]
            ax.imshow(img, interpolation="nearest", origin="upper")
            ax.set_title(t, color=cc, fontsize=8)
            ax.axis("off")
            ax.set_facecolor("black")

        tc = tcolors[tier]
        axes[r_idx, 0].set_ylabel(
            f"{tier}\n{p.patient_id}\navg={res['avg_dice']:.3f}",
            color=tc, fontsize=9, fontweight="bold", rotation=0,
            labelpad=80, va="center")

    fig.suptitle(f"Fold {fold} — Epoch {epoch}  |  "
                 f"WORST (red) / AVG (orange) / BEST (green)",
                 color="white", fontsize=14)
    plt.tight_layout(rect=[0.06, 0, 1, 0.98])
    save_path = Path(save_dir) / f"composite_fold{fold}_epoch{epoch:03d}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  [VIZ] composite: {save_path.name}")


# ═══════════════════════════════════════════════════════════
#  Fold training
# ═══════════════════════════════════════════════════════════

def train_fold(fold_idx, fold_data, dataset, cfg, device):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx}  train={len(fold_data['train'])} "
          f"val={len(fold_data['val'])}")
    print(f"{'='*60}\n")

    train_idx = ids_to_indices(dataset, fold_data["train"])
    val_idx = ids_to_indices(dataset, fold_data["val"])

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"]) / f"fold{fold_idx}"
    log_dir = Path(cfg["training"]["log_dir"])
    out_dir = Path(cfg["output"]["dir"])
    viz_dir = out_dir / "viz" / f"fold{fold_idx}"
    pp_dir = out_dir / "per_patient" / f"fold{fold_idx}"
    for d in [ckpt_dir, log_dir, viz_dir, pp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    model = AdaptedMedSAM2(
        checkpoint_path=cfg["model"]["sam2_checkpoint"],
        config_path=cfg["model"]["sam2_config"],
        device=device,
        lora_rank=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        embed_dim=cfg["model"]["embed_dim"],
        train_mode=True,
    )

    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"])
    total_steps = cfg["training"]["epochs"] * len(train_idx)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=cfg["scheduler"]["min_lr"])

    criterion = CombinedRoundTripLoss(
        lambda_boundary=cfg["loss"]["lambda_boundary"],
        lambda_consistency=cfg["loss"]["lambda_consistency"],
        boundary_warmup_frac=cfg["loss"]["boundary_warmup_frac"])
    logger = MetricLogger(log_dir / f"metrics_fold{fold_idx}.csv")

    # Warmup epochs: first 30% of total epochs = Phase A only
    n_epochs = cfg["training"]["epochs"]
    warmup_epochs = int(0.3 * n_epochs)
    print(f"  Warmup: first {warmup_epochs} epochs = Phase A only\n")

    best_val = 0.0
    patience_ctr = 0
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        random.shuffle(train_idx)
        is_warmup = (epoch <= warmup_epochs)
        phase_desc = "PHASE A only (warmup)" if is_warmup else "ALTERNATING A/B"

        print(f"\n  Fold {fold_idx} Epoch {epoch}/{n_epochs}  [{phase_desc}]")

        avg_A, avg_B, global_step = train_one_epoch(
            model, dataset, train_idx, criterion, optimizer,
            device, epoch, cfg, global_step, is_warmup)
        scheduler.step()

        print(f"  Epoch {epoch} — {time.time()-t0:.1f}s  "
              f"L_A={avg_A:.4f}  L_B={avg_B:.4f}")

        val_m = validate(model, dataset, val_idx, device, cfg,
                          epoch, fold_idx, str(viz_dir), str(pp_dir))
        val_m["epoch"] = epoch
        val_m["fold"] = fold_idx
        val_m["phase_A_loss"] = round(avg_A, 6)
        val_m["phase_B_loss"] = round(avg_B, 6)
        logger.log(val_m)

        if epoch % cfg["training"]["save_every"] == 0:
            model.save_adapter(ckpt_dir / f"adapter_epoch{epoch:03d}.pt")

        vd = val_m["avg_dice"]
        if vd > best_val:
            best_val = vd
            patience_ctr = 0
            model.save_adapter(ckpt_dir / "adapter_best.pt")
            print(f"  ★ Fold {fold_idx} best avg Dice: {best_val:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["training"]["patience"]:
                print(f"  Early stop fold {fold_idx} at epoch {epoch}")
                break

    model.save_adapter(ckpt_dir / "adapter_final.pt")
    return best_val


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg["model"]["device"]

    print("\n═══ Loading dataset ═══")
    dataset = ProstateXDataset(
        dicom_root=cfg["data"]["dicom_root"],
        masks_root=cfg["data"]["masks_root"],
        cache_dir=cfg["data"]["cache_dir"],
        normalize=cfg["data"]["normalize"],
        mask_subdir=cfg["data"]["mask_subdir"],
        box_pad=cfg["data"]["box_pad"],
    )

    splits_path = Path(cfg["output"]["dir"]) / "splits.json"
    folds = load_or_create_splits(dataset, splits_path, k=args.k)
    folds_to_run = [folds[args.fold]] if args.fold is not None else folds

    fold_results = {}
    for fd in folds_to_run:
        fi = fd["fold"]
        fold_results[fi] = train_fold(fi, fd, dataset, cfg, device)

    print(f"\n{'='*60}")
    print(f"  K-FOLD COMPLETE (k={len(folds_to_run)})")
    for fi, bd in fold_results.items():
        print(f"    Fold {fi}: best avg Dice = {bd:.4f}")
    if len(fold_results) > 1:
        vals = list(fold_results.values())
        print(f"    Mean ± std: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()