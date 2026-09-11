"""
inference.py
============

Forward-only pipeline using a trained adapter.

Usage:
    python -m geosam_rt.inference \
        --config geosam_rt/configs/train_config.yaml \
        --adapter checkpoints/adapter_best.pt \
        --patient_id ProstateX-0042 \
        --output_dir ./inference_output
"""

import argparse
import numpy as np
from pathlib import Path
import yaml

from data.dataset import ProstateXPatient, discover_patients
from models.adapted_medsam2 import AdaptedMedSAM2
from geometry.reverse_projection import reverse_project_numpy
from evaluate import evaluate_round_trip, dice_score
from utils.visualization import save_round_trip_figure


def run_inference(cfg, adapter_path, patient_id, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Find patient ──
    patients = discover_patients(
        cfg['data']['dicom_root'], cfg['data']['masks_root'],
        cfg['data']['mask_subdir'])
    info = next((p for p in patients if p['patient_id'] == patient_id), None)
    if info is None or info['missing']:
        print(f"Patient {patient_id} not found or has missing data")
        return

    # ── Load patient data (cached) ──
    patient = ProstateXPatient(
        patient_id=info['patient_id'],
        axial_dir=str(info['axial_dir']),
        sagittal_dir=str(info['sagittal_dir']),
        coronal_dir=str(info['coronal_dir']),
        mask_path=str(info['mask_path']),
        ref_nifti_path=str(info['ref_nifti']) if info['ref_nifti'] else None,
        cache_dir=cfg['data']['cache_dir'],
        normalize=cfg['data']['normalize'],
    )

    # ── Load model + adapter ──
    model = AdaptedMedSAM2(
        checkpoint_path=cfg['model']['sam2_checkpoint'],
        config_path=cfg['model']['sam2_config'],
        device=cfg['model']['device'],
        lora_rank=cfg['lora']['rank'],
        lora_alpha=cfg['lora']['alpha'],
        embed_dim=cfg['model']['embed_dim'],
        train_mode=False,
    )
    if adapter_path:
        model.load_adapter(adapter_path)
        print(f"  Loaded adapter from {adapter_path}")

    # ── Predict ──
    from train import patient_forward_pass
    preds = patient_forward_pass(
        model, patient, cfg['model']['device'],
        cfg['model']['prompt_mode'], cfg['model']['min_fg_px'])

    # ── Save predictions ──
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    np.save(str(pred_dir / "sagittal_masks.npy"),
            np.stack(preds['sag']))
    np.save(str(pred_dir / "coronal_masks.npy"),
            np.stack(preds['cor']))

    # ── Reverse project + evaluate ──
    recon_sag = reverse_project_numpy(
        preds['sag'], patient.sag_series,
        patient.M_ax, patient.origin_ax, patient.M_ax_inv,
        patient.axial_vol.shape, patient.coord_mode,
        grid_raw=patient.sag_grid_raw)
    recon_cor = reverse_project_numpy(
        preds['cor'], patient.cor_series,
        patient.M_ax, patient.origin_ax, patient.M_ax_inv,
        patient.axial_vol.shape, patient.coord_mode,
        grid_raw=patient.cor_grid_raw)

    metrics_sag = evaluate_round_trip(recon_sag, patient.mask_vol, "sag_")
    metrics_cor = evaluate_round_trip(recon_cor, patient.mask_vol, "cor_")

    print(f"\n  {patient_id} Results:")
    for k, v in {**metrics_sag, **metrics_cor}.items():
        print(f"    {k}: {v:.4f}")

    # ── Visualize ──
    for z in range(patient.num_axial_slices):
        ax_img, gt = patient.get_axial_slice(z)
        if patient.coord_mode == 'nifti':
            rs, rc = recon_sag[:, :, z], recon_cor[:, :, z]
        else:
            rs, rc = recon_sag[z], recon_cor[z]
        save_round_trip_figure(
            ax_img, gt, rs, rc, z,
            output_dir / f"axial_{z:03d}.png")

    print(f"  Output → {output_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--patient_id", required=True)
    p.add_argument("--output_dir", default="./inference_output")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_inference(cfg, args.adapter, args.patient_id, args.output_dir)


if __name__ == "__main__":
    main()
