"""
models/adapted_medsam2.py  (v2 — supports axial + sag/cor prediction)
======================================================================
Adds predict_axial() using SAM2's point prompt at centroid.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import shutil
from pathlib import Path

from models.lora_utils import inject_lora, extract_lora_state_dict, count_trainable_params
from models.prior_encoder import PriorEncoder


class AdaptedMedSAM2(nn.Module):
    def __init__(self, checkpoint_path, config_path, device='cuda',
                 lora_rank=4, lora_alpha=8.0, lora_dropout=0.1,
                 embed_dim=256, train_mode=False):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim

        from sam2.build_sam import build_sam2_video_predictor
        self.predictor = build_sam2_video_predictor(
            config_path, checkpoint_path, device=device)

        for param in self.predictor.parameters():
            param.requires_grad = False

        mask_decoder = self._find_mask_decoder()
        if mask_decoder is not None:
            n_replaced, n_params = inject_lora(
                mask_decoder, rank=lora_rank, alpha=lora_alpha,
                dropout=lora_dropout)
            print(f"  LoRA: injected into {n_replaced} linear layers, "
                  f"{n_params:,} params")

        self.prior_encoder = PriorEncoder(embed_dim=embed_dim).to(device)

        trainable = count_trainable_params(self)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Total params: {total:,}  |  Trainable: {trainable:,} "
              f"({100*trainable/total:.2f}%)")

        if train_mode:
            self.prior_encoder.train()
        else:
            self.prior_encoder.eval()

    def _find_mask_decoder(self):
        model = self.predictor
        for attr in ['sam_mask_decoder', 'mask_decoder',
                      'model.mask_decoder', 'sam.mask_decoder']:
            parts = attr.split('.')
            obj = model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                return obj
            except AttributeError:
                continue
        for name, mod in model.named_modules():
            if 'mask_decoder' in name.lower() and len(list(mod.children())) > 0:
                return mod
        return None

    def get_trainable_parameters(self):
        params = list(self.prior_encoder.parameters())
        for name, p in self.predictor.named_parameters():
            if p.requires_grad:
                params.append(p)
        return params

    def save_adapter(self, path):
        state = {
            'prior_encoder': self.prior_encoder.state_dict(),
            'lora': extract_lora_state_dict(self.predictor),
        }
        torch.save(state, str(path))

    def load_adapter(self, path):
        state = torch.load(str(path), map_location=self.device)
        self.prior_encoder.load_state_dict(state['prior_encoder'])
        model_sd = dict(self.predictor.named_parameters())
        for name, val in state['lora'].items():
            if name in model_sd:
                model_sd[name].data.copy_(val)

    # ═══════════════════════════════════════════════════════
    #  Common: write frames as JPEGs for video predictor
    # ═══════════════════════════════════════════════════════

    def _write_frames(self, series, work_dir):
        frames_dir = Path(work_dir)
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        for z in range(series.num_slices):
            img = series.get_slice(z)
            lo, hi = np.percentile(img, 1), np.percentile(img, 99)
            if hi - lo < 1e-6: hi = lo + 1
            win = np.clip((img - lo) / (hi - lo), 0, 1)
            gray_u8 = (win * 255).astype(np.uint8)
            rgb = np.stack([gray_u8] * 3, axis=-1)
            cv2.imwrite(str(frames_dir / f"{z:05d}.jpg"),
                         cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return frames_dir

    # ═══════════════════════════════════════════════════════
    #  Predict axial from POINT prompts (centroid of GT/coarse)
    # ═══════════════════════════════════════════════════════

    def predict_axial_from_points(self, axial_series_or_vol, prompt_masks,
                                    min_fg_px=15, work_dir='/tmp/_tropos_ax'):
        """
        Predict axial masks using POINT prompts at mask centroids.

        Parameters
        ----------
        axial_series_or_vol : DicomSeries OR np.ndarray (D, H, W)
            axial data source; if ndarray, we build a temp "series-like" object
        prompt_masks : list of 2D masks (per axial slice)
            Used to compute centroid points. Empty slices → no prompt.

        Returns
        -------
        dict {slice_idx: 2D binary mask}
        """
        # Handle numpy volume by writing frames directly
        if isinstance(axial_series_or_vol, np.ndarray):
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
                             cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        else:
            frames_dir = self._write_frames(axial_series_or_vol, work_dir)

        obj_id = 1
        state = self.predictor.init_state(video_path=str(frames_dir))
        self.predictor.reset_state(state)

        # Point-prompt each slice with sufficient fg
        # Safety: don't exceed the number of frames in the video
        n_frames = len([f for f in Path(str(frames_dir)).glob("*.jpg")])
        prompt_frames = []
        for z, m in enumerate(prompt_masks):
            if z >= n_frames:
                break
            if m.sum() < min_fg_px:
                continue
            # Centroid
            ys, xs = np.where(m > 0.5)
            cy = float(ys.mean())
            cx = float(xs.mean())
            points = np.array([[cx, cy]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)  # 1 = foreground

            self.predictor.add_new_points_or_box(
                inference_state=state, frame_idx=z, obj_id=obj_id,
                points=points, labels=labels)
            prompt_frames.append(z)

        if not prompt_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)
            return {}

        # Propagate
        segments = {}
        for fi, oids, logits in self.predictor.propagate_in_video(state):
            if obj_id in oids:
                i = oids.index(obj_id)
                segments[fi] = (logits[i] > 0.0).cpu().numpy().squeeze()
        for fi, oids, logits in self.predictor.propagate_in_video(
                state, reverse=True):
            if fi not in segments and obj_id in oids:
                i = oids.index(obj_id)
                segments[fi] = (logits[i] > 0.0).cpu().numpy().squeeze()

        shutil.rmtree(frames_dir, ignore_errors=True)
        return segments

    # ═══════════════════════════════════════════════════════
    #  Predict sag/cor using bbox prompts (existing method)
    # ═══════════════════════════════════════════════════════

    def predict_view(self, native_series, coarse_masks, boxes,
                      prompt_mode='box', min_fg_px=15,
                      work_dir='/tmp/_tropos_frames'):
        frames_dir = self._write_frames(native_series, work_dir)

        obj_id = 1
        state = self.predictor.init_state(video_path=str(frames_dir))
        self.predictor.reset_state(state)

        fg_counts = [int(m.sum()) for m in coarse_masks]
        use_box = prompt_mode.startswith('box')

        if prompt_mode in ('keyframe', 'box_keyframe'):
            best = int(np.argmax(fg_counts))
            prompt_frames = [best] if fg_counts[best] >= min_fg_px else []
        else:
            prompt_frames = [z for z, c in enumerate(fg_counts)
                             if c >= min_fg_px]

        if not prompt_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)
            return {}

        for z in prompt_frames:
            if use_box and boxes[z] is not None:
                self.predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=z, obj_id=obj_id,
                    box=boxes[z])
            else:
                self.predictor.add_new_mask(
                    inference_state=state, frame_idx=z, obj_id=obj_id,
                    mask=(coarse_masks[z] > 0.5))

        segments = {}
        for fi, oids, logits in self.predictor.propagate_in_video(state):
            if obj_id in oids:
                i = oids.index(obj_id)
                segments[fi] = (logits[i] > 0.0).cpu().numpy().squeeze()
        for fi, oids, logits in self.predictor.propagate_in_video(
                state, reverse=True):
            if fi not in segments and obj_id in oids:
                i = oids.index(obj_id)
                segments[fi] = (logits[i] > 0.0).cpu().numpy().squeeze()

        shutil.rmtree(frames_dir, ignore_errors=True)
        return segments