# GeoSAM-RT: Geometry-Anchored Round-Trip Adaptation of MedSAM2 for Multi-View Prostate MRI Segmentation Without Cross-Plane Ground Truth

> **One-line summary:** We segment the prostate in sagittal and coronal
> MRI planes where no ground truth exists, by projecting the axial GT
> through exact DICOM geometry, refining it with a lightly adapted
> MedSAM2, and training the adapter using only a round-trip consistency
> loss against the axial GT — no registration, no cross-plane labels,
> no full model retraining.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Core Insight](#2-core-insight)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Stage 0 — Input Data and Coordinate Systems](#4-stage-0--input-data-and-coordinate-systems)
5. [Stage 1 — Geometric Forward Projection](#5-stage-1--geometric-forward-projection)
6. [Stage 2 — MedSAM2 Zero-Shot Refinement](#6-stage-2--medsam2-zero-shot-refinement)
7. [Stage 3 — Geometric Reverse Projection](#7-stage-3--geometric-reverse-projection)
8. [Stage 4 — Round-Trip Loss Function](#8-stage-4--round-trip-loss-function)
9. [Stage 5 — Learnable Architecture (LoRA + Prior Encoder)](#9-stage-5--learnable-architecture-lora--prior-encoder)
10. [Stage 6 — Iterative Training Loop](#10-stage-6--iterative-training-loop)
11. [Convergence and Late Fusion](#11-convergence-and-late-fusion)
12. [Inference at Test Time](#12-inference-at-test-time)
13. [Evaluation Strategy](#13-evaluation-strategy)
14. [Mathematical Formulation](#14-mathematical-formulation)
15. [Implementation Details](#15-implementation-details)
16. [Failure Modes and Mitigations](#16-failure-modes-and-mitigations)
17. [Related Work and Positioning](#17-related-work-and-positioning)
18. [Glossary](#18-glossary)

---

## 1. Problem Statement

### 1.1 Clinical context

In prostate MRI (e.g. the ProstateX dataset), each patient is scanned in
three orthogonal planes — axial, sagittal, and coronal — as **separate
native DICOM acquisitions**. These are not reformatted views of a single
isotropic volume: each plane has its own resolution, FOV, SNR, slice
thickness, and partial-volume characteristics. They share a common
physical coordinate system (DICOM `FrameOfReferenceUID`), but they are
genuinely different physical measurements of the same anatomy.

### 1.2 The annotation bottleneck

Ground-truth segmentation masks exist **only for the axial plane** (from
the `rcuocolo/PROSTATEx_masks` repository). Creating pixel-level masks
for sagittal and coronal acquisitions would require separate manual
annotation campaigns — expensive, time-consuming, and currently
unavailable for ProstateX or most multi-planar prostate MRI datasets.

### 1.3 What we want

High-quality segmentation masks on the native sagittal and coronal
images, produced without any sagittal/coronal ground truth, and
ideally fused into a single 3D segmentation that exploits the
complementary resolution of all three planes.

### 1.4 Why naive approaches fail

| Approach | Why it fails |
|---|---|
| Train on axial, apply to sag/cor | Domain gap: different resolution, contrast, orientation. Models trained on axial slices have never seen sagittal anatomy presentation. |
| Register sag/cor to axial volume | Inter-scan motion, different FOVs, anisotropic resolution make registration unreliable and introduce interpolation artifacts. |
| Reformat axial into sag/cor views | Axial slice spacing (~3–4 mm) is 6–8× coarser than in-plane resolution (~0.5 mm). Reformatted views are blocky and lose the native high-resolution information entirely. |
| Pure pseudo-labeling | No anchor to ground truth. Pseudo-labels drift over iterations with no convergence guarantee. |

---

## 2. Core Insight

The DICOM affine provides an **exact, analytical, parameter-free mapping**
between any two acquisition planes within the same scan session (same
`FrameOfReferenceUID`). This mapping is not learned, not estimated, not
registered — it is a known geometric fact derived from the scanner's
coordinate system.

This means:

1. We can project the axial GT mask onto the native sagittal/coronal
   grids to get a **geometrically exact but resolution-limited** coarse
   mask (the "prior").

2. Any mask we produce on the sagittal/coronal grid can be projected
   **back** onto the axial grid via the inverse of the same affine.

3. The round-trip **axial → sag/cor → axial** creates a **free
   supervision signal**: the back-projected mask must match the original
   axial GT. No cross-plane ground truth is needed.

4. The only errors in this round-trip are:
   - Through-plane partial volume (axial slices are thick)
   - Inter-scan patient motion (usually small for prostate)
   - The model's own prediction quality

   The first two are fixed physical properties of the data. The third is
   what we optimize. This means the training objective is clean: minimize
   the discrepancy between back-projected predictions and axial GT,
   accounting for the known geometric limitations.

---

## 3. Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  STAGE 0: Input Data                                             │
│    Axial T2 + GT mask (NIfTI)                                    │
│    Sagittal T2 (native DICOM, no GT)                             │
│    Coronal T2 (native DICOM, no GT)                              │
│                                                                  │
│  STAGE 1: Geometric Forward Projection  [no learning]            │
│    Axial GT ──[DICOM affine]──> coarse sag mask, coarse cor mask │
│                                                                  │
│  STAGE 2: MedSAM2 Refinement  [frozen backbone + LoRA adapter]   │
│    coarse mask + native MRI ──[adapted MedSAM2]──> refined mask  │
│    (video predictor: memory propagation across slice stack)       │
│                                                                  │
│  STAGE 3: Geometric Reverse Projection  [no learning]            │
│    refined sag mask ──[inverse DICOM affine]──> recon_axial_sag  │
│    refined cor mask ──[inverse DICOM affine]──> recon_axial_cor  │
│                                                                  │
│  STAGE 4: Round-Trip Loss                                        │
│    L = L_dice(recon, GT)                                         │
│      + λ₁ · L_boundary(recon, GT)                                │
│      + λ₂ · L_consistency(recon_sag, recon_cor)                  │
│      + λ₃ · L_sdm(recon, GT)                                    │
│                                                                  │
│  STAGE 5: Backprop through LoRA adapter only                     │
│    ∇L ──> update LoRA weights in mask decoder                    │
│    ──> update prior encoder conv layers                          │
│    (SAM2 image encoder + memory attention = frozen)              │
│                                                                  │
│  STAGE 6: Iterate until convergence                              │
│    Updated adapter → better sag/cor masks → better recon         │
│    → lower loss → repeat                                         │
│                                                                  │
│  OUTPUT:                                                         │
│    Refined sag mask, refined cor mask                             │
│    ──[late fusion]──> 3D multi-view segmentation                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Stage 0 — Input Data and Coordinate Systems

### 4.1 Data per patient

| Item | Source | Format | Shape (typical) |
|---|---|---|---|
| Axial T2 image | `rcuocolo/PROSTATEx_masks` repo NIfTI | `.nii.gz` | (384, 384, 19) |
| Axial GT mask | Same repo (prostate or lesion) | `.nii.gz` | (384, 384, 19) or differs |
| Sagittal T2 image | TCIA ProstateX DICOM | DICOM series | 320×320×19 slices |
| Coronal T2 image | TCIA ProstateX DICOM | DICOM series | 320×320×15 slices |

### 4.2 Coordinate systems

| System | Convention | Used by |
|---|---|---|
| LPS | Left-Posterior-Superior | DICOM `ImagePositionPatient`, `ImageOrientationPatient` |
| RAS | Right-Anterior-Superior | NIfTI affine matrices |
| Conversion | `RAS = diag(-1, -1, 1) @ LPS` | Applied whenever crossing between systems |

### 4.3 Critical caveat: axial NIfTI transposition

The `rcuocolo/PROSTATEx_masks` repository documents (Issue #9) a known
DICOM-to-NIfTI transposition in the axial T2 series. The repo ships its
own pre-converted NIfTI images whose affines are guaranteed to align
with the masks.

**Rule:** Always use the repo's NIfTI as the axial coordinate reference
(`--ref_nifti`). Never reconvert axial DICOM yourself.

Sagittal and coronal series are read directly from DICOM headers (no
transposition issue).

### 4.4 The shared coordinate frame

All three series within a patient scan session share the same
`FrameOfReferenceUID`. This means `ImagePositionPatient` (IPP) and
`ImageOrientationPatient` (IOP) across all three series are expressed
in the same physical LPS coordinate system. The affine mapping between
any two series is therefore exact by construction — no estimation or
registration is involved.

---

## 5. Stage 1 — Geometric Forward Projection

### 5.1 Goal

Map the axial GT mask onto the native sagittal and coronal pixel grids,
producing a coarse but geometrically exact mask for each non-axial view.

### 5.2 Mathematics

For each pixel `(col, row)` in each native target slice `z_t`:

```
world_LPS = origin_target + col * M_target[:,0] + row * M_target[:,1] + z_t * M_target[:,2]
```

where `M_target` is the 3×3 affine built from the target series' IOP
and PixelSpacing, and `origin_target` is the IPP of the first sorted
slice.

Then map to axial voxel coordinates:

```
axial_vox = M_axial_inv @ (world_LPS - origin_axial)
```

Sample the axial mask at `axial_vox` using `scipy.ndimage.map_coordinates`
with `order=0` (nearest-neighbor for binary masks).

### 5.3 Properties of the coarse projected mask

- **Geometrically exact:** the projection is analytically correct given
  the DICOM metadata.
- **Resolution-limited:** axial slice spacing (~3.6 mm) is much coarser
  than native sagittal/coronal in-plane resolution (~0.5 mm). The
  projected mask appears as horizontal stripes or blocky contours when
  viewed on the native grid.
- **May have holes:** discrete nearest-neighbor sampling can miss thin
  structures or produce discontinuities.
- **Covers the right anatomical region:** despite being coarse, the
  projected mask correctly identifies *where* the prostate is in the
  native view.

This is the "prior" — geometrically grounded but visually coarse.

### 5.4 Already implemented

`cross_view_mask_projection.py` → `project_mask_generic()` function.

---

## 6. Stage 2 — MedSAM2 Zero-Shot Refinement

### 6.1 Goal

Use the coarse projected mask as a prompt to MedSAM2, which refines it
into a smooth, anatomically plausible mask on the native high-resolution
image. During the zero-shot phase (before training), this tests whether
MedSAM2 can improve upon the coarse projection at all. During the
training phase, this is where the learnable adapter operates.

### 6.2 Why video predictor, not image predictor

A native sagittal or coronal DICOM series is a stack of parallel slices —
structurally identical to a "video" of frames. SAM2's video predictor
has a memory-attention mechanism that maintains temporal (here: spatial)
consistency across frames. This is superior to independent per-slice
prediction because:

- Adjacent slices in a sagittal series show similar anatomy
- The memory bank smooths noisy per-slice predictions
- Slice-to-slice coherence is free (no additional loss needed)

### 6.3 Prompt strategy: bounding box (not dense mask)

| Prompt type | Behavior | Suitability |
|---|---|---|
| Dense mask on all slices | SAM2 echoes the prompt with minimal change | Useless for refinement |
| Dense mask on keyframe only | SAM2 propagates from one slice via memory | Tests propagation but noisy |
| **Bounding box on all slices** | SAM2 finds the actual contour within each box | **Best for organ segmentation** |
| Bounding box on keyframe only | Box seeds one slice, memory propagates rest | Good balance |

The bounding box is derived from the coarse projected mask:

```python
ys, xs = np.where(coarse_mask > 0.5)
box = [xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad]
```

with `pad = 10` pixels. This tells SAM2 "the prostate is roughly here"
but lets it decide the actual boundary using its trained understanding
of tissue interfaces.

### 6.4 MedSAM2 architecture (frozen components)

```
Input frame (H×W×3 uint8)
    │
    ▼
┌─────────────────────────┐
│  Hiera-Tiny Image       │  ← FROZEN
│  Encoder (ViT backbone) │     Pre-trained visual features
│  Output: (H/16, W/16,   │     No gradients flow here
│           embed_dim)     │
└────────────┬────────────┘
             │
    ┌────────▼─────────┐
    │  Memory Attention │  ← FROZEN
    │  Cross-attention  │     Maintains slice-to-slice consistency
    │  between current  │     via a memory bank of prior frames
    │  frame and memory │
    │  bank features    │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │  Prompt Encoder   │  ← FROZEN
    │  Encodes box/mask │     Converts geometric prompt to embeddings
    │  into embeddings  │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │  Mask Decoder     │  ← PARTIALLY TRAINABLE (LoRA injected here)
    │  2-layer          │     Produces mask logits from features +
    │  transformer +    │     prompt embeddings
    │  upsampling head  │
    └────────┬─────────┘
             │
             ▼
    Predicted mask logits (H×W)
```

### 6.5 Already implemented (zero-shot, pre-training)

`medsam2_video_cross_view.py` — uses `build_sam2_video_predictor`,
`init_state`, `add_new_points_or_box`, `propagate_in_video`.

---

## 7. Stage 3 — Geometric Reverse Projection

### 7.1 Goal

Map the refined sagittal/coronal masks **back** onto the axial grid,
producing a "reconstructed axial mask" that can be compared against the
real axial GT.

### 7.2 Mathematics

Exactly the same affine math as Stage 1, with source and target swapped:

For each voxel `(i, j, k)` in the axial NIfTI grid:

```
world_LPS = origin_axial + M_axial @ [i, j, k]^T
target_vox = M_target_inv @ (world_LPS - origin_target)
```

Sample the refined sagittal (or coronal) mask at `target_vox`:

```python
recon_axial_from_sag[i,j,k] = map_coordinates(
    refined_sag_mask_volume, target_vox_coords,
    order=1, mode='constant', cval=0.0)
```

**Critical difference from Stage 1:** here we use `order=1` (trilinear
interpolation) instead of `order=0` (nearest-neighbor). This is because:

- The refined mask is now a soft probability map (not a hard binary),
  so interpolation is meaningful.
- We need gradients to flow through this sampling operation during
  training, and trilinear interpolation is differentiable (nearest-
  neighbor is not).
- The soft reconstructed mask is compared against the GT via
  differentiable loss functions.

### 7.3 Differentiable sampling

For training, `scipy.ndimage.map_coordinates` is not differentiable.
We need a PyTorch equivalent: `torch.nn.functional.grid_sample`.

```python
# Precompute the coordinate grid (fixed, not learned):
# For each axial voxel (i,j,k), compute its (col, row, slice)
# in the sagittal series using the affine math above.
# Normalize to [-1, 1] range as required by grid_sample.

recon_axial = F.grid_sample(
    refined_sag_mask.unsqueeze(0).unsqueeze(0),  # (1, 1, D, H, W)
    coord_grid.unsqueeze(0),                      # (1, D_ax, H_ax, W_ax, 3)
    mode='bilinear',
    padding_mode='zeros',
    align_corners=True
).squeeze()
```

The coordinate grid is **precomputed once** from the DICOM affines and
stored as a fixed tensor. It is never learned — it is a geometric
constant. Only the mask values being sampled depend on the model's
predictions.

### 7.4 Two reconstructions

Each refined view produces an independent axial reconstruction:

- `recon_axial_sag`: refined sagittal mask → back-projected to axial
- `recon_axial_cor`: refined coronal mask → back-projected to axial

Both are compared against the same axial GT. Additionally, they are
compared against *each other* for multi-view consistency.

### 7.5 Not yet implemented

This is the first piece that needs to be built.

---

## 8. Stage 4 — Round-Trip Loss Function

### 8.1 Design principles

1. **Volumetric overlap** (Dice) ensures the bulk of the mask is correct.
2. **Boundary precision** (boundary loss / SDM) ensures contours are
   sharp and anatomically placed, not just "roughly right."
3. **Multi-view consistency** ensures sagittal and coronal predictions
   agree with each other, not just with the GT.
4. **All terms are differentiable** with respect to the model's mask
   predictions.

### 8.2 Term 1: Soft Dice Loss

```
L_dice = 1 - (2 * Σ(p * g) + ε) / (Σ(p²) + Σ(g²) + ε)
```

where `p` is the reconstructed soft mask (back-projected from sag or cor),
`g` is the axial GT mask, and `ε = 1e-5` for numerical stability.

Applied separately to each view's reconstruction:

```
L_dice_total = L_dice(recon_sag, GT) + L_dice(recon_cor, GT)
```

**Why Dice alone is insufficient:** For a large organ like the prostate,
Dice can be 0.90+ even when the boundary is off by several pixels. The
interior dominates the numerator, masking boundary errors.

### 8.3 Term 2: Boundary Loss (Kervadec et al., 2019)

Instead of comparing binary masks, compute the signed distance map (SDM)
of the GT boundary and weight the prediction by it:

```python
sdm_gt = distance_transform_edt(GT) - distance_transform_edt(1 - GT)
L_boundary = Σ(p * sdm_gt) / Σ(|sdm_gt|)
```

Interpretation: predictions far from the true boundary incur a large
penalty proportional to their distance from it. A prediction that is
correct at the boundary has near-zero boundary loss regardless of
interior values. This directly penalizes the boundary displacement that
Dice misses.

### 8.4 Term 3: Multi-View Consistency Loss

The sagittal and coronal reconstructions are independent views of the
same anatomy. They should agree:

```
L_consistency = 1 - Dice(recon_sag, recon_cor)
```

This term is powerful because it provides supervision even in regions
where the axial GT mask is ambiguous (e.g. near the apex/base of the
prostate where axial slices are thick and boundaries are uncertain).
If the sagittal view resolves a boundary that the coronal view doesn't
(or vice versa), this loss encourages them to converge.

### 8.5 Term 4: Signed Distance Map (SDM) Regression

Optionally, instead of predicting a binary mask, predict the signed
distance field:

```
pred_sdm = model(image, prior)      # output: continuous, + inside, - outside
gt_sdm = compute_sdm(GT_mask)

L_sdm = ||pred_sdm - gt_sdm||₁     # L1 regression on the distance field
```

This transforms the segmentation problem from classification (inside/
outside) to regression (how far from the boundary). The model is
forced to learn the *exact* boundary location as the zero-crossing of
the distance field, rather than just getting the interior right.

### 8.6 Combined loss

```
L_total = L_dice_total
        + λ₁ · L_boundary_total
        + λ₂ · L_consistency
        + λ₃ · L_sdm_total
```

Recommended starting weights (tune on validation):

| Term | Weight | Rationale |
|---|---|---|
| `L_dice` | 1.0 | Baseline volumetric overlap |
| `λ₁` (boundary) | 0.5 | Boundary precision, ramp up from 0 over first 20% of training |
| `λ₂` (consistency) | 0.3 | Cross-view agreement, always on |
| `λ₃` (SDM) | 0.2 | Distance field regression, optional |

**Boundary loss warm-up:** start `λ₁ = 0` and linearly increase to its
target value over the first 20% of training iterations. This prevents
the boundary loss from dominating early when the model's predictions
are still coarse (distance values would be large and noisy).

---

## 9. Stage 5 — Learnable Architecture (LoRA + Prior Encoder)

### 9.1 Design philosophy

Keep the foundation model frozen. Train only a small adapter that
specializes MedSAM2 for this specific task (multi-view prostate
segmentation with a geometric prior). This ensures:

- Fast training (few parameters)
- No catastrophic forgetting of MedSAM2's general knowledge
- Easy to share (adapter weights are tiny, ~2–5 MB)
- Theoretically clean (foundation model = fixed feature extractor,
  adapter = task-specific head)

### 9.2 What is frozen (no gradients)

| Component | Parameters | Status |
|---|---|---|
| Hiera-Tiny image encoder | ~5.6M | **Frozen** |
| Memory attention module | ~2.1M | **Frozen** |
| Prompt encoder | ~0.1M | **Frozen** |
| Memory bank / memory encoder | ~1.2M | **Frozen** |

### 9.3 What is trained

#### 9.3.1 LoRA adapters in the mask decoder

LoRA (Low-Rank Adaptation, Hu et al. 2022) injects trainable low-rank
matrices into existing linear layers without changing the original weights:

```
output = W_frozen @ x + (B @ A) @ x
```

where `W_frozen` is the original weight matrix (frozen), `A` has shape
`(rank, in_features)` and `B` has shape `(out_features, rank)`. Typical
`rank = 4` or `rank = 8`.

We inject LoRA into:

- All `nn.Linear` layers in the mask decoder's two transformer blocks
  (self-attention Q, K, V projections and the FFN)
- The final mask prediction MLP

This adds approximately **0.1–0.5% additional parameters** relative
to the mask decoder.

```python
# Pseudocode for LoRA injection
for name, module in mask_decoder.named_modules():
    if isinstance(module, nn.Linear):
        lora = LoRALinear(module, rank=4, alpha=8)
        replace_module(mask_decoder, name, lora)
```

#### 9.3.2 Prior encoder (new, small CNN)

A lightweight convolutional network that encodes the coarse projected
mask (the geometric prior) into a feature map compatible with SAM2's
internal representation:

```
Input: coarse_mask (H, W, 1)    — the Stage 1 projected mask
       ──> Conv2d(1, 16, 3, pad=1), GELU
       ──> Conv2d(16, 32, 3, pad=1), GELU
       ──> Conv2d(32, 64, 3, stride=2, pad=1), GELU   # downsample 2×
       ──> Conv2d(64, 64, 3, stride=2, pad=1), GELU   # downsample 2×
       ──> Conv2d(64, 256, 1)                          # project to embed_dim
Output: prior_features (H/4, W/4, 256)
```

These prior features are **added element-wise** to the image encoder's
output before it enters the mask decoder:

```python
image_features = frozen_image_encoder(native_mri)       # (H/16, W/16, 256)
prior_features = prior_encoder(coarse_projected_mask)    # (H/4, W/4, 256)
prior_features_down = F.interpolate(prior_features, image_features.shape[-2:])
fused_features = image_features + prior_features_down    # element-wise addition
mask_logits = lora_mask_decoder(fused_features, prompt_embeddings)
```

Total trainable parameters in the prior encoder: ~50K.

#### 9.3.3 Total trainable parameters

| Component | Parameters | % of MedSAM2 |
|---|---|---|
| LoRA adapters (rank=4) | ~15K | ~0.15% |
| Prior encoder | ~50K | ~0.5% |
| **Total trainable** | **~65K** | **~0.65%** |

### 9.4 Architecture diagram

```
                                    ┌─────────────────────┐
Native MRI slice (H×W×3)           │                     │
        │                           │  Coarse projected   │
        ▼                           │  mask (H×W×1)       │
┌───────────────────┐               │  (from Stage 1)     │
│ Frozen Hiera-Tiny │               └──────────┬──────────┘
│ Image Encoder     │                          │
│ (no grad)         │                          ▼
└────────┬──────────┘               ┌──────────────────────┐
         │                          │ Prior Encoder (NEW)  │
         │  image_features          │ 5 conv layers, ~50K  │
         │  (H/16, W/16, 256)      │ TRAINABLE            │
         │                          └──────────┬───────────┘
         │                                     │
         │              prior_features (H/4, W/4, 256)
         │                   │  ──> interpolate to (H/16, W/16, 256)
         │                   │
         └──────► (+) ◄──────┘
                  │
                  │  fused_features
                  ▼
         ┌──────────────────┐
         │ Frozen Memory    │
         │ Attention        │
         │ (cross-attn with │
         │  memory bank)    │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐       ┌──────────────────┐
         │ Mask Decoder     │◄──────│ Frozen Prompt     │
         │ + LoRA adapters  │       │ Encoder           │
         │ (TRAINABLE)      │       │ (encodes bbox)    │
         └────────┬─────────┘       └──────────────────┘
                  │
                  ▼
         Refined mask logits (H × W)
                  │
                  ▼ sigmoid
         Soft probability mask p ∈ [0, 1]
```

---

## 10. Stage 6 — Iterative Training Loop

### 10.1 Training procedure for one patient

```
for iteration in range(max_iterations):

    # --- Forward: Sagittal ---
    for z_s in range(num_sagittal_slices):
        coarse_sag[z_s] = forward_project(axial_GT, axial_affine, sag_affine, z_s)
        native_sag[z_s] = sagittal_series.get_slice(z_s)

    # Run adapted MedSAM2 video predictor on sagittal stack
    refined_sag = adapted_medsam2_video(
        frames=native_sag,
        prompts=bbox_from(coarse_sag),
        prior_masks=coarse_sag,        # fed through prior encoder
    )

    # --- Forward: Coronal ---
    for z_c in range(num_coronal_slices):
        coarse_cor[z_c] = forward_project(axial_GT, axial_affine, cor_affine, z_c)
        native_cor[z_c] = coronal_series.get_slice(z_c)

    refined_cor = adapted_medsam2_video(
        frames=native_cor,
        prompts=bbox_from(coarse_cor),
        prior_masks=coarse_cor,
    )

    # --- Reverse projection ---
    recon_axial_sag = reverse_project(refined_sag, sag_affine, axial_affine)
    recon_axial_cor = reverse_project(refined_cor, cor_affine, axial_affine)

    # --- Loss ---
    loss = (  dice_loss(recon_axial_sag, axial_GT)
            + dice_loss(recon_axial_cor, axial_GT)
            + λ₁ * (boundary_loss(recon_axial_sag, axial_GT)
                   + boundary_loss(recon_axial_cor, axial_GT))
            + λ₂ * consistency_loss(recon_axial_sag, recon_axial_cor)
            + λ₃ * (sdm_loss(recon_axial_sag, axial_GT)
                   + sdm_loss(recon_axial_cor, axial_GT))
    )

    # --- Backprop ---
    loss.backward()     # gradients flow through:
                        #   reverse_project (grid_sample, differentiable)
                        #   → refined mask logits
                        #   → LoRA layers in mask decoder
                        #   → prior encoder conv layers
                        # gradients do NOT flow through:
                        #   frozen image encoder
                        #   frozen memory attention
                        #   frozen prompt encoder
                        #   forward projection (fixed geometry)

    optimizer.step()
    optimizer.zero_grad()
```

### 10.2 Training across patients

The adapter is trained across all patients in the training set:

```
for epoch in range(num_epochs):
    for patient in training_patients:
        load DICOM series (axial, sagittal, coronal)
        load axial GT mask
        run one iteration of the loop above
        accumulate gradients
    optimizer.step()  # or step per patient
```

### 10.3 Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW | Standard for LoRA fine-tuning |
| Learning rate | 1e-4 | For LoRA + prior encoder jointly |
| Weight decay | 1e-2 | Standard regularization |
| LR schedule | Cosine annealing | Warm-up 5% of steps, then cosine to 0 |
| Batch size | 1 patient | Due to variable slice counts per patient |
| LoRA rank | 4 | Increase to 8 if underfitting |
| LoRA alpha | 8 | Scaling factor = alpha / rank = 2 |
| LoRA dropout | 0.1 | Applied during training only |
| Epochs | 50–100 | Monitor round-trip Dice on validation set |
| Gradient clipping | max_norm=1.0 | Prevent exploding gradients through grid_sample |
| Boundary loss warm-up | 20% of total steps | λ₁ linearly increases from 0 |

### 10.4 Memory considerations

Since only ~65K parameters require gradients, memory usage is dominated
by the forward pass through the frozen MedSAM2, not by the optimizer
state. Expected GPU memory: ~4–6 GB for the tiny-Hiera backbone with
one patient's sagittal stack (19 frames at 320×320).

---

## 11. Convergence and Late Fusion

### 11.1 Convergence criterion

Training converges when:
- Round-trip Dice (recon_axial vs GT) plateaus for 10 consecutive
  epochs on the validation set
- OR the absolute improvement per epoch drops below 0.001 Dice

### 11.2 Late fusion: 3D multi-view segmentation

After training, each view produces a refined mask on its native grid.
To combine them into a single 3D segmentation:

```
For each voxel in a target 3D grid (e.g. 0.5mm isotropic):
    p_axial   = sample axial GT mask at this world coordinate
    p_sagittal = sample refined sagittal mask at this world coordinate
    p_coronal  = sample refined coronal mask at this world coordinate

    p_fused = (w_ax * p_axial + w_sag * p_sagittal + w_cor * p_coronal)
            / (w_ax + w_sag + w_cor)

    # Weights proportional to each view's resolution in the
    # direction perpendicular to this voxel's local gradient:
    # views with fine resolution in the boundary-normal direction
    # get higher weight.
```

The axial view has high in-plane resolution but poor through-plane
resolution. The sagittal view has high resolution in the L-R direction.
The coronal view has high resolution in the A-P direction. Fusing them
with direction-dependent weights produces a 3D mask that is sharp in
all three directions.

---

## 12. Inference at Test Time

### 12.1 For a new patient (with axial GT available)

Same pipeline but **no training** — just a forward pass through the
frozen backbone + trained adapter:

```
1. Load axial GT + sagittal/coronal DICOM
2. Forward project axial GT → coarse sag/cor masks (geometry)
3. Run adapted MedSAM2 video predictor with bbox prompts + prior encoder
4. Get refined sag/cor masks
5. (Optional) Late fusion into 3D
```

### 12.2 For a truly new patient (no GT at all)

If the adapter has been trained on enough patients, it generalizes:

```
1. Run a pre-trained axial segmentation model to get an axial mask
   (e.g. nnU-Net, which achieves ~0.90 Dice on ProstateX axial)
2. Use that predicted axial mask as the "GT" for projection
3. Forward project → coarse sag/cor masks
4. Run adapted MedSAM2 → refined sag/cor masks
5. (Optional) Round-trip check: back-project and compare with
   the axial prediction as a confidence measure
```

### 12.3 Test-time adaptation (TTT) variant

For a single test patient where axial GT is available but the adapter
was not trained on this specific patient's anatomy:

```
1. Load pre-trained adapter weights
2. Run N iterations of the training loop (Section 10.1) on THIS
   patient only, using its axial GT
3. The adapter further specializes to this patient's anatomy
4. Use the patient-adapted model for final sag/cor prediction
```

This is the classical test-time training (TTT) setup and may improve
results on challenging cases at the cost of ~30–60 seconds of per-patient
optimization.

---

## 13. Evaluation Strategy

### 13.1 Metrics we CAN compute

| Metric | What it measures | Computed on |
|---|---|---|
| Round-trip Dice | How well sag/cor predictions reconstruct axial GT | Axial grid |
| Round-trip Hausdorff (95th) | Boundary error of reconstructed vs GT | Axial grid |
| Round-trip boundary F1 | Precision/recall of boundary pixels | Axial grid |
| Cross-view consistency Dice | Agreement between sag-recon and cor-recon | Axial grid |
| Improvement over baseline | Round-trip Dice (trained) - Round-trip Dice (zero-shot) | Axial grid |

### 13.2 Metrics we CANNOT compute (no sag/cor GT)

| Metric | Why unavailable |
|---|---|
| Sagittal Dice vs GT | No sagittal ground truth exists |
| Coronal Dice vs GT | No coronal ground truth exists |

### 13.3 Qualitative evaluation

Per-slice visual comparison (already implemented):

```
[ STACKED ] [ REFORMATTED ] [ NATIVE + coarse mask ] [ NATIVE + MedSAM2 mask ]
```

Key things to look for:
- Does MedSAM2 fill the holes left by discrete sampling?
- Are the boundaries smoother and more anatomically plausible?
- Does the mask follow the actual prostate boundary visible in the
  native MRI, rather than just reproducing the blocky projection?

### 13.4 Indirect quantitative validation (if resources permit)

If a radiologist can annotate even a few sagittal/coronal slices for
a small subset of patients, those serve as held-out test cases for
computing actual Dice on non-axial views. Even 5–10 annotated slices
across a few patients would be highly informative.

---

## 14. Mathematical Formulation

### 14.1 Notation

| Symbol | Meaning |
|---|---|
| `x_a` | Axial image volume, indexed `x_a[i,j,k]` |
| `y_a` | Axial GT mask, same grid as `x_a` |
| `x_s, x_c` | Native sagittal / coronal image stacks |
| `Φ_{a→s}` | Forward projection operator (axial grid → sagittal grid) |
| `Φ_{s→a}` | Reverse projection operator (sagittal grid → axial grid) |
| `Φ_{a→c}, Φ_{c→a}` | Same for coronal |
| `m_s^coarse = Φ_{a→s}(y_a)` | Coarse sagittal mask (geometric projection) |
| `f_θ` | Adapted MedSAM2 (trainable parameters θ) |
| `m_s^refined = f_θ(x_s, m_s^coarse)` | Refined sagittal mask |
| `ŷ_a^s = Φ_{s→a}(m_s^refined)` | Reconstructed axial from sagittal |
| `ŷ_a^c = Φ_{c→a}(m_c^refined)` | Reconstructed axial from coronal |

### 14.2 Objective

```
θ* = argmin_θ  L_dice(ŷ_a^s, y_a) + L_dice(ŷ_a^c, y_a)
              + λ₁ [L_bnd(ŷ_a^s, y_a) + L_bnd(ŷ_a^c, y_a)]
              + λ₂  L_dice(ŷ_a^s, ŷ_a^c)
              + λ₃ [L_sdm(ŷ_a^s, y_a) + L_sdm(ŷ_a^c, y_a)]
```

Subject to:
- `Φ_{a→s}, Φ_{s→a}, Φ_{a→c}, Φ_{c→a}` are **fixed** (DICOM geometry)
- `f_θ` = MedSAM2 with frozen backbone + trainable LoRA + prior encoder
- `θ` = {LoRA weights, prior encoder weights}

### 14.3 Gradient flow

```
∂L/∂θ = ∂L/∂ŷ_a · ∂ŷ_a/∂m_refined · ∂m_refined/∂θ
```

where:
- `∂L/∂ŷ_a` comes from the loss function (Dice + boundary + SDM)
- `∂ŷ_a/∂m_refined` comes from `grid_sample` (differentiable bilinear
  interpolation in the reverse projection)
- `∂m_refined/∂θ` comes from the LoRA layers and prior encoder in
  MedSAM2's mask decoder

The forward projection `Φ_{a→s}` does NOT need to be differentiable
because it operates on the GT mask (a fixed input), not on a predicted
quantity. Only the reverse projection `Φ_{s→a}` needs to be
differentiable (it operates on the model's predictions).

### 14.4 Why this converges

The round-trip constraint is **over-determined**: the axial GT has
`N_axial` voxels of supervision, but the model predicts on sagittal
(`N_sag` voxels) and coronal (`N_cor` voxels) grids that are much
finer in their respective through-plane directions. The model cannot
simply memorize the GT — it must generalize to the finer grids in a
way that, when projected back, reproduces the coarser GT.

The geometric anchoring prevents drift because:
- `Φ` and `Φ⁻¹` are fixed, known, and exact
- The GT is real and does not change
- The only free variables are the ~65K adapter parameters

---

## 15. Implementation Details

### 15.1 File structure (proposed)

```
CorssViewVisualization/
├── cross_view_mask_projection.py       # Stage 1: forward projection (existing)
├── medsam2_video_cross_view.py         # Stage 2: zero-shot MedSAM2 (existing)
├── reverse_projection.py              # Stage 3: reverse projection (to build)
├── losses.py                          # Stage 4: loss functions (to build)
├── prior_encoder.py                   # Stage 5a: prior encoder CNN (to build)
├── lora_utils.py                      # Stage 5b: LoRA injection (to build)
├── train.py                           # Stage 6: training loop (to build)
├── inference.py                       # Stage 7: inference pipeline (to build)
├── evaluate.py                        # Evaluation metrics (to build)
├── configs/
│   └── train_config.yaml              # Hyperparameters
├── MedSam2/
│   └── MedSAM2/                       # Cloned bowang-lab repo
│       ├── checkpoints/
│       │   └── MedSAM2_latest.pt
│       └── sam2/
│           └── configs/
│               └── sam2.1_hiera_t512.yaml
└── README_MultiView_Pipeline.md       # This document
```

### 15.2 Key implementation considerations

**Differentiable reverse projection:** The `grid_sample` call requires
coordinates normalized to `[-1, 1]`. Precompute the affine-derived
coordinate grids once per patient and cache them as tensors.

**Video predictor in training mode:** SAM2's video predictor was
designed for inference. For training, we need to:
1. Set the LoRA layers and prior encoder to `train()` mode
2. Keep everything else in `eval()` mode
3. Ensure `torch.no_grad()` is NOT applied globally
4. Handle the frame-by-frame propagation with gradient checkpointing
   if memory is tight

**Batch size = 1 patient:** Each patient has a different number of
slices and different image dimensions. Batching across patients requires
padding or dynamic batching. Starting with batch_size=1 (one patient
per optimization step) is simplest and sufficient given the small
parameter count.

**Mixed precision:** Use `torch.cuda.amp.autocast()` for the frozen
forward pass and full precision for the LoRA/prior encoder gradients.
This matches MedSAM2's original training setup.

### 15.3 Dependencies

```
torch >= 2.0          # grid_sample, autocast, LoRA support
sam2                   # MedSAM2 / SAM2 package
pydicom                # DICOM reading
nibabel                # NIfTI reading
numpy                  # array math
scipy                  # distance_transform_edt, map_coordinates (eval only)
matplotlib             # visualization
peft >= 0.6            # optional: Hugging Face PEFT for LoRA injection
                       # (or implement manually, ~50 lines)
```

---

## 16. Failure Modes and Mitigations

| Failure mode | Cause | Mitigation |
|---|---|---|
| Round-trip Dice is high but sag/cor masks look wrong | The model found a degenerate solution that satisfies the axial GT but doesn't match the anatomy in other views | Add L_consistency (cross-view agreement) and visual inspection. The consistency loss prevents this by requiring sag and cor to agree. |
| Boundary loss explodes early in training | SDM values are large when predictions are far off | Boundary loss warm-up (λ₁ ramps from 0 over first 20% of training) |
| Grid_sample gradients are noisy | Bilinear interpolation gradients can be unstable at integer boundaries | Gradient clipping (max_norm=1.0) |
| MedSAM2 produces empty masks for some slices | Box prompt doesn't intersect any recognizable anatomy | Fall back to dense mask prompt for those slices; or propagate from neighboring frames via memory |
| Model overfits to training patients | Small dataset (201 patients) | Moderate LoRA rank (4), dropout (0.1), early stopping on validation round-trip Dice |
| Inter-scan motion breaks the affine assumption | Patient moved between axial and sag/cor acquisitions | FrameOfReferenceUID match confirms shared coordinate system. Residual motion is small for prostate (pelvic anatomy is relatively fixed). If needed, add a small learned affine correction (~6 parameters) |
| Thick axial slices create partial volume in round-trip | Axial slice spacing ~3.6mm means each back-projected voxel averages over a large axial extent | Use soft (probabilistic) masks throughout; the loss naturally handles the averaging. Alternatively, weight the loss by the confidence of the back-projection (lower weight for voxels that map to inter-slice gaps) |

---

## 17. Related Work and Positioning

### 17.1 Multi-planar segmentation

Standard multi-planar approaches (triplanar U-Net, multi-view CNNs)
assume GT exists in all planes, or work with isotropic volumes where
any plane can be trivially reformatted. Our setting — separate native
acquisitions with GT in only one plane — is fundamentally different.

### 17.2 Test-time training / adaptation

TTT (Sun et al., 2020), TENT (Wang et al., 2021), and DUA adapt models
at test time using self-supervised signals. Our round-trip consistency
is a novel self-supervised signal specific to multi-view medical imaging.
Unlike entropy minimization (TENT) or rotation prediction (TTT), our
signal is grounded in exact geometry and real GT.

### 17.3 SAM/MedSAM adaptation

Recent work adapts SAM for medical imaging via LoRA (MA-SAM, SAMed,
MedSAM), prompt engineering, or adapter layers. We build on this by
adding a geometric prior channel and training with round-trip
consistency rather than direct supervision on the target view.

### 17.4 Cycle consistency

CycleGAN (Zhu et al., 2017) established cycle/round-trip consistency
for unpaired image translation. Our approach is analogous but differs
in that our "cycle" is through a known geometric transformation (not a
learned one), and we have GT at one end of the cycle (not at neither
end).

### 17.5 What is novel in this work

1. **The problem setup itself**: multi-view segmentation from separate
   native acquisitions where GT exists in only one plane.
2. **Geometric prior as a spatial prompt channel**: not registration,
   not an atlas — exact DICOM affine projection.
3. **Round-trip consistency training with a geometric anchor**: the
   DICOM affine is exact and fixed, preventing the drift inherent in
   pure cycle-consistency or pseudo-label methods.
4. **Lightweight adaptation of a foundation model (MedSAM2) via LoRA +
   prior encoder**, trained with the round-trip signal — no cross-plane
   GT required.

> **Caveat:** This analysis is based on knowledge through May 2025.
> A thorough literature search should be performed to confirm no
> concurrent work has addressed the same problem setup. Please verify
> all cited papers against the actual literature — I may hallucinate
> citations.

---

## 18. Glossary

| Term | Definition |
|---|---|
| IPP | `ImagePositionPatient` — DICOM LPS position of pixel (0,0) of a slice |
| IOP | `ImageOrientationPatient` — two direction cosines defining the image plane |
| LPS | Left-Posterior-Superior, DICOM world coordinate convention |
| RAS | Right-Anterior-Superior, NIfTI world coordinate convention |
| Forward projection | Mapping a mask from its source grid onto a different target grid via affine geometry |
| Reverse projection | Mapping a mask back from the target grid to the source grid via inverse affine |
| Round-trip | Forward project → refine → reverse project; the composition should reconstruct the original |
| Coarse mask | The Stage 1 geometrically projected mask, resolution-limited but spatially correct |
| Refined mask | The MedSAM2 output, smooth and anatomically plausible on the native grid |
| Prior encoder | Small CNN that encodes the coarse mask into features compatible with SAM2 |
| LoRA | Low-Rank Adaptation — injects small trainable matrices into frozen linear layers |
| SDM | Signed Distance Map — positive inside the mask, negative outside, zero at the boundary |
| Boundary loss | Loss computed on the SDM that directly penalizes boundary displacement |
| Consistency loss | Loss measuring agreement between sagittal-derived and coronal-derived axial reconstructions |
| TTT | Test-Time Training — adapting the model to a specific test sample using self-supervision |
| Late fusion | Combining predictions from multiple views into a single 3D segmentation |
| FrameOfReferenceUID | DICOM tag identifying the shared coordinate system across all series in a study |
| `grid_sample` | PyTorch function for differentiable bilinear sampling on a coordinate grid |
