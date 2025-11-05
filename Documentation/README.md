# GAN Variant1 - Baseline CUT Training System

**REVERTED TO BASELINE** - This folder contains the minimal, proven configuration that achieved **103-105 MiFID** in the original baseline run.

## What Was Reverted

This codebase was simplified back to baseline by removing experimental features:

### **Removed Features**:
- ❌ Self-Attention blocks
- ❌ Channel Attention blocks  
- ❌ AdaIN Style Dropout
- ❌ Multiscale discriminator (reverted to single-scale PatchGAN)
- ❌ Spectral Normalization
- ❌ Palette Prior loss
- ❌ k-NN Repulsion loss (CLIP-based)
- ❌ Feature Matching loss
- ❌ In-training FID/CLIP metrics
- ❌ Early stopping logic

### **Baseline Architecture** (What Remains):
- ✅ **Generator**: Pure ResNet-9 (no attention, no style dropout)
- ✅ **Discriminator**: Single-scale PatchGAN (70x70)
- ✅ **Loss**: Hinge adversarial + PatchNCE + Identity (warmup only)
- ✅ **Training**: EMA, AMP, Lazy R1, DiffAugment
- ✅ **Stability**: Gradient clipping, NaN detection, numerical safeguards in PatchNCE

## Setup

```powershell
# Create environment
conda create -n GAN310 python=3.10 -y
conda activate GAN310

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pyyaml tqdm pillow scipy
```

## Data Structure

```
../data/
├── photo_jpg/    # Photos (source domain)
└── monet_jpg/    # Monet paintings (target domain)
```

## Training

```powershell
# Train baseline
python -m training.train_cutpp --config configs/train_gan_cutpp.yaml
```

### Key Config Parameters

```yaml
# Baseline optimizer (standard GAN betas)
optim:
  G:
    lr: 2.0e-4
    betas: [0.5, 0.999]
  D:
    lr: 2.0e-4
    betas: [0.5, 0.999]

# Loss weights (simple, stable)
loss_weights:
  adv: 1.0
  patchnce: 1.0
  identity_warm: 0.1
  identity_final: 0.0

# PatchNCE (baseline)
patchnce:
  temperature: 0.07
  num_patches: 256
  nce_layers: [0, 4, 8, 12, 16]

# Gradient clipping (safety net)
grad_clip_g: 10.0
grad_clip_d: 10.0

# R1 regularization
r1:
  gamma: 10.0
  every: 16  # Lazy R1
```

## Resume Training

```powershell
python -m training.train_cutpp \
  --config configs/train_gan_cutpp.yaml \
  --resume checkpoints/ckpt_step10000.pt
```

## Generate Images

After training, generate images for evaluation:

```powershell
python generate_folder.py \
  --ckpt checkpoints/ckpt_step40000.pt \
  --photos ../data/photo_jpg \
  --out ./outputs/ckpt_step40000 \
  --batch 64 \
  --limit 7038 \
  --device cuda
```

## Evaluate MiFID

Use the separate `EVAL` folder to compute MiFID:

```powershell
cd ../EVAL
python -m eval.cli \
  --config configs/eval_local.yaml \
  --real ../data/monet_jpg \
  --fake ../GAN_Variant1/outputs/ckpt_step40000 \
  --batch 64 \
  --workers 8 \
  --device cuda \
  --out ./cache/reports/ckpt_step40000.json
```

## Monitoring

Training logs: `logs/train_log.txt`

Key losses to watch:
- `g_loss` - Generator loss
- `d_loss` - Discriminator loss  
- `nce` - PatchNCE contrastive loss
- `identity` - Identity loss (should decay to 0 after warmup)
- `r1` - R1 gradient penalty

## Checkpoints

Checkpoints saved to `checkpoints/`:
- `ckpt_step{N}.pt` - Every 2000 steps
- `ckpt_final.pt` - Final checkpoint

Each checkpoint contains:
- Generator weights (training)
- Generator EMA weights (for inference)
- Discriminator weights
- Optimizer states
- Training step
- Config

## Numerical Stability Features

The baseline includes safety mechanisms to prevent training collapse:

1. **Gradient Clipping**: Max norm of 10.0 for both G and D
2. **PatchNCE Safeguards**: 
   - Feature normalization with epsilon
   - Logit clamping to prevent exp() overflow
   - NaN detection and fallback
3. **NaN Detection**: Training stops immediately if NaN detected

These were added to prevent the collapse observed in experimental runs.

## Hardware Requirements

**Recommended**:
- GPU: NVIDIA A100 (40GB+)
- Batch size: 12 @ 256x256
- Training time: ~8-10 hours

**Minimum**:
- GPU: NVIDIA RTX 3090 (24GB)
- Batch size: 8 @ 256x256
- Training time: ~12-16 hours

## File Structure

```
GAN_Variant1/
├── configs/
│   └── train_gan_cutpp.yaml       # Baseline config ⭐
├── dataio/
│   ├── photos_dataset.py          # Photos dataset
│   ├── monet_dataset.py           # Monet dataset
│   └── transforms.py              # Transforms + Lab conversion
├── models/
│   ├── generator_resnet_attn.py   # ResNet-9 (baseline, no attention)
│   └── discriminator_patchgan.py  # PatchGAN (single-scale)
├── losses/
│   ├── adv_hinge.py               # Hinge adversarial loss
│   ├── patchnce_cut.py            # PatchNCE (with numerical safeguards)
│   └── identity_l1.py             # Identity loss
├── training/
│   ├── train_cutpp.py             # Main training script ⭐
│   ├── diffaugment.py             # DiffAugment
│   └── sched_optim.py             # Optimizers
├── utils/
│   ├── amp_utils.py               # AMP (with gradient clipping)
│   ├── io_ckpt.py                 # Checkpointing + EMA
│   └── seed_dist.py               # Seeding
├── generate_folder.py             # Generate images for eval ⭐
└── README.md                      # This file
```

## Why This Baseline?

This configuration:
- **Achieved 103-105 MiFID** in original runs
- **Stable training** (no NaN collapses with new safeguards)
- **Minimal complexity** (easier to debug and iterate)
- **Proven architecture** (ResNet-9 + PatchGAN + CUT)

Use this as the foundation for future experiments. Any new features should be tested as deltas from this baseline.

## Troubleshooting

**NaN losses**:
- Check gradient clipping is enabled (`grad_clip_g` and `grad_clip_d`)
- Reduce learning rates if needed: `--set optim.G.lr=1e-4`
- Increase PatchNCE temperature for stability: `--set patchnce.temperature=0.1`

**Poor image quality**:
- Train longer (baseline reached best MiFID around step 40,000-50,000)
- Check EMA is working (generation uses EMA weights, not raw weights)

**Out of memory**:
- Reduce batch size: `--set batch_size=8`
- Disable AMP if needed: `--set io.amp=false` (not recommended)

## Design Philosophy

This is the **minimum viable GAN** that achieved strong results. It contains:
- ✅ Core CUT architecture (ResNet-9 + PatchGAN + PatchNCE)
- ✅ Proven training techniques (EMA, DiffAugment, Lazy R1)
- ✅ Safety mechanisms (gradient clipping, NaN detection)
- ✅ Simple, understandable code

It does **NOT** contain:
- ❌ Experimental architectural additions
- ❌ Experimental loss functions
- ❌ Complex evaluation pipelines (handled in separate EVAL folder)
- ❌ Kaggle-specific code

## Citation

This baseline is based on:
- **CUT**: Contrastive Learning for Unpaired Image-to-Image Translation (Park et al., 2020)
- **DiffAugment**: Differentiable Augmentation for Data-Efficient GAN Training (Zhao et al., 2020)

---

**Status**: Baseline Restored  
**Last Updated**: 2025-11-04  
**Expected MiFID**: 103-105
