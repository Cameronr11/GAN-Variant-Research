# Baseline Reversion Summary

**Date**: 2025-11-04  
**Purpose**: Revert `GAN_Variant1` to the minimal baseline that achieved 103-105 MiFID

## What Was Removed

### 1. Experimental Loss Functions
- `losses/palette_prior_lab.py` - Palette prior loss (Lab color statistics)
- `losses/repulsion_knn.py` - k-NN repulsion loss (CLIP-based anti-memorization)
- `losses/feat_matching.py` - Feature matching loss for stability

### 2. Experimental Architecture
- `models/attention_blocks.py` - Self-attention, channel attention, AdaIN gates
  - Removed all attention mechanism code from `generator_resnet_attn.py`
  - Generator now uses pure ResNet-9 architecture (baseline)
  - Discriminator remains as PatchGAN (baseline uses `num_scales=1` for single-scale)

### 3. Metrics and Evaluation
- `metrics/` folder (entire directory)
  - `clip_knn_distance.py` - CLIP distance computation
  - `fid_inception.py` - Inline FID computation
- Removed `evaluate_metrics()` function from `train_cutpp.py`
- Removed `EarlyStoppingTracker` class
- All evaluation now handled by separate `EVAL` folder

### 4. Config Files
- `configs/train_gan_cutpp.yaml` (old experimental config)
  - Replaced with `train_gan_cutpp_STABLE.yaml` → renamed to `train_gan_cutpp.yaml`

## What Remains (Baseline)

### Core Architecture
- **Generator**: ResNet-9 (no attention, no style dropout)
  - 7x7 initial conv
  - 2x downsampling
  - 9 residual blocks
  - 2x upsampling
  - 7x7 output conv + tanh

- **Discriminator**: Single-scale PatchGAN (70x70)
  - 3 layers
  - No spectral normalization
  - No multiscale wrapper

### Core Losses
- `losses/adv_hinge.py` - Hinge adversarial loss
- `losses/patchnce_cut.py` - PatchNCE contrastive loss (with numerical safeguards)
- `losses/identity_l1.py` - Identity loss (warmup only)

### Training Components
- `training/train_cutpp.py` - Main training script (simplified)
- `training/diffaugment.py` - DiffAugment
- `training/sched_optim.py` - Optimizers and schedulers

### Utilities
- `utils/amp_utils.py` - AMP context (with gradient clipping)
- `utils/io_ckpt.py` - Checkpointing and EMA
- `utils/seed_dist.py` - Seeding

### Data I/O
- `dataio/photos_dataset.py` - Photos dataset
- `dataio/monet_dataset.py` - Monet dataset  
- `dataio/transforms.py` - Augmentation transforms

### Generation
- `generate_folder.py` - Generate images from checkpoints (FIXED to load EMA correctly)

## Key Code Changes

### 1. `training/train_cutpp.py`
- **Removed imports**: `PalettePriorLoss`, `RepulsionKNNLoss`, `feature_matching_loss`, CLIP metrics, FID metrics
- **Removed functions**: `build_clip_features_if_needed()`, `build_repulsion_loss()`, `evaluate_metrics()`, `EarlyStoppingTracker`
- **Simplified `train_step()`**: 
  - Removed `palette_loss_fn` and `repulsion_loss_fn` parameters
  - Removed palette, repulsion, and feature matching loss computations
  - Simplified loss dict to only include: `d_loss`, `g_loss`, `g_adv`, `nce`, `identity`, `r1`
- **Simplified `main()`**:
  - Removed palette, CLIP, repulsion, FID, early stopping initialization
  - Removed inline evaluation loop
  - Kept only checkpoint saving

### 2. `models/generator_resnet_attn.py`
- **Removed import**: `attention_blocks`
- **Simplified `__init__()`**: 
  - Removed `self_attns`, `channel_attns`, `style_gates` ModuleDicts
  - Only creates `res_blocks` ModuleList
- **Simplified `forward()`**: 
  - Removed attention and style dropout application
  - Pure residual block sequence
- **Simplified `get_feature_layers()`**: 
  - Removed attention references

### 3. `utils/amp_utils.py`
- **Added**: `max_grad_norm` parameter to `step_optimizer()`
- **Added**: Gradient norm calculation and clipping
- **Added**: NaN/inf check before optimizer step

### 4. `losses/patchnce_cut.py`
- **Added**: `eps=1e-6` to feature normalization
- **Added**: Logit clamping (`min=-50, max=50`)
- **Added**: NaN checks on `batch_loss` and `total_loss`

### 5. `generate_folder.py`
- **Fixed**: `_pick_state_dict()` to correctly load `ckpt['ema_G']['shadow']` for EMA weights
- **Added**: Informative logging of which state dict is loaded

## Baseline Config (`configs/train_gan_cutpp.yaml`)

```yaml
# Architecture
model.generator:
  use_attention: false
  use_channel_attn: false
  use_style_dropout: false

model.discriminator:
  base: "patchgan"
  num_scales: 1
  use_spectral_norm: false

# Loss weights
loss_weights:
  adv: 1.0
  patchnce: 1.0
  identity_warm: 0.1
  identity_final: 0.0
  palette: 0.0
  repulsion: 0.0
  featmatch: 0.0

# Optimizer (standard GAN betas)
optim:
  G.betas: [0.5, 0.999]
  D.betas: [0.5, 0.999]

# PatchNCE (baseline)
patchnce:
  temperature: 0.07

# Gradient clipping (added for stability)
grad_clip_g: 10.0
grad_clip_d: 10.0
```

## Numerical Stability Additions

These features were **added** to the baseline (they weren't in the original, but are needed to prevent NaN collapses):

1. **Gradient Clipping**: Max norm 10.0 for both G and D
2. **PatchNCE Safeguards**: 
   - Feature normalization epsilon
   - Logit clamping before softmax
   - NaN detection and fallback
3. **Training NaN Detection**: Immediate stop if any loss becomes NaN

These additions do **not** change the architecture or loss formulation - they only add safety rails to prevent numerical instability.

## Expected Performance

- **MiFID**: 103-105 (as achieved in original baseline)
- **Training Time**: ~8-10 hours on A100
- **Convergence**: Best results around steps 40,000-50,000

## Files Deleted Summary

```
GAN_Variant1/
├── losses/
│   ├── palette_prior_lab.py      ❌ DELETED
│   ├── repulsion_knn.py          ❌ DELETED
│   └── feat_matching.py          ❌ DELETED
├── models/
│   └── attention_blocks.py       ❌ DELETED
├── metrics/                      ❌ DELETED (entire folder)
│   ├── __init__.py
│   ├── clip_knn_distance.py
│   └── fid_inception.py
└── configs/
    └── train_gan_cutpp.yaml      ❌ REPLACED (was experimental)
```

## Testing Checklist

Before training:
- [ ] No import errors when running `python -m training.train_cutpp --config configs/train_gan_cutpp.yaml`
- [ ] Config loads successfully
- [ ] Generator and discriminator build without errors
- [ ] First training step executes without NaN

After training:
- [ ] Checkpoints contain `ema_G['shadow']` key
- [ ] `generate_folder.py` loads EMA weights correctly
- [ ] Generated images are visually diverse (not identical)
- [ ] MiFID evaluation produces varied scores for different checkpoints

## Rollback Plan

If the baseline doesn't work as expected, the experimental code can be restored from git history:

```bash
# View deleted files
git log --diff-filter=D --summary

# Restore a specific file
git checkout <commit-hash> -- path/to/file.py
```

Key commits to reference:
- Before reversion: Check git log for last commit before this change
- Experimental config: `configs/train_gan_cutpp.yaml` (before rename)

## Next Steps

1. **Test Training**: Run baseline training for a few hundred steps to verify stability
2. **Test Generation**: Generate images from a checkpoint to verify diversity
3. **Test Evaluation**: Run MiFID evaluation to verify scores are varied
4. **Full Training**: If all checks pass, run full training to 50k steps

Once baseline is confirmed working at 103-105 MiFID, any new features should be added incrementally as deltas from this proven baseline.

---

**Reversion Complete**: 2025-11-04  
**Status**: Ready for baseline validation training

