# GAN_Variant1: CUT++ Baseline Implementation

## Overview

This repository contains a complete implementation of a **Contrastive Unpaired Translation (CUT++)** GAN for photo-to-Monet style transfer. The implementation is based on the CUT framework with several baseline optimizations and is designed as a stable, production-ready training system.

**Key Characteristics:**
- **Architecture**: ResNet-9 generator with PatchGAN discriminator
- **Training Method**: CUT (Contrastive Unpaired Translation) with PatchNCE loss
- **Loss Functions**: Hinge adversarial loss, PatchNCE contrastive loss, Identity loss (warmup)
- **Regularization**: R1 gradient penalty (lazy), DiffAugment data augmentation
- **Optimization**: Adam optimizer with cosine learning rate scheduling, EMA (Exponential Moving Average)
- **Mixed Precision**: Automatic Mixed Precision (AMP) for faster training

---

## Table of Contents

1. [Comparison: GAN_Variant1 vs Basic_GAN](#comparison-gan_variant1-vs-basic_gan)
2. [Architecture](#architecture)
3. [Loss Functions](#loss-functions)
4. [Training Process](#training-process)
5. [Technical Decisions](#technical-decisions)
6. [File Structure](#file-structure)
7. [Usage](#usage)

---

## Comparison: GAN_Variant1 vs Basic_GAN

This section highlights the key differences between **GAN_Variant1** (CUT-based implementation) and **Basic_GAN** (CycleGAN-based baseline) in this project.

### Core Framework Differences

| Aspect | Basic_GAN (CycleGAN) | GAN_Variant1 (CUT) |
|--------|----------------------|-------------------|
| **Framework** | CycleGAN (bidirectional translation) | CUT (Contrastive Unpaired Translation) |
| **Generators** | Two generators: `G_A2B` (photo→Monet) and `G_B2A` (Monet→photo) | Single generator: photo→Monet only |
| **Discriminators** | Two discriminators: `D_A` (photo domain) and `D_B` (Monet domain) | Single discriminator (single-scale PatchGAN) |
| **Translation Direction** | Bidirectional (both photo→Monet and Monet→photo) | Unidirectional (photo→Monet only) |

### Loss Function Differences

**Basic_GAN (CycleGAN) uses:**
- **GAN Loss**: LSGAN (MSE) or BCE loss against target labels
- **Cycle Consistency Loss**: `λ_cycle * ||G_B2A(G_A2B(photo)) - photo||₁` (ensures bidirectional reconstruction)
- **Identity Loss**: `λ_identity * ||G_A2B(monet) - monet||₁` (preserves colors when input is already in target domain)

**GAN_Variant1 (CUT) uses:**
- **Hinge Adversarial Loss**: More stable than LSGAN/BCE, provides better gradients
- **PatchNCE Contrastive Loss**: Core innovation - ensures corresponding patches in input/output are similar in feature space (extracts features from multiple generator layers)
- **Identity Loss (Warmup)**: Only active during warmup phase (first 20k steps), then disabled
- **R1 Regularization**: Gradient penalty applied lazily (every 16 steps) for discriminator stability

### Training Methodology Differences

| Aspect | Basic_GAN | GAN_Variant1 |
|--------|-----------|--------------|
| **Batch Size** | 1 (CycleGAN standard) | 12 (enables better contrastive learning) |
| **Training Duration** | 200 epochs | 70 epochs |
| **Learning Rate Schedule** | Linear decay after epoch 100 | Cosine annealing (smooth decay) |
| **Data Augmentation** | Standard transforms (resize, crop, flip) | DiffAugment (differentiable: color, translation, cutout) |
| **EMA (Exponential Moving Average)** | ❌ Not used | ✅ EMA with decay=0.999 for stable inference |
| **Gradient Clipping** | ❌ Not used | ✅ Clipping (max_norm=10.0) for both G and D |
| **R1 Regularization** | ❌ Not used | ✅ Lazy R1 (every 16 steps, γ=10.0) |

### Key Architectural Advantages of GAN_Variant1

1. **More Efficient Training**: 
   - Single generator instead of two (lower memory, faster training)
   - Can use larger batch sizes (12 vs 1) for better contrastive learning
   - Fewer forward passes per training step

2. **Better Content Preservation**:
   - PatchNCE directly enforces feature-level correspondence between input and output patches
   - Extracts features from multiple generator layers `[0, 4, 8, 12, 16]` at different scales
   - More explicit content preservation compared to cycle consistency

3. **Advanced Training Features**:
   - **EMA**: Provides more stable generator weights for inference
   - **DiffAugment**: Differentiable data augmentation that allows gradients to flow through
   - **R1 Regularization**: Improves discriminator training stability
   - **Gradient Clipping**: Prevents gradient explosion
   - **NaN Detection**: Automatic error detection for training stability

4. **Modern Loss Functions**:
   - Hinge loss is more stable than LSGAN/BCE
   - PatchNCE enables unpaired translation without requiring cycle consistency


### Performance Notes

- **Basic_GAN**: Standard CycleGAN baseline, achieves competitive results with proper training
- **GAN_Variant1**: Baseline configuration achieves **66 MiFID**, demonstrating strong performance with the CUT framework

---

## Architecture

### Generator: ResNet-9

The generator follows a U-Net-like architecture with residual blocks, implementing a vanilla ResNet-9 design (no attention mechanisms in the baseline configuration).

#### Architecture Components

1. **Initial Convolution Block**
   - 7×7 convolution with reflection padding
   - Instance normalization
   - ReLU activation
   - Output: `ngf` channels (default: 64)

2. **Downsampling Block**
   - Two downsampling layers
   - Each layer: 3×3 conv (stride=2) → InstanceNorm → ReLU
   - Channel progression: `ngf` → `2*ngf` → `4*ngf`
   - Final feature map: 64×64 (from 256×256 input)

3. **Residual Blocks (9 blocks)**
   - Each block contains:
     - Two 3×3 convolutions with reflection padding
     - Instance normalization after each conv
     - ReLU activation (first conv only)
     - Residual connection: `output = input + conv_block(input)`
   - All blocks operate at `4*ngf = 256` channels
   - **Baseline**: No self-attention, channel attention, or style dropout

4. **Upsampling Block**
   - Two upsampling layers (transposed convolutions)
   - Each layer: 3×3 transpose conv (stride=2) → InstanceNorm → ReLU
   - Channel progression: `4*ngf` → `2*ngf` → `ngf`

5. **Output Block**
   - 7×7 convolution with reflection padding
   - Tanh activation (outputs in [-1, 1])

#### Feature Extraction for PatchNCE

The generator implements `get_feature_layers()` to extract intermediate features at multiple layers:
- Layers extracted: `[0, 4, 8, 12, 16]` (configurable)
- Layer 0: After initial conv
- Layers 1-2: After downsampling layers
- Layers 3-11: After each residual block
- Layers 12-13: After upsampling layers
- These features are used for PatchNCE contrastive learning

**Technical Decision**: The baseline uses a pure ResNet architecture without attention mechanisms to ensure stability and reproducibility. Attention layers (self-attention, channel attention) and style dropout are available in the code but disabled in the baseline configuration.

---

### Discriminator: Single-Scale PatchGAN

The discriminator uses a **single-scale PatchGAN** architecture (70×70 receptive field) in the baseline configuration.

#### Architecture Components

1. **PatchGAN Discriminator**
   - Input: 3-channel RGB images
   - Architecture:
     - Conv1: 4×4, stride=2, padding=1 → `ndf` channels (64)
     - Conv2: 4×4, stride=2, padding=1 → `2*ndf` channels (128)
     - Conv3: 4×4, stride=1, padding=1 → `4*ndf` channels (256)
     - Output: 4×4, stride=1, padding=1 → 1 channel (patch predictions)
   - All layers use LeakyReLU(0.2) except output
   - Output: Patch-level predictions (not pixel-level)

2. **Multiscale Support**
   - The code supports multiscale discriminators (multiple scales of the input)
   - **Baseline**: `num_scales=1` (single-scale only)
   - When multiscale is enabled, inputs are downsampled using average pooling before each discriminator

3. **Spectral Normalization**
   - Available but **disabled in baseline** (`use_spectral_norm=false`)
   - Can be enabled for additional training stability

**Technical Decision**: Single-scale PatchGAN was chosen for the baseline to match the original CUT setup and ensure stable training. Multiscale discriminators can improve results but add complexity.

---

## Loss Functions

### 1. Adversarial Loss (Hinge Loss)

**Discriminator Loss:**
```
L_D = E[max(0, 1 - D(real))] + E[max(0, 1 + D(fake))]
```

**Generator Loss:**
```
L_G_adv = -E[D(fake)]
```

- **Purpose**: Train the discriminator to distinguish real from fake, and the generator to fool the discriminator
- **Implementation**: `losses/adv_hinge.py`
- **Weight**: `loss_weights.adv = 1.0`

**Technical Decision**: Hinge loss is more stable than vanilla GAN loss (logistic) and provides better gradients, especially in the early stages of training.

---

### 2. PatchNCE Loss (Contrastive Loss)

**Formula:**
```
L_NCE = -log(exp(sim(q, k+)) / (exp(sim(q, k+)) + Σ exp(sim(q, k-))))
```

Where:
- `q`: Query patches from generated image features
- `k+`: Positive patches (corresponding patches from input image)
- `k-`: Negative patches (other patches from input image)

**Implementation Details:**
- Extracts features from multiple generator layers: `[0, 4, 8, 12, 16]`
- Samples 256 patches per image (configurable)
- Uses cosine similarity with temperature `τ = 0.07`
- Normalizes features with epsilon for numerical stability
- Clamps logits to [-50, 50] to prevent overflow

**Purpose**: Ensures that corresponding patches in the input and output images are similar in feature space, preserving content structure while allowing style transfer.

**Weight**: `loss_weights.patchnce = 1.0`

**Technical Decision**: PatchNCE is the core innovation of CUT, enabling unpaired image translation without cycle consistency. The temperature and number of patches were tuned for stability.

---

### 3. Identity Loss (Warmup)

**Formula:**
```
L_idt = ||G(monet) - monet||_1
```

**Purpose**: During warmup, prevents the generator from making unnecessary changes when the input is already in the target domain (Monet style).

**Weight Schedule:**
- Warmup phase (first `warmup_steps`): `identity_warm = 0.1`
- After warmup: `identity_final = 0.0` (disabled)
- Linear interpolation during warmup

**Technical Decision**: Identity loss helps stabilize early training by preventing the generator from over-transforming Monet images. It's gradually phased out as training progresses.

---

### 4. R1 Regularization (Gradient Penalty)

**Formula:**
```
L_R1 = γ * E[||∇D(real)||²]
```

**Implementation:**
- Applied **lazily** (every 16 steps) to reduce computational cost
- Computes gradients of discriminator output w.r.t. real images
- Weight: `r1.gamma = 10.0`
- Frequency: `r1.every = 16`

**Purpose**: Regularizes the discriminator to have smooth gradients, preventing mode collapse and improving training stability.

**Technical Decision**: Lazy R1 (applied every N steps) provides most of the regularization benefit at a fraction of the computational cost.

---

## Training Process

### Training Loop Structure

1. **Initialization**
   - Load configuration from YAML
   - Set random seeds for reproducibility
   - Initialize models, optimizers, EMA, AMP scaler
   - Create data loaders

2. **Per-Step Training**
   - **Discriminator Update**:
     - Generate fake images: `fake = G(photos)`
     - Apply DiffAugment to both real and fake
     - Compute hinge loss
     - Backward pass with gradient clipping
     - (Every 16 steps) Compute and apply R1 regularization
   
   - **Generator Update**:
     - Generate fake images: `fake = G(photos)`
     - Apply DiffAugment
     - Compute adversarial loss
     - Compute PatchNCE loss (extract features from multiple layers)
     - Compute identity loss (if in warmup phase)
     - Total loss: weighted sum of all losses
     - Backward pass with gradient clipping
     - Update EMA

3. **Logging & Checkpointing**
   - Log losses every 100 steps
   - Save checkpoint every 2000 steps
   - Track losses in CSV for plotting

### Data Augmentation: DiffAugment

**Policy**: `['color', 'translation', 'cutout']`

1. **Color Augmentations**:
   - Random brightness adjustment
   - Random saturation adjustment
   - Random contrast adjustment

2. **Translation**: Random spatial translation (12.5% of image size)

3. **Cutout**: Random patch erasure (50% of image size)

**Purpose**: Increases data diversity and improves generalization. Applied only to discriminator inputs to prevent overfitting.

**Technical Decision**: DiffAugment is differentiable, allowing gradients to flow through augmentations, unlike traditional data augmentation.

---

## Technical Decisions

### 1. Architecture Choices

**Generator: ResNet-9 (No Attention)**
- **Rationale**: Baseline stability and reproducibility
- Attention mechanisms (self-attention, channel attention) are available but disabled
- Style dropout (AdaIN gates) is available but disabled
- Pure ResNet architecture ensures consistent behavior

**Discriminator: Single-Scale PatchGAN**
- **Rationale**: Matches original CUT setup
- Simpler than multiscale, faster to train
- 70×70 receptive field is sufficient for style transfer

### 2. Loss Function Choices

**Hinge Loss over Logistic Loss**
- More stable gradients
- Better convergence properties
- Standard in modern GAN training

**PatchNCE over CycleGAN**
- Enables unpaired training without cycle consistency
- More efficient (no need for reverse generator)
- Better content preservation

**Identity Loss with Warmup**
- Stabilizes early training
- Prevents over-transformation of target domain images
- Gradually phased out to allow full style transfer

### 3. Optimization Choices

**Adam Optimizer**
- Learning rate: `2e-4` (standard for GANs)
- Betas: `[0.5, 0.999]` (standard GAN betas)
- Weight decay: `0.0` (no L2 regularization)

**Cosine Learning Rate Schedule**
- Initial LR: `2e-4`
- Minimum LR: `5e-5`
- Smooth decay prevents sudden training instability

**Gradient Clipping**
- Generator: `max_grad_norm = 10.0`
- Discriminator: `max_grad_norm = 10.0`
- Prevents gradient explosion

**Exponential Moving Average (EMA)**
- Decay: `0.999`
- Warmup: 100 steps
- Provides more stable generator for inference

### 4. Regularization Choices

**R1 Regularization (Lazy)**
- Applied every 16 steps (lazy R1)
- Weight: `10.0`
- Balances regularization benefit with computational cost

**DiffAugment**
- Lightweight augmentation policy
- Applied only to discriminator inputs
- Differentiable (gradients flow through)

### 5. Training Stability Choices

**Mixed Precision Training (AMP)**
- Reduces memory usage
- Speeds up training (~2x)
- Automatic loss scaling prevents underflow

**NaN Detection**
- Checks for NaN losses after each step
- Raises error immediately to prevent corrupted checkpoints
- Helps debug training issues early

**Checkpointing Strategy**
- Saves every 2000 steps
- Keeps last 5 checkpoints
- Includes full state: models, optimizers, EMA, scaler, config

### 6. Data Pipeline Choices

**Image Preprocessing**
- Random crop + resize (scale: 0.85-1.0)
- Random horizontal flip (p=0.5)
- Color jitter (subtle)
- Normalize to [-1, 1]

**Data Loading**
- 8 workers for parallel loading
- Pin memory for faster GPU transfer
- Prefetch factor: 4
- Drop last batch for consistent batch sizes

---

## File Structure

```
GAN_Variant1/
├── README.md                          # This file
├── requirements.txt                    # Python dependencies
├── generate_folder.py                  # Inference script for batch generation
│
├── configs/
│   └── train_gan_cutpp.yaml           # Main training configuration
│
├── dataio/
│   ├── __init__.py
│   ├── photos_dataset.py              # Photo dataset loader
│   ├── monet_dataset.py               # Monet dataset loader
│   └── transforms.py                  # Image transforms and preprocessing
│
├── losses/
│   ├── __init__.py
│   ├── adv_hinge.py                   # Hinge adversarial loss
│   ├── patchnce_cut.py                # PatchNCE contrastive loss
│   └── identity_l1.py                 # Identity loss (warmup)
│
├── models/
│   ├── __init__.py
│   ├── generator_resnet_attn.py       # ResNet-9 generator
│   └── discriminator_patchgan.py      # PatchGAN discriminator
│
├── training/
│   ├── __init__.py
│   ├── train_cutpp.py                 # Main training script
│   ├── diffaugment.py                 # Differentiable augmentation
│   └── sched_optim.py                 # Optimizer and scheduler setup
│
└── utils/
    ├── __init__.py
    ├── amp_utils.py                   # Mixed precision utilities
    ├── io_ckpt.py                     # Checkpoint I/O and EMA
    ├── loss_tracker.py                # Loss tracking and CSV logging
    ├── plot_losses.py                 # Loss plotting utilities
    └── seed_dist.py                   # Random seed and distributed training setup
```

---

## Usage

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies:**
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- CLIP (for optional features, not used in baseline)
- FAISS (for optional features, not used in baseline)
- PyYAML, tqdm, matplotlib, numpy, scipy, Pillow

### Training

```bash
# Basic training
python -m GAN_Variant1.training.train_cutpp \
    --config GAN_Variant1/configs/train_gan_cutpp.yaml

# Resume from checkpoint
python -m GAN_Variant1.training.train_cutpp \
    --config GAN_Variant1/configs/train_gan_cutpp.yaml \
    --resume GAN_Variant1/checkpoints/ckpt_step4000.pt

# Override config values
python -m GAN_Variant1.training.train_cutpp \
    --config GAN_Variant1/configs/train_gan_cutpp.yaml \
    --set loss_weights.patchnce=2.0 optim.G.lr=1e-4
```

### Inference (Batch Generation)

```bash
python GAN_Variant1/generate_folder.py \
    --ckpt GAN_Variant1/checkpoints/ckpt_final.pt \
    --photos data/photo_jpg \
    --out outputs/variant1_generated \
    --batch 32 \
    --size 256 \
    --device cuda
```

### Expected Outputs

**Training:**
- Checkpoints: `GAN_Variant1/checkpoints/ckpt_stepXXXX.pt`
- Loss logs: `GAN_Variant1/logs/train_log.txt`
- Loss CSV: `GAN_Variant1/logs/losses_history.csv`
- Loss plots: `GAN_Variant1/logs/losses_plot.png`

**Inference:**
- Generated images in the specified output directory

---



