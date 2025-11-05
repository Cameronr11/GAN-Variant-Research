# CycleGAN - I'm Something of a Painter Myself

Kaggle competition: Photo → Monet style transfer using CycleGAN.

## Project Structure

```
Basic_GAN/
├── configs/          # Configuration files (baseline.yaml)
├── data/            # Training data (Monet and photo images)
├── src/             # Source code
│   ├── data.py      # Dataset and data loading
│   ├── models.py    # Generator and discriminator architectures
│   ├── losses.py    # Loss functions (GAN, cycle, identity)
│   ├── train.py     # Training script
│   └── utils.py     # General utilities
├── checkpoints/     # Saved model checkpoints
├── runs/            # TensorBoard logs
└── requirements.txt
```

**Note**: Inference and evaluation systems are handled by the centralized `Evaluation/` folder at the GAN_Project level.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

Train the CycleGAN baseline:
```bash
python -m src.train --config configs/baseline.yaml
```

## Inference and Evaluation

Inference and evaluation are handled by the centralized system at `GAN_Project/Evaluation/`.

This allows the same evaluation pipeline to be used for all GAN variants (Basic_GAN, GAN_Variant1, etc.).

## Competition Details

**Kaggle Competition**: [I'm Something of a Painter Myself](https://www.kaggle.com/c/gan-getting-started)

**Goal**: Transform photographs into Monet-style paintings using GANs

**Evaluation Metric**: MiFID (Memorization-informed Fréchet Inception Distance)

