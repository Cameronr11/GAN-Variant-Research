# CycleGAN - I'm Something of a Painter Myself

Kaggle competition: Photo → Monet style transfer using CycleGAN.

## Project Structure

```
GAN_Project/
├── configs/          # Configuration files (baseline.yaml)
├── data/            # Training data (Monet and photo images)
├── src/             # Source code
│   ├── data.py      # Dataset and data loading
│   ├── models.py    # Generator and discriminator architectures
│   ├── losses.py    # Loss functions (GAN, cycle, identity)
│   ├── train.py     # Training script
│   ├── infer.py     # Inference/generation script
│   ├── utils.py     # General utilities
│   ├── eval_metrics.py    # FID, KID, memorization proxy
│   ├── utils_eval.py      # Evaluation helpers
│   └── eval.py            # Offline evaluation CLI
├── checkpoints/     # Saved model checkpoints
├── runs/            # TensorBoard logs
├── results/         # Evaluation results (JSON, CSV)
└── requirements.txt
```

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

## Inference

Generate Monet-style images from photos:
```bash
python -m src.infer --checkpoint checkpoints/ckpt_e60.pt --input_dir data/photo_jpg --output_dir submissions/generated
```

## Offline Evaluation

The evaluation system computes metrics that approximate Kaggle's MiFID (Memorization-informed FID) leaderboard score locally without hitting the server.

### Metrics Computed

- **FID** (Fréchet Inception Distance): Measures the quality and diversity of generated images compared to real Monet paintings
- **KID** (Kernel Inception Distance): Alternative metric, optional but provides additional validation
- **Memorization Proxy**: Average minimum cosine distance in InceptionV3 feature space, measuring how much the model memorizes training data
- **MiFID'**: Heuristic approximation of Kaggle's MiFID = FID / max(d_avg, epsilon)

### Quick Local Check (Subset)

For fast iteration during development:

```bash
python -m src.eval \
  --gen_dir submissions/group_X_basic \
  --subset_real 1000 \
  --subset_gen 1000 \
  --batch_size 128 \
  --epsilon 0.1
```

### Full Evaluation (Production)

For final evaluation with all available images:

```bash
python -m src.eval \
  --real_dir data/monet_jpg \
  --gen_dir submissions/group_X_basic \
  --subset_real 300 \
  --subset_gen 3000 \
  --batch_size 256 \
  --epsilon 0.1
```

### On ISAAC Cluster (GPU)

```bash
python -m src.eval \
  --real_dir /lustre/isaac24/scratch/crader6/GAN_Project/data/monet_jpg \
  --gen_dir /lustre/isaac24/scratch/crader6/GAN_Project/submissions/group_X_basic \
  --subset_real 300 \
  --subset_gen 3000 \
  --batch_size 256 \
  --epsilon 0.1
```

### Command-Line Options

- `--real_dir`: Directory containing real Monet images (default: from config `data.root/data.domain_b`)
- `--gen_dir`: Directory containing generated images (**required**)
- `--subset_real`: Max number of real images to use (default: 1000, use `None` for all)
- `--subset_gen`: Max number of generated images to use (default: 1000, use `None` for all)
- `--batch_size`: Batch size for embedding extraction (default: 128)
- `--epsilon`: Epsilon for MiFID' calculation (default: 0.1)
- `--no_kid`: Skip KID computation for faster evaluation
- `--out`: Output JSON file path (default: `results/eval_<timestamp>.json`)
- `--seed`: Random seed for reproducibility (default: 42)
- `--num_workers`: Number of workers for data loading (default: 0)

### Output

The evaluation script produces:

1. **Console output**: Human-readable summary table
2. **JSON file**: `results/eval_<timestamp>.json` with detailed metrics
3. **CSV log**: `results/eval_log.csv` accumulating all runs for tracking progress

Example console output:
```
================================================================================
Evaluation @ 2025-10-19 11:30:00
================================================================================
Real:       data/monet_jpg
            (300 available, using 300)
Generated:  submissions/group_X_basic
            (7038 available, using 1000)
================================================================================

Computing FID...
FID: 48.73

Computing KID...
KID: 2.31e-2

Computing memorization proxy...
Extracting real embeddings: 100%|██████████| 5/5 [00:10<00:00]
Extracting gen embeddings: 100%|██████████| 16/16 [00:32<00:00]
Computing min distances: 100%|██████████| 16/16 [00:05<00:00]

Memorization (avg min cosine dist): 0.2140

================================================================================
RESULTS SUMMARY
================================================================================
FID:                        48.73
KID:                        2.31e-2
Memorization (avg min cosine dist): 0.2140
MiFID' (heuristic):         227.71
                            (FID / max(d_avg, 0.1))
================================================================================

Saved: results/eval_20251019_113000.json
Appended row to: results/eval_log.csv

Evaluation complete!
```

### Notes

- The evaluation automatically detects GPU availability and uses it for faster computation
- Images are processed in batches to avoid OOM errors
- Invalid or corrupt images are skipped with a warning
- The `MiFID'` score is a **heuristic approximation** since Kaggle's exact epsilon threshold is not public
- Use consistent `--subset_real` and `--subset_gen` values across runs for fair comparisons

## Competition Details

**Kaggle Competition**: [I'm Something of a Painter Myself](https://www.kaggle.com/c/gan-getting-started)

**Goal**: Transform photographs into Monet-style paintings using GANs

**Evaluation Metric**: MiFID (Memorization-informed Fréchet Inception Distance)

