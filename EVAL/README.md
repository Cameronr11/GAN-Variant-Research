# Kaggle MiFID Evaluator

**Standalone evaluation package for "I'm Something of a Painter Myself" competition**

A fast, cache-friendly, Kaggle-faithful evaluator that scores generated Monet-style images using **MiFID** (Memorization-Informed Fréchet Inception Distance)—the primary leaderboard metric for the competition.

---

## What is MiFID?

**MiFID (Memorization-Informed FID)** is a variant of FID that penalizes models for memorizing the training set. It's designed to reward genuine creative synthesis over simple reproduction.

**Formula:** `MiFID = FID / M`

Where:
- **FID**: Standard Fréchet Inception Distance (measures image quality and diversity)
- **M**: Memorization penalty derived from average minimum cosine distance between generated and real features

**Lower MiFID is better.** High similarity to real images (low cosine distances) increases the penalty, resulting in higher MiFID scores.

### Key References
- [TorchMetrics MiFID Documentation](https://lightning.ai/docs/torchmetrics/stable/image/memorization_informed_frechet_inception_distance.html)
- [Kaggle Competition: I'm Something of a Painter Myself](https://www.kaggle.com/competitions/gan-getting-started)
- Competition uses InceptionV3 pool3 (2048-dim) features on 299×299 images

---

## Features

✅ **Kaggle-faithful**: Matches competition metric (InceptionV3 pool3, 299×299, uint8 inputs)  
✅ **Fast**: Cached real statistics, batched feature extraction, GPU-accelerated  
✅ **Comprehensive**: Reports MiFID, FID, and memorization analysis (cosine distances)  
✅ **Standalone**: No dependencies on training code  
✅ **Configurable**: YAML configs + CLI overrides  
✅ **Production-ready**: Handles 7k–10k images, validates counts, checks for data leaks

---

## Installation

```bash
# Navigate to EVAL directory
cd GAN_Project/EVAL

# Install dependencies (recommended: use a virtual environment)
pip install -r requirements_eval.txt
```

**Recommended setup** (use existing GAN310 venv):
```bash
source /nfs/home/crader6/envs/GAN310/bin/activate
pip install -r requirements_eval.txt
```

---

## Quick Start

### 1. Edit Configuration

Open `configs/eval_local.yaml` and update paths:

```yaml
real:
  path: "/path/to/GAN_Project/Basic_GAN/data/monet_jpg"  # Kaggle Monet folder

fake:
  path: "/path/to/your/generated_images"  # 7k–10k generated JPGs
```

### 2. Run Evaluation

```bash
python -m eval.cli \
  --config configs/eval_local.yaml \
  --fake /path/to/generated_images \
  --real /path/to/monet_jpg \
  --out ./cache/reports/my_run.json
```

### 3. View Results

Check the console output for a quick summary, or open:
- **JSON report**: `./cache/reports/my_run.json`
- **Text summary**: `./cache/logs/YYYYMMDD_HHMMSS_run_name.txt`
- **Worst cases CSV**: `./cache/reports/my_run_worst_cases.csv`

---

## Usage Examples

### Evaluate a GAN_Variant1 checkpoint

```bash
python -m eval.cli \
  --config configs/eval_local.yaml \
  --fake ./outputs/GAN_Variant1_epoch50 \
  --batch 64 --workers 8
```

### Quick CPU-only evaluation (no GPU)

```bash
python -m eval.cli \
  --config configs/eval_local.yaml \
  --fake ./outputs/test_images \
  --device cpu \
  --batch 16 --workers 4
```

### Compare multiple checkpoints

```bash
# Run for each checkpoint
for ckpt in epoch10 epoch20 epoch30 epoch40 epoch50; do
  python -m eval.cli \
    --config configs/eval_local.yaml \
    --fake ./outputs/${ckpt} \
    --out ./cache/reports/${ckpt}.json
done

# Compare JSON reports
cat ./cache/reports/epoch*.json | grep "mifid"
```

---

## Configuration

### YAML Structure

```yaml
name: "run_name"                    # Identifier for this evaluation

real:
  mode: "folder"                    # "folder" or "tfds"
  path: "/path/to/monet_jpg"        # Real image directory
  recursive: true                   # Search subdirectories

fake:
  path: "/path/to/generated"        # Generated image directory
  recursive: true

io:
  batch_size: 64                    # Batch size for feature extraction
  num_workers: 8                    # DataLoader workers
  pin_memory: true                  # Pin memory for GPU transfer

metric:
  img_size: 299                     # Resize to 299×299 (InceptionV3)
  cosine_eps: 0.1                   # Epsilon for MiFID cosine distance

cache:
  dir: "./cache"                    # Cache directory for real features

report:
  out_json: "./cache/reports/out.json"  # Output JSON path
```

### CLI Overrides

All config values can be overridden via CLI:

```bash
python -m eval.cli \
  --config configs/eval_local.yaml \
  --fake /new/path \
  --real /new/monet \
  --batch 128 \
  --workers 16 \
  --img-size 299 \
  --device cuda \
  --cosine-eps 0.1 \
  --out ./reports/custom.json
```

---

## Output Schema

### JSON Report

```json
{
  "run": {
    "name": "local_monet_mifid",
    "timestamp_utc": "2025-10-23T22:15:00Z",
    "fake_dir": "/path/to/fakes",
    "real_mode": "folder",
    "real_dir_or_tfds": "/path/to/monet_jpg",
    "num_fake": 9000,
    "num_real": 300,
    "img_size": 299,
    "batch_size": 64,
    "num_workers": 8,
    "warnings": []
  },
  "scores": {
    "mifid": 41.2370,
    "fid": 45.1100,
    "cosine_min_distance": {
      "median": 0.4120,
      "mean": 0.4087,
      "std": 0.0823,
      "p10": 0.3050,
      "p90": 0.5210,
      "hist_bins": [0.2, 0.25, ..., 0.7],
      "hist_counts": [38, 211, ...]
    }
  },
  "hashes": {
    "fake_list_sha1": "abc123...",
    "real_list_sha1": "def456...",
    "real_cache_key": "monet_jpg@sha1:def456..."
  },
  "memorization_analysis": {
    "worst_cases": [
      {
        "fake_path": "/path/to/fake_001.jpg",
        "distance": 0.1234,
        "cosine_similarity": 0.8766,
        "nearest_real_path": "/path/to/monet_42.jpg"
      },
      ...
    ],
    "description": "Top-16 fake images with smallest cosine distance..."
  },
  "notes": "TorchMetrics MiFID/FID with InceptionV3 pool3 (2048 dims)..."
}
```

---

## Understanding Results

### Primary Metrics

| Metric | Description | Goal |
|--------|-------------|------|
| **MiFID** | Memorization-Informed FID | **Lower is better** (Kaggle leaderboard) |
| **FID** | Fréchet Inception Distance | Lower = better image quality |

**Interpreting MiFID vs FID:**
- If `MiFID > FID`: Model is memorizing the real set (high penalty)
- If `MiFID ≈ FID`: Good balance (low memorization)
- If `MiFID < FID`: Unusual (check for bugs)

### Memorization Analysis

**Cosine Min Distance** measures how similar each generated image is to its nearest real image in feature space:
- **Higher distances (>0.5)**: Low memorization risk ✅
- **Medium distances (0.3–0.5)**: Acceptable range ⚠️
- **Lower distances (<0.3)**: High memorization risk ❌

**Worst Cases CSV** lists the 16 generated images most similar to real images—useful for visual inspection.

---

## Caching

Real image features are cached to `./cache/real_feats/{hash}.npz` for fast subsequent runs.

**Cache key** is computed from:
- Sorted list of image paths (relative)
- File sizes

**To invalidate cache:**
```bash
rm -rf ./cache/real_feats/*.npz
```

**Cache contents:**
- `mu`: Mean feature vector [2048]
- `sigma`: Covariance matrix [2048, 2048]
- `features`: Raw features [N, 2048] (for cosine distance)
- `n`: Number of images

---

## Validation & Safety

The evaluator performs several checks:

1. **Count validation**: Warns if fake count is outside 7k–10k or real count < 300
2. **Overlap detection**: Checks for identical filenames between fake/real sets
3. **Image format**: Auto-converts to RGB if needed
4. **Hash fingerprinting**: Detects dataset changes across runs

---

## Performance Tips

### GPU Memory Issues

If you run out of GPU memory:
```bash
python -m eval.cli --config configs/eval_local.yaml --batch 32 --workers 4
```

### Slow DataLoading

If disk I/O is slow (e.g., networked storage):
```bash
# Reduce workers
python -m eval.cli --config configs/eval_local.yaml --workers 2
```

### Approximate Evaluation (Subsample)

For quick iteration:
```bash
# Evaluate on first 3000 images (sorted alphabetically)
# (Copy a subset to a temp folder first)
```

---

## Advanced: TFDS Mode (Optional)

If you have TensorFlow Datasets installed:

```bash
pip install tensorflow-datasets tensorflow

python -m eval.cli \
  --config configs/eval_tfds.yaml \
  --fake /path/to/generated_images
```

This uses the TFDS `cycle_gan/monet2photo` dataset as the real reference.

---

## Troubleshooting

### "No fake images found"
- Check `--fake` path is correct
- Verify images have `.jpg`, `.jpeg`, or `.png` extensions
- Use `--recursive true` if images are in subdirectories

### "torchmetrics not found"
```bash
pip install torchmetrics[image]
```

### "CUDA out of memory"
```bash
# Reduce batch size
python -m eval.cli --config configs/eval_local.yaml --batch 16

# Or use CPU (slow)
python -m eval.cli --config configs/eval_local.yaml --device cpu
```

### Cache issues
```bash
# Clear cache and recompute
rm -rf ./cache/real_feats/*.npz
```

---

## Project Structure

```
EVAL/
├── README.md                      # This file
├── requirements_eval.txt          # Python dependencies
├── configs/
│   ├── eval_local.yaml           # Local Monet JPG config
│   └── eval_tfds.yaml            # TFDS config (optional)
├── eval/
│   ├── __init__.py
│   ├── cli.py                    # Command-line interface
│   ├── datasets.py               # Image loading with transforms
│   ├── features.py               # Inception feature extraction
│   ├── mifid.py                  # MiFID/FID computation
│   ├── report.py                 # JSON/text report generation
│   └── utils.py                  # Utilities (hashing, validation)
├── cache/
│   ├── real_feats/               # Cached real features (*.npz)
│   ├── logs/                     # Text summaries of past runs
│   └── reports/                  # JSON reports
└── scripts/
    └── run_eval.sh               # Example shell scripts
```

---

## Citation & References

This evaluator implements the MiFID metric as described in:

- **TorchMetrics**: https://lightning.ai/docs/torchmetrics/stable/image/memorization_informed_frechet_inception_distance.html
- **Kaggle Competition**: https://www.kaggle.com/competitions/gan-getting-started
- **Original FID Paper**: Heusel et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (2017)

---

## Support

For issues or questions:
1. Check this README
2. Review example configs in `configs/`
3. Run with `--help` for CLI options: `python -m eval.cli --help`

---

## License

This evaluator is part of the GAN_Project codebase. Use freely for academic and competition purposes.

