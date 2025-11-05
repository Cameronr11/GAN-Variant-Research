# Quick Start Guide - Kaggle MiFID Evaluator

Get your MiFID score in 3 steps.

## 1. Install

```bash
cd EVAL
pip install -r requirements_eval.txt
```

## 2. Configure Paths

Edit `configs/eval_local.yaml`:

```yaml
real:
  path: "/path/to/GAN_Project/Basic_GAN/data/monet_jpg"  # Your Monet folder

fake:
  path: "/path/to/your/generated_images"  # Your generated images (7k-10k JPGs)
```

## 3. Run

```bash
python -m eval.cli \
  --config configs/eval_local.yaml \
  --fake /path/to/your/generated_images \
  --real /path/to/monet_jpg \
  --out ./cache/reports/my_run.json
```

## Results

Check console output or open:
- **JSON**: `./cache/reports/my_run.json`
- **Text**: `./cache/logs/YYYYMMDD_HHMMSS_*.txt`

### Key Metric

**MiFID Score** = The number you care about (lower is better)
- This is the Kaggle leaderboard metric
- Typical good scores: 30-60
- <30 = excellent, >100 = needs work

---

## Common Options

```bash
# Use CPU instead of GPU
--device cpu

# Adjust batch size (default: 64)
--batch 32

# More workers for faster loading (default: 8)
--workers 16

# Custom output path
--out ./my_results.json
```

---

## Interpret Results

```json
{
  "scores": {
    "mifid": 41.24,    ← YOUR KAGGLE SCORE (lower = better)
    "fid": 45.11,      ← Quality/diversity baseline
    "cosine_min_distance": {
      "median": 0.41   ← Memorization risk (higher = safer)
    }
  }
}
```

- **MiFID > FID**: Model is memorizing (bad)
- **MiFID ≈ FID**: Good balance
- **Cosine distance > 0.4**: Safe from memorization
- **Cosine distance < 0.3**: High memorization risk

---

## Troubleshooting

**"No fake images found"**
→ Check path, ensure `.jpg`/`.jpeg` extensions

**"CUDA out of memory"**
→ Use `--batch 16` or `--device cpu`

**Slow performance**
→ Use `--workers 4` if on slow disk

---

## Full Documentation

See [README.md](README.md) for comprehensive documentation.

