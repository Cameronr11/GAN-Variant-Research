#!/bin/bash
# Example evaluation scripts for Kaggle MiFID Evaluator

# ==============================================================================
# SETUP: Activate virtual environment
# ==============================================================================
# Uncomment and edit path to your venv:
# source /nfs/home/crader6/envs/GAN310/bin/activate

# ==============================================================================
# EXAMPLE 1: Basic evaluation with config file
# ==============================================================================
echo "Example 1: Basic evaluation"
python -m eval.cli \
  --config configs/eval_local.yaml \
  --fake ../outputs/variant1_epoch50 \
  --real ../Basic_GAN/data/monet_jpg \
  --out ./cache/reports/variant1_epoch50.json \
  --batch 64 \
  --workers 8

# ==============================================================================
# EXAMPLE 2: Quick CPU evaluation (no GPU)
# ==============================================================================
echo "Example 2: CPU evaluation"
python -m eval.cli \
  --config configs/eval_local.yaml \
  --fake ../outputs/test_run \
  --device cpu \
  --batch 16 \
  --workers 4 \
  --out ./cache/reports/test_run_cpu.json

# ==============================================================================
# EXAMPLE 3: Evaluate multiple checkpoints in a loop
# ==============================================================================
echo "Example 3: Multi-checkpoint evaluation"

CHECKPOINTS=("epoch10" "epoch20" "epoch30" "epoch40" "epoch50")
FAKE_DIR_BASE="../outputs"
REAL_DIR="../Basic_GAN/data/monet_jpg"

for ckpt in "${CHECKPOINTS[@]}"; do
  echo "Evaluating checkpoint: $ckpt"
  python -m eval.cli \
    --config configs/eval_local.yaml \
    --fake "${FAKE_DIR_BASE}/${ckpt}" \
    --real "$REAL_DIR" \
    --out "./cache/reports/${ckpt}.json" \
    --batch 64 \
    --workers 8
done

echo "All checkpoints evaluated. Reports in ./cache/reports/"

# ==============================================================================
# EXAMPLE 4: Compare MiFID scores across checkpoints
# ==============================================================================
echo "Example 4: Compare MiFID scores"
echo "Checkpoint | MiFID Score"
echo "-----------|------------"

for report in ./cache/reports/epoch*.json; do
  ckpt=$(basename "$report" .json)
  mifid=$(grep -oP '"mifid":\s*\K[0-9.]+' "$report")
  printf "%-10s | %.4f\n" "$ckpt" "$mifid"
done

# ==============================================================================
# EXAMPLE 5: High-performance evaluation (large batch, more workers)
# ==============================================================================
echo "Example 5: High-performance evaluation"
python -m eval.cli \
  --config configs/eval_local.yaml \
  --fake ../outputs/final_submission \
  --real ../Basic_GAN/data/monet_jpg \
  --out ./cache/reports/final_submission.json \
  --batch 128 \
  --workers 16 \
  --device cuda

# ==============================================================================
# EXAMPLE 6: Debug run with minimal resources
# ==============================================================================
echo "Example 6: Debug/minimal evaluation"
python -m eval.cli \
  --config configs/eval_local.yaml \
  --fake ../outputs/debug_small_set \
  --real ../Basic_GAN/data/monet_jpg \
  --batch 8 \
  --workers 1 \
  --device cpu \
  --out ./cache/reports/debug.json

echo "All examples complete!"

