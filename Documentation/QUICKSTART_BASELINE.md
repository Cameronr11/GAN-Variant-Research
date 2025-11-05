# Baseline Training Quick Start

## 1. Setup Environment

```bash
conda create -n GAN310 python=3.10 -y
conda activate GAN310
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pyyaml tqdm pillow scipy
```

## 2. Verify Data Structure

```
../data/
├── photo_jpg/    # 7038 photos
└── monet_jpg/    # 300 Monet paintings
```

## 3. Train Baseline

```bash
cd GAN_Variant1
python -m training.train_cutpp --config configs/train_gan_cutpp.yaml
```

Expected output:
```
Using device: cuda
Photos: 7038, Monet: 300
Training for 60000 steps
Step 100: {"d_loss": 0.45, "g_loss": 2.1, "nce": 1.2, ...}
...
```

## 4. Monitor Training

Watch `logs/train_log.txt`:
- `g_loss` should stabilize around 1.0-2.0
- `nce` should decrease from ~2.0 to ~1.0
- `identity` should decrease to ~0.0 after 20k steps
- No NaN values should appear

## 5. Generate Images

After training reaches step 40k-50k:

```bash
python generate_folder.py \
  --ckpt checkpoints/ckpt_step40000.pt \
  --photos ../data/photo_jpg \
  --out outputs/step40000 \
  --batch 64 \
  --limit 7038 \
  --device cuda
```

Check output:
```bash
ls outputs/step40000/*.jpg | head -5
# Should show 7038 generated images
```

## 6. Evaluate MiFID

```bash
cd ../EVAL
python -m eval.cli \
  --config configs/eval_local.yaml \
  --real ../data/monet_jpg \
  --fake ../GAN_Variant1/outputs/step40000 \
  --batch 64 \
  --workers 8 \
  --device cuda \
  --out cache/reports/baseline_step40000.json
```

Expected MiFID: **103-105**

## Troubleshooting

### Training crashes with NaN
- Check GPU memory: reduce batch_size if needed
- Verify data is loading correctly
- Check logs for specific step where NaN appeared

### Generated images are black
- Verify checkpoint has `ema_G['shadow']` key
- Check `generate_folder.py` logs for which state dict was loaded

### All images look identical
- This was the original bug - should be fixed now
- Verify different checkpoints produce different images:
  ```bash
  md5sum outputs/step40000/*.jpg | head -5
  md5sum outputs/step42000/*.jpg | head -5
  # Hashes should be different
  ```

### MiFID is much higher than 103-105
- Train longer (baseline peaks around step 40-50k)
- Check that EMA weights are being used in generation
- Verify data paths are correct

## Expected Training Time

- **A100 (40GB)**: ~8-10 hours for 60k steps
- **RTX 3090 (24GB)**: ~12-16 hours for 60k steps

## Key Files

- `configs/train_gan_cutpp.yaml` - Config (all baseline settings)
- `training/train_cutpp.py` - Training script
- `generate_folder.py` - Generation script
- `logs/train_log.txt` - Training logs
- `checkpoints/ckpt_step*.pt` - Model checkpoints

## Success Criteria

✅ Training completes without NaN  
✅ Generated images are visually diverse  
✅ Different checkpoints produce different images  
✅ MiFID score: 103-105  

---

If all checks pass, the baseline is validated and ready for experimentation!

