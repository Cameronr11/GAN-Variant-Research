"""
Main training script for CUT++ (Track A).
Single entry point for all ablations.
"""
import sys
import os
from pathlib import Path
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Imports from our modules
from GAN_Variant1.dataio.photos_dataset import PhotosDataset
from GAN_Variant1.dataio.monet_dataset import MonetDataset
from GAN_Variant1.dataio.transforms import get_train_transforms, rgb_to_lab, denormalize
from GAN_Variant1.models.generator_resnet_attn import ResNetGenerator
from GAN_Variant1.models.discriminator_patchgan import MultiscaleDiscriminator
from GAN_Variant1.losses.adv_hinge import discriminator_hinge_loss, generator_hinge_loss
from GAN_Variant1.losses.patchnce_cut import compute_patchnce_loss
from GAN_Variant1.losses.identity_l1 import identity_loss
from GAN_Variant1.training.diffaugment import DiffAugment
from GAN_Variant1.training.sched_optim import get_optimizer
from GAN_Variant1.utils.seed_dist import set_seed
from GAN_Variant1.utils.amp_utils import AMPContext
from GAN_Variant1.utils.io_ckpt import EMA, save_checkpoint, load_checkpoint
from GAN_Variant1.utils.loss_tracker import LossTracker
from GAN_Variant1.utils.plot_losses import plot_training_losses


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CUT++ GAN')
    parser.add_argument('--config', type=str, default='GAN_Variant1/configs/train_gan_cutpp.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--set', nargs='+', default=[],
                        help='Override config values (e.g., loss_weights.palette=0.5)')
    return parser.parse_args()


def override_config(config, overrides):
    """Override config values from command line."""
    for override in overrides:
        if '=' not in override:
            continue
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        
        # Navigate to the right nested dict
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value (try to parse as float/int/bool)
        try:
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
        except AttributeError:
            pass
        
        current[keys[-1]] = value
    
    return config


def build_models(config, device):
    """Build generator and discriminator."""
    gen_cfg = config['model']['generator']
    disc_cfg = config['model']['discriminator']
    
    # Generator
    generator = ResNetGenerator(
        input_nc=3,
        output_nc=3,
        ngf=gen_cfg['ngf'],
        n_blocks=gen_cfg['n_blocks'],
        n_downsampling=gen_cfg['n_downsampling'],
        padding_type=gen_cfg['padding_type'],
        norm=gen_cfg['norm'],
        activation=gen_cfg['activation'],
        use_attention=gen_cfg.get('use_attention', True),
        attn_layers=gen_cfg.get('attn_layers', [3, 7]),
        use_channel_attn=gen_cfg.get('use_channel_attn', True),
        channel_attn_layers=gen_cfg.get('channel_attn_layers', [5]),
        use_style_dropout=gen_cfg.get('use_style_dropout', True),
        alpha_min=gen_cfg.get('style_dropout', {}).get('alpha_min', 0.4),
        alpha_max=gen_cfg.get('style_dropout', {}).get('alpha_max', 0.9)
    ).to(device)
    
    # Discriminator
    discriminator = MultiscaleDiscriminator(
        input_nc=3,
        ndf=disc_cfg['ndf'],
        n_layers=disc_cfg['n_layers'],
        num_scales=disc_cfg['num_scales'],
        use_spectral_norm=disc_cfg.get('use_spectral_norm', True)
    ).to(device)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    return generator, discriminator


def build_dataloaders(config):
    """Build training dataloaders."""
    data_cfg = config['data']
    
    # Transforms
    transform = get_train_transforms(config['image_size'])
    
    # Datasets
    photos_dataset = PhotosDataset(data_cfg['photos_dir'], transform)
    monet_dataset = MonetDataset(data_cfg['monet_dir'], transform)
    
    # Dataloaders
    photos_loader = DataLoader(
        photos_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 8),
        prefetch_factor=config.get('prefetch_factor', 4),
        pin_memory=config.get('pin_memory', True),
        drop_last=True
    )
    
    monet_loader = DataLoader(
        monet_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 8),
        prefetch_factor=config.get('prefetch_factor', 4),
        pin_memory=config.get('pin_memory', True),
        drop_last=True
    )
    
    return photos_loader, monet_loader, monet_dataset


# Removed CLIP and repulsion helper functions - not part of baseline


def r1_regularization(discriminator, real_images, amp_ctx):
    """
    Compute R1 gradient penalty.
    
    Args:
        discriminator: discriminator model
        real_images: (B, 3, H, W) real images
        amp_ctx: AMP context with scaler
    
    Returns:
        r1_loss: R1 penalty
    """
    real_images.requires_grad_(True)
    
    with torch.amp.autocast('cuda', enabled=False):
        real_pred = discriminator(real_images.float())
        
        # Handle multiscale
        if isinstance(real_pred, list):
            real_pred = sum([pred.sum() for pred in real_pred])
        else:
            real_pred = real_pred.sum()
    
    # Compute gradients
    grad_real = torch.autograd.grad(
        outputs=amp_ctx.scaler.scale(real_pred) if amp_ctx.enabled else real_pred,
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    if amp_ctx.enabled:
        grad_real = grad_real / amp_ctx.scaler.get_scale()
    
    # R1 penalty
    r1_loss = grad_real.pow(2).reshape(grad_real.size(0), -1).sum(1).mean()
    
    return r1_loss


def train_step(
    step,
    photos,
    monets,
    generator,
    discriminator,
    opt_G,
    opt_D,
    ema_G,
    amp_ctx,
    diffaugment,
    config,
    device
):
    """Single training step."""
    loss_weights = config['loss_weights']
    
    # Compute identity weight (linear annealing during warmup)
    warmup_steps = config.get('warmup_steps', 20000)
    if step < warmup_steps:
        identity_weight = loss_weights['identity_warm'] + (loss_weights['identity_final'] - loss_weights['identity_warm']) * (step / warmup_steps)
    else:
        identity_weight = loss_weights['identity_final']
    
    # ======================
    # Train Discriminator
    # ======================
    opt_D.zero_grad()
    
    with amp_ctx.autocast():
        # Generate fake images
        fake_photos_monet = generator(photos)
        
        # Apply DiffAugment to discriminator inputs
        if diffaugment is not None:
            photos_aug = diffaugment(photos)
            fake_aug = diffaugment(fake_photos_monet.detach())
        else:
            photos_aug = photos
            fake_aug = fake_photos_monet.detach()
        
        # Discriminator predictions
        real_pred = discriminator(photos_aug)
        fake_pred = discriminator(fake_aug)
        
        # Hinge loss
        d_loss = discriminator_hinge_loss(real_pred, fake_pred)
    
    # Backward with gradient clipping
    amp_ctx.scale_backward(d_loss)
    amp_ctx.step_optimizer(opt_D, max_grad_norm=config.get('grad_clip_d', 10.0))
    
    # R1 regularization (lazy)
    r1_loss = torch.tensor(0.0, device=device)
    if config['r1']['gamma'] > 0 and step % config['r1']['every'] == 0:
        opt_D.zero_grad()
        r1_loss = r1_regularization(discriminator, photos, amp_ctx)
        r1_weighted = r1_loss * config['r1']['gamma'] * config['r1']['every']
        amp_ctx.scale_backward(r1_weighted)
        amp_ctx.step_optimizer(opt_D, max_grad_norm=config.get('grad_clip_d', 10.0))
    
    # ======================
    # Train Generator
    # ======================
    opt_G.zero_grad()
    
    with amp_ctx.autocast():
        # Generate fake images
        fake_photos_monet = generator(photos)
        
        # Apply DiffAugment
        if diffaugment is not None:
            fake_aug = diffaugment(fake_photos_monet)
        else:
            fake_aug = fake_photos_monet
        
        # Adversarial loss
        fake_pred = discriminator(fake_aug)
        g_adv_loss = generator_hinge_loss(fake_pred)
        
        # PatchNCE loss
        nce_loss = torch.tensor(0.0, device=device)
        if loss_weights['patchnce'] > 0:
            nce_loss = compute_patchnce_loss(
                generator,
                photos,
                fake_photos_monet,
                nce_layers=config['patchnce']['nce_layers'],
                temperature=config['patchnce']['temperature'],
                num_patches=config['patchnce']['num_patches']
            )
        
        # Identity loss (warmup)
        idt_loss = torch.tensor(0.0, device=device)
        if identity_weight > 0:
            idt_loss = identity_loss(generator, monets)
        
        # Total generator loss (baseline: adv + patchnce + identity)
        g_loss = (
            loss_weights['adv'] * g_adv_loss +
            loss_weights['patchnce'] * nce_loss +
            identity_weight * idt_loss
        )
    
    # Backward with gradient clipping
    amp_ctx.scale_backward(g_loss)
    amp_ctx.step_optimizer(opt_G, max_grad_norm=config.get('grad_clip_g', 10.0))
    
    # Update EMA
    if ema_G is not None:
        ema_G.update()
    
    # Collect losses with NaN detection (baseline only)
    losses = {
        'd_loss': d_loss.item(),
        'g_loss': g_loss.item(),
        'g_adv': g_adv_loss.item(),
        'nce': nce_loss.item(),
        'identity': idt_loss.item(),
        'r1': r1_loss.item(),
        'identity_weight': identity_weight
    }
    
    # Check for NaN and raise error if found
    if any(not torch.isfinite(torch.tensor(v)).item() for k, v in losses.items() if k != 'identity_weight'):
        print(f"\n⚠️  NaN detected at step {step}!")
        print(f"Losses: {losses}")
        raise ValueError(f"NaN loss detected at step {step}. Training stopped to prevent corruption.")
    
    return losses


# Removed evaluate_metrics function - evaluation done separately in EVAL folder


# Removed EarlyStoppingTracker - baseline trains for fixed number of steps


def main():
    """Main training loop."""
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config
    config = override_config(config, args.set)
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    Path(config['output']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['output']['log_dir']).mkdir(parents=True, exist_ok=True)
    
    # Initialize loss tracker
    loss_tracker = LossTracker(config['output']['log_dir'])
    loss_tracker.start()
    
    # Build dataloaders
    photos_loader, monet_loader, monet_dataset = build_dataloaders(config)
    print(f"Photos: {len(photos_loader.dataset)}, Monet: {len(monet_loader.dataset)}")
    
    # Build models
    generator, discriminator = build_models(config, device)
    
    # Optimizers
    opt_G = get_optimizer(generator, config['optim']['G'])
    opt_D = get_optimizer(discriminator, config['optim']['D'])
    
    # EMA
    ema_G = EMA(generator, decay=config['ema']['decay'])
    
    # AMP
    amp_ctx = AMPContext(enabled=config.get('amp', True))
    
    # DiffAugment
    diffaugment = None
    if config['diffaugment'].get('enable', False):
        diffaugment = DiffAugment(config['diffaugment'].get('policy', ['color', 'translation', 'cutout']))
    
    # Baseline: No palette, CLIP, repulsion, FID, or early stopping
    
    # Resume from checkpoint
    start_step = 0
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume, generator, discriminator, opt_G, opt_D, ema_G, amp_ctx.scaler, device
        )
        start_step = checkpoint['step']
        print(f"Resumed from step {start_step}")
    
    # Training loop
    max_steps = config.get('max_steps', None)
    if max_steps is None:
        max_steps = config['epochs'] * len(photos_loader)
    
    print(f"Training for {max_steps} steps")
    
    generator.train()
    discriminator.train()
    
    step = start_step
    photos_iter = iter(photos_loader)
    monet_iter = iter(monet_loader)
    
    pbar = tqdm(total=max_steps - start_step, initial=0, desc="Training")
    
    loss_accumulator = defaultdict(list)
    
    while step < max_steps:
        # Get batches
        try:
            photos = next(photos_iter)
        except StopIteration:
            photos_iter = iter(photos_loader)
            photos = next(photos_iter)
        
        try:
            monets = next(monet_iter)
        except StopIteration:
            monet_iter = iter(monet_loader)
            monets = next(monet_iter)
        
        photos = photos.to(device)
        monets = monets.to(device)
        
        # Training step
        losses = train_step(
            step, photos, monets, generator, discriminator,
            opt_G, opt_D, ema_G, amp_ctx, diffaugment,
            config, device
        )
        
        # Accumulate losses
        for k, v in losses.items():
            loss_accumulator[k].append(v)
        
        # Track losses for plotting
        loss_tracker.log(step, losses['d_loss'], losses['g_loss'])
        
        # Logging
        if step % config.get('log_every', 100) == 0 and step > 0:
            avg_losses = {k: np.mean(v) for k, v in loss_accumulator.items()}
            loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
            pbar.set_postfix_str(loss_str)
            
            # Write to log
            log_path = Path(config['output']['log_dir']) / 'train_log.txt'
            with open(log_path, 'a') as f:
                f.write(f"Step {step}: {json.dumps(avg_losses)}\n")
            
            loss_accumulator.clear()
        
        # No inline evaluation - use external EVAL folder after training
        
        # Save checkpoint
        if step % config['metrics']['save_checkpoint_every'] == 0 and step > 0:
            ckpt_path = Path(config['output']['checkpoint_dir']) / f'ckpt_step{step}.pt'
            save_checkpoint(
                str(ckpt_path), step, generator, discriminator, opt_G, opt_D,
                ema_G, amp_ctx.scaler, config=config
            )
            print(f"\nSaved checkpoint to {ckpt_path}")
        
        step += 1
        pbar.update(1)
    
    # Final checkpoint
    ckpt_path = Path(config['output']['checkpoint_dir']) / f'ckpt_final.pt'
    save_checkpoint(
        str(ckpt_path), step, generator, discriminator, opt_G, opt_D,
        ema_G, amp_ctx.scaler, config=config
    )
    print(f"\nTraining complete. Final checkpoint: {ckpt_path}")
    
    # Close loss tracker and generate plot
    loss_tracker.close()
    
    # Load history and create plot
    history = loss_tracker.load_history()
    if len(history['steps']) > 0:
        plot_training_losses(
            config['output']['log_dir'],
            history['steps'],
            history['d_losses'],
            history['g_losses']
        )
    else:
        print("No loss data to plot.")
    
    pbar.close()


if __name__ == '__main__':
    main()

