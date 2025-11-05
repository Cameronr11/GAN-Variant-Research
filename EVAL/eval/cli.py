"""
Command-line interface for MiFID evaluation.
"""

import sys
from pathlib import Path
from typing import Dict
import yaml
import click
import torch

# Import evaluation modules
from eval.utils import (
    enumerate_images,
    compute_image_list_hash,
    validate_image_counts,
    check_dataset_overlap,
    pretty_print_validation,
    timer
)
from eval.datasets import create_dataloader
from eval.mifid import compute_full_evaluation
from eval.report import (
    create_report,
    save_report,
    save_text_summary,
    save_worst_cases_csv,
    print_quick_summary
)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


@click.command()
@click.option('--config', type=str, default=None, help='Path to YAML config file')
@click.option('--fake', type=str, required=True, help='Path to fake images folder')
@click.option('--real', type=str, default=None, help='Path to real images folder (for folder mode)')
@click.option('--out', type=str, default=None, help='Output JSON report path')
@click.option('--batch', type=int, default=None, help='Batch size (overrides config)')
@click.option('--workers', type=int, default=None, help='Number of workers (overrides config)')
@click.option('--img-size', type=int, default=None, help='Image size (overrides config)')
@click.option('--device', type=str, default=None, help='Device: cuda or cpu')
@click.option('--cosine-eps', type=float, default=None, help='Cosine distance epsilon for MiFID')
@click.option('--no-cache', is_flag=True, help='Disable caching of real features')
def main(config, fake, real, out, batch, workers, img_size, device, cosine_eps, no_cache):
    """
    Kaggle MiFID Evaluator
    
    Evaluate generated images against real Monet set using MiFID metric.
    """
    print("\n" + "="*70)
    print("KAGGLE MiFID EVALUATOR")
    print("="*70 + "\n")
    
    # Load config if provided
    if config:
        cfg = load_config(config)
        print(f"Loaded config: {config}")
    else:
        cfg = {
            'name': 'default_run',
            'real': {'mode': 'folder'},
            'io': {},
            'metric': {},
            'cache': {'dir': './cache'},
            'report': {}
        }
        print("Using default configuration")
    
    # Override with CLI arguments
    if fake:
        cfg['fake'] = {'path': fake, 'recursive': True}
    if real:
        cfg['real']['path'] = real
    if batch:
        cfg['io']['batch_size'] = batch
    if workers:
        cfg['io']['num_workers'] = workers
    if img_size:
        cfg['metric']['img_size'] = img_size
    if out:
        cfg['report']['out_json'] = out
    if cosine_eps is not None:
        cfg['metric']['cosine_eps'] = cosine_eps
    
    # Set defaults
    cfg.setdefault('io', {})
    cfg['io'].setdefault('batch_size', 64)
    cfg['io'].setdefault('num_workers', 8)
    cfg['io'].setdefault('pin_memory', True)
    
    cfg.setdefault('metric', {})
    cfg['metric'].setdefault('img_size', 299)
    cfg['metric'].setdefault('cosine_eps', 0.1)
    
    cfg.setdefault('cache', {})
    cfg['cache'].setdefault('dir', './cache')
    
    cfg.setdefault('report', {})
    cfg['report'].setdefault('out_json', './cache/reports/report.json')
    
    # Determine device
    if device:
        device_str = device
    else:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device_str}")
    if device_str == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Validate required paths
    if 'fake' not in cfg or 'path' not in cfg['fake']:
        print("Error: --fake path is required")
        sys.exit(1)
    
    real_mode = cfg['real'].get('mode', 'folder')
    if real_mode == 'folder' and 'path' not in cfg['real']:
        print("Error: --real path is required for folder mode")
        sys.exit(1)
    
    if real_mode == 'tfds':
        print("Error: TFDS mode is not yet implemented in this CLI")
        sys.exit(1)
    
    # Step 1: Enumerate images
    print("="*70)
    print("STEP 1: ENUMERATE IMAGES")
    print("="*70)
    
    with timer("Enumerating fake images"):
        fake_path = Path(cfg['fake']['path'])
        fake_images = enumerate_images(
            fake_path,
            recursive=cfg['fake'].get('recursive', True)
        )
        print(f"Found {len(fake_images):,} fake images in {fake_path}")
    
    with timer("Enumerating real images"):
        real_path = Path(cfg['real']['path'])
        real_images = enumerate_images(
            real_path,
            recursive=cfg['real'].get('recursive', True)
        )
        print(f"Found {len(real_images):,} real images in {real_path}")
    
    # Step 2: Validate counts
    print("\n" + "="*70)
    print("STEP 2: VALIDATE DATASETS")
    print("="*70)
    
    validation = validate_image_counts(fake_images, real_images)
    pretty_print_validation(validation)
    
    # Check for overlap
    overlap = check_dataset_overlap(fake_images, real_images)
    if overlap['has_overlap']:
        print(f"⚠ WARNING: Found {overlap['overlap_count']} overlapping filenames!")
        print(f"  Examples: {overlap['overlap_examples'][:5]}")
    else:
        print("✓ No filename overlap between fake and real sets")
    print()
    
    # Step 3: Compute hashes
    print("="*70)
    print("STEP 3: COMPUTE DATASET HASHES")
    print("="*70)
    
    fake_hash = compute_image_list_hash(fake_images, fake_path)
    real_hash = compute_image_list_hash(real_images, real_path)
    
    print(f"Fake dataset hash: {fake_hash}")
    print(f"Real dataset hash: {real_hash}")
    print()
    
    # Step 4: Create dataloaders
    print("="*70)
    print("STEP 4: CREATE DATALOADERS")
    print("="*70)
    
    batch_size = cfg['io']['batch_size']
    num_workers = cfg['io']['num_workers']
    pin_memory = cfg['io']['pin_memory']
    img_size = cfg['metric']['img_size']
    
    print(f"Batch size: {batch_size}")
    print(f"Workers: {num_workers}")
    print(f"Image size: {img_size}x{img_size}")
    print()
    
    with timer("Creating dataloaders"):
        fake_loader = create_dataloader(
            image_paths=fake_images,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            img_size=img_size,
            shuffle=False
        )
        
        real_loader = create_dataloader(
            image_paths=real_images,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            img_size=img_size,
            shuffle=False
        )
    
    # Step 5: Compute MiFID and FID
    print("\n" + "="*70)
    print("STEP 5: COMPUTE MiFID, FID, AND MEMORIZATION METRICS")
    print("="*70)
    print()
    
    with timer("Full evaluation", verbose=True):
        scores = compute_full_evaluation(
            real_loader=real_loader,
            fake_loader=fake_loader,
            fake_paths=fake_images,
            real_paths=real_images,
            device=device_str,
            cosine_eps=cfg['metric']['cosine_eps']
        )
    
    # Print quick summary
    print_quick_summary(scores, validation)
    
    # Step 6: Generate report
    print("="*70)
    print("STEP 6: GENERATE REPORT")
    print("="*70)
    print()
    
    # Prepare run config for report
    run_config = {
        'name': cfg.get('name', 'unnamed_run'),
        'fake_dir': str(fake_path),
        'real_mode': real_mode,
        'real_dir': str(real_path),
        'img_size': img_size,
        'batch_size': batch_size,
        'num_workers': num_workers
    }
    
    # Prepare hashes
    hashes = {
        'fake_list_sha1': fake_hash,
        'real_list_sha1': real_hash,
        'real_cache_key': f"monet_jpg@sha1:{real_hash[:16]}"
    }
    
    # Create report
    report = create_report(
        scores=scores,
        run_config=run_config,
        hashes=hashes,
        validation=validation,
        worst_cases=scores.get('worst_memorization_cases', [])
    )
    
    # Save JSON report
    json_path = cfg['report']['out_json']
    save_report(report, json_path, verbose=True)
    
    # Save text summary
    timestamp = report['run']['timestamp_utc'].replace(':', '').replace('-', '').replace('Z', '')[:15]
    run_name = cfg.get('name', 'run').replace(' ', '_')
    text_path = Path(cfg['cache']['dir']) / 'logs' / f"{timestamp}_{run_name}.txt"
    save_text_summary(report, text_path, verbose=True)
    
    # Save worst cases CSV
    if 'worst_memorization_cases' in scores:
        csv_path = Path(json_path).parent / f"{Path(json_path).stem}_worst_cases.csv"
        save_worst_cases_csv(scores['worst_memorization_cases'], csv_path)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nFinal MiFID Score: {scores['mifid']:.4f}")
    print(f"Reports saved to: {Path(json_path).parent}")
    print()


if __name__ == '__main__':
    main()

