"""
Utility functions for image enumeration, hashing, validation, and logging.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any
import time
from contextlib import contextmanager


def enumerate_images(path: str, recursive: bool = True) -> List[Path]:
    """
    Enumerate all image files (jpg, jpeg, png) in a directory.
    
    Args:
        path: Directory path to search
        recursive: If True, search subdirectories recursively
        
    Returns:
        Sorted list of Path objects pointing to image files
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    if not path_obj.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")
    
    # Image extensions (case-insensitive)
    img_extensions = {'.jpg', '.jpeg', '.png'}
    
    images = []
    if recursive:
        for ext in img_extensions:
            images.extend(path_obj.rglob(f'*{ext}'))
            images.extend(path_obj.rglob(f'*{ext.upper()}'))
    else:
        for ext in img_extensions:
            images.extend(path_obj.glob(f'*{ext}'))
            images.extend(path_obj.glob(f'*{ext.upper()}'))
    
    # Remove duplicates and sort
    images = sorted(set(images))
    return images


def compute_image_list_hash(image_paths: List[Path], base_path: Path = None) -> str:
    """
    Compute a stable SHA1 hash from a list of image paths.
    Uses relative paths and file sizes for reproducibility.
    
    Args:
        image_paths: List of image file paths
        base_path: Base path for computing relative paths (optional)
        
    Returns:
        SHA1 hex digest string
    """
    hasher = hashlib.sha1()
    
    # Sort paths for stability
    sorted_paths = sorted(image_paths)
    
    for img_path in sorted_paths:
        # Use relative path if base_path provided
        if base_path:
            try:
                rel_path = img_path.relative_to(base_path)
            except ValueError:
                rel_path = img_path
        else:
            rel_path = img_path
        
        # Get file size
        try:
            file_size = img_path.stat().st_size
        except OSError:
            file_size = 0
        
        # Hash: path + size
        entry = f"{rel_path.as_posix()}:{file_size}\n"
        hasher.update(entry.encode('utf-8'))
    
    return hasher.hexdigest()


def validate_image_counts(fake_images: List[Path], real_images: List[Path]) -> Dict[str, Any]:
    """
    Validate image counts and return a summary dictionary.
    
    Args:
        fake_images: List of fake image paths
        real_images: List of real image paths
        
    Returns:
        Dictionary with validation results and warnings
    """
    num_fake = len(fake_images)
    num_real = len(real_images)
    
    warnings = []
    
    # Check fake count (7k-10k expected)
    if num_fake < 7000:
        warnings.append(f"Fake image count ({num_fake}) is below expected range (7000-10000)")
    elif num_fake > 10000:
        warnings.append(f"Fake image count ({num_fake}) is above expected range (7000-10000)")
    
    # Check real count (≥300 expected)
    if num_real < 300:
        warnings.append(f"Real image count ({num_real}) is below expected minimum (300)")
    
    # Check for empty sets
    if num_fake == 0:
        raise ValueError("No fake images found!")
    if num_real == 0:
        raise ValueError("No real images found!")
    
    # Compute total sizes
    fake_total_bytes = sum(p.stat().st_size for p in fake_images)
    real_total_bytes = sum(p.stat().st_size for p in real_images)
    
    return {
        'num_fake': num_fake,
        'num_real': num_real,
        'fake_total_mb': fake_total_bytes / (1024 * 1024),
        'real_total_mb': real_total_bytes / (1024 * 1024),
        'warnings': warnings,
        'valid': len(warnings) == 0
    }


def check_dataset_overlap(fake_paths: List[Path], real_paths: List[Path]) -> Dict[str, Any]:
    """
    Check if there's overlap between fake and real image sets.
    
    Args:
        fake_paths: List of fake image paths
        real_paths: List of real image paths
        
    Returns:
        Dictionary with overlap information
    """
    # Use filenames for comparison (not full paths)
    fake_names = {p.name for p in fake_paths}
    real_names = {p.name for p in real_paths}
    
    overlap = fake_names & real_names
    
    return {
        'has_overlap': len(overlap) > 0,
        'overlap_count': len(overlap),
        'overlap_examples': sorted(list(overlap))[:10] if overlap else []
    }


def compute_pixel_statistics(images_tensor, name: str = "dataset") -> Dict[str, float]:
    """
    Compute basic pixel statistics for sanity checking.
    
    Args:
        images_tensor: Tensor of images [N, C, H, W] in [0, 255] range
        name: Name for logging
        
    Returns:
        Dictionary with mean and std statistics
    """
    import torch
    
    mean = images_tensor.float().mean().item()
    std = images_tensor.float().std().item()
    min_val = images_tensor.float().min().item()
    max_val = images_tensor.float().max().item()
    
    return {
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val
    }


@contextmanager
def timer(name: str, verbose: bool = True):
    """
    Context manager for timing code blocks.
    
    Args:
        name: Name of the timed operation
        verbose: If True, print timing information
        
    Yields:
        Dictionary that will contain 'elapsed' key after completion
    """
    result = {}
    start = time.time()
    
    if verbose:
        print(f"[Timer] Starting: {name}")
    
    try:
        yield result
    finally:
        elapsed = time.time() - start
        result['elapsed'] = elapsed
        
        if verbose:
            print(f"[Timer] Completed: {name} in {elapsed:.2f}s")


def pretty_print_validation(validation_result: Dict[str, Any]):
    """
    Pretty print validation results.
    
    Args:
        validation_result: Output from validate_image_counts()
    """
    print("\n" + "="*60)
    print("IMAGE COUNT VALIDATION")
    print("="*60)
    print(f"Fake images:  {validation_result['num_fake']:,}")
    print(f"Real images:  {validation_result['num_real']:,}")
    print(f"Fake total:   {validation_result['fake_total_mb']:.1f} MB")
    print(f"Real total:   {validation_result['real_total_mb']:.1f} MB")
    
    if validation_result['warnings']:
        print("\nWARNINGS:")
        for warning in validation_result['warnings']:
            print(f"  ⚠ {warning}")
    else:
        print("\n✓ All counts within expected ranges")
    
    print("="*60 + "\n")


def save_json_report(report: Dict[str, Any], output_path: str):
    """
    Save report as formatted JSON.
    
    Args:
        report: Report dictionary
        output_path: Path to output JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {output_path}")

