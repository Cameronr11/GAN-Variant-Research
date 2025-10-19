"""
Utility functions for evaluation metrics
"""
import os
import json
import csv
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Any, Optional


def list_images(directory: str) -> List[str]:
    """
    List all image files in a directory.
    
    Args:
        directory: Path to directory containing images
        
    Returns:
        List of full paths to image files
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_paths = []
    
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            image_paths.append(str(file_path))
    
    return sorted(image_paths)


def sample_paths(paths: List[str], n: Optional[int], seed: int = 42) -> List[str]:
    """
    Randomly sample n paths from the list.
    
    Args:
        paths: List of file paths
        n: Number of paths to sample (None means all)
        seed: Random seed for reproducibility
        
    Returns:
        Sampled list of paths
    """
    if n is None or n >= len(paths):
        return paths
    
    rng = random.Random(seed)
    return sorted(rng.sample(paths, n))


def write_json(path: str, obj: Dict[str, Any]) -> None:
    """
    Write object to JSON file.
    
    Args:
        path: Output JSON file path
        obj: Dictionary to write
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def append_csv(path: str, dict_row: Dict[str, Any], ordered_cols: List[str]) -> None:
    """
    Append a row to a CSV file, creating it if it doesn't exist.
    
    Args:
        path: CSV file path
        dict_row: Dictionary with column values
        ordered_cols: List of column names in desired order
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = path.exists()
    
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ordered_cols)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(dict_row)


def short_hash(dict_like: Dict[str, Any]) -> str:
    """
    Generate a short hash of a dictionary for tracking configs.
    
    Args:
        dict_like: Dictionary to hash
        
    Returns:
        8-character hex hash
    """
    # Canonicalize the dictionary as sorted JSON
    canonical = json.dumps(dict_like, sort_keys=True)
    
    # Compute MD5 hash
    hash_obj = hashlib.md5(canonical.encode('utf-8'))
    
    # Return first 8 characters
    return hash_obj.hexdigest()[:8]

