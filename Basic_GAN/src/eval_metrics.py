"""
Core evaluation metrics for CycleGAN: FID, KID, and memorization proxy
"""
import os
import warnings
from typing import Optional, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from PIL import Image
from tqdm import tqdm

from src.utils_eval import list_images, sample_paths


def compute_fid(real_dir: str, gen_dir: str, num_workers: int = 0) -> float:
    """
    Compute Fréchet Inception Distance between real and generated images.
    
    Args:
        real_dir: Directory containing real images
        gen_dir: Directory containing generated images
        num_workers: Number of workers for data loading
        
    Returns:
        FID score (lower is better)
    """
    try:
        from cleanfid import fid
    except ImportError:
        raise ImportError(
            "clean-fid is not installed. Please install it with:\n"
            "  pip install clean-fid"
        )
    
    # Use clean-fid's compute_fid function
    fid_score = fid.compute_fid(
        real_dir, 
        gen_dir, 
        mode="clean",
        num_workers=num_workers,
        verbose=False
    )
    
    return fid_score


def compute_kid(real_dir: str, gen_dir: str, num_workers: int = 0) -> float:
    """
    Compute Kernel Inception Distance between real and generated images.
    
    Args:
        real_dir: Directory containing real images
        gen_dir: Directory containing generated images
        num_workers: Number of workers for data loading
        
    Returns:
        KID score (lower is better)
    """
    try:
        from cleanfid import fid
    except ImportError:
        raise ImportError(
            "clean-fid is not installed. Please install it with:\n"
            "  pip install clean-fid"
        )
    
    # Use clean-fid's compute_kid function
    kid_score = fid.compute_kid(
        real_dir, 
        gen_dir, 
        mode="clean",
        num_workers=num_workers,
        verbose=False
    )
    
    return kid_score


def compute_memorization_proxy(
    real_dir: str,
    gen_dir: str,
    max_real: Optional[int] = None,
    max_gen: Optional[int] = None,
    batch_size: int = 64,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed: int = 42
) -> float:
    """
    Compute memorization proxy as average minimum cosine distance.
    
    For each generated image, compute its minimum cosine distance to any real image
    in InceptionV3 feature space (pool3, 2048-d). Return the average.
    
    Args:
        real_dir: Directory containing real Monet images
        gen_dir: Directory containing generated images
        max_real: Maximum number of real images to use (None = all)
        max_gen: Maximum number of generated images to use (None = all)
        batch_size: Batch size for embedding extraction
        device: Device to use ('cuda' or 'cpu')
        seed: Random seed for sampling
        
    Returns:
        Average minimum cosine distance (higher means less memorization)
    """
    # List and sample images
    real_paths = list_images(real_dir)
    gen_paths = list_images(gen_dir)
    
    if len(real_paths) == 0:
        raise ValueError(f"No images found in real_dir: {real_dir}")
    if len(gen_paths) == 0:
        raise ValueError(f"No images found in gen_dir: {gen_dir}")
    
    real_paths = sample_paths(real_paths, max_real, seed)
    gen_paths = sample_paths(gen_paths, max_gen, seed)
    
    print(f"Computing memorization proxy: {len(real_paths)} real, {len(gen_paths)} generated")
    
    # Load InceptionV3 model
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    # Remove the final classification layer to get pool3 features
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Extract embeddings
    def extract_embeddings(image_paths: List[str], desc: str) -> torch.Tensor:
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc=desc):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []
                
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert('RGB')
                        img_tensor = transform(img)
                        batch_images.append(img_tensor)
                    except Exception as e:
                        warnings.warn(f"Failed to load {path}: {e}")
                        continue
                
                if len(batch_images) == 0:
                    continue
                
                batch_tensor = torch.stack(batch_images).to(device)
                batch_embeddings = model(batch_tensor)
                embeddings.append(batch_embeddings.cpu())
        
        if len(embeddings) == 0:
            raise ValueError(f"No valid embeddings extracted from {desc}")
        
        return torch.cat(embeddings, dim=0)
    
    # Extract embeddings for real and generated images
    real_embeddings = extract_embeddings(real_paths, "Extracting real embeddings")
    gen_embeddings = extract_embeddings(gen_paths, "Extracting gen embeddings")
    
    # Normalize embeddings for cosine similarity
    real_embeddings = F.normalize(real_embeddings, dim=1)
    gen_embeddings = F.normalize(gen_embeddings, dim=1)
    
    # Compute minimum cosine distances
    # Process in chunks to avoid OOM
    min_distances = []
    
    real_embeddings = real_embeddings.to(device)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(gen_embeddings), batch_size), desc="Computing min distances"):
            gen_chunk = gen_embeddings[i:i + batch_size].to(device)
            
            # Compute cosine similarity: gen_chunk @ real_embeddings.T
            # Shape: [chunk_size, num_real]
            cos_sim = gen_chunk @ real_embeddings.T
            
            # Convert to distance: 1 - cos_sim
            cos_dist = 1.0 - cos_sim
            
            # Get minimum distance for each generated image
            min_dist_chunk = cos_dist.min(dim=1).values
            min_distances.append(min_dist_chunk.cpu())
    
    # Concatenate and compute average
    all_min_distances = torch.cat(min_distances, dim=0)
    avg_min_distance = all_min_distances.mean().item()
    
    return avg_min_distance

