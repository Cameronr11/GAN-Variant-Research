"""
MiFID and FID computation using TorchMetrics.
Includes cosine distance analysis for memorization metrics.
"""

from typing import Dict, Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance


def compute_mifid_and_fid(
    real_loader: DataLoader,
    fake_loader: DataLoader,
    device: str = 'cuda',
    feature_dim: int = 2048,
    cosine_eps: float = 0.1,
    return_features: bool = True
) -> Dict:
    """
    Compute MiFID and FID scores using TorchMetrics.
    
    Args:
        real_loader: DataLoader for real images (uint8 [0,255])
        fake_loader: DataLoader for fake images (uint8 [0,255])
        device: Device for computation
        feature_dim: Feature dimension (2048 for pool3)
        cosine_eps: Epsilon for cosine distance in MiFID
        return_features: Whether to return raw features for additional analysis
        
    Returns:
        Dictionary with 'mifid', 'fid', and optionally 'real_features', 'fake_features'
    """
    # Initialize metrics
    mifid_metric = MemorizationInformedFrechetInceptionDistance(
        feature=feature_dim,
        normalize=False,  # We're feeding uint8 directly
        cosine_distance_eps=cosine_eps
    ).to(device)
    
    fid_metric = FrechetInceptionDistance(
        feature=feature_dim,
        normalize=False
    ).to(device)
    
    # Lists to store features if requested
    real_feats_list = [] if return_features else None
    fake_feats_list = [] if return_features else None
    
    # Process real images
    print("Processing real images...")
    with torch.no_grad():
        for batch in tqdm(real_loader, desc="Real images"):
            batch = batch.to(device)
            
            # Update metrics
            mifid_metric.update(batch, real=True)
            fid_metric.update(batch, real=True)
            
            # Extract features if needed
            if return_features:
                # Access inception model from metric
                inception = mifid_metric.inception
                inception.eval()
                feats = inception(batch)
                real_feats_list.append(feats.cpu().numpy())
    
    # Process fake images
    print("Processing fake images...")
    with torch.no_grad():
        for batch in tqdm(fake_loader, desc="Fake images"):
            batch = batch.to(device)
            
            # Update metrics
            mifid_metric.update(batch, real=False)
            fid_metric.update(batch, real=False)
            
            # Extract features if needed
            if return_features:
                inception = mifid_metric.inception
                inception.eval()
                feats = inception(batch)
                fake_feats_list.append(feats.cpu().numpy())
    
    # Compute final scores
    print("Computing MiFID...")
    mifid_score = mifid_metric.compute().item()
    
    print("Computing FID...")
    fid_score = fid_metric.compute().item()
    
    result = {
        'mifid': mifid_score,
        'fid': fid_score
    }
    
    # Add features if requested
    if return_features:
        result['real_features'] = np.vstack(real_feats_list)
        result['fake_features'] = np.vstack(fake_feats_list)
    
    return result


def compute_cosine_distances_batched(
    fake_features: np.ndarray,
    real_features: np.ndarray,
    batch_size: int = 1000
) -> np.ndarray:
    """
    Compute minimum cosine distances from each fake to all real features.
    Uses batched computation for memory efficiency.
    
    Args:
        fake_features: Fake features [N_fake, D]
        real_features: Real features [N_real, D]
        batch_size: Batch size for computation
        
    Returns:
        Array of minimum cosine distances [N_fake]
    """
    # Normalize features for cosine similarity
    fake_norm = fake_features / (np.linalg.norm(fake_features, axis=1, keepdims=True) + 1e-8)
    real_norm = real_features / (np.linalg.norm(real_features, axis=1, keepdims=True) + 1e-8)
    
    n_fake = len(fake_norm)
    min_distances = np.zeros(n_fake)
    
    print("Computing cosine distances...")
    for i in tqdm(range(0, n_fake, batch_size), desc="Cosine distances"):
        end_idx = min(i + batch_size, n_fake)
        batch_fake = fake_norm[i:end_idx]
        
        # Compute cosine similarity: [batch, real]
        cosine_sim = batch_fake @ real_norm.T
        
        # Cosine distance = 1 - cosine_similarity
        cosine_dist = 1.0 - cosine_sim
        
        # Find minimum distance for each fake
        min_distances[i:end_idx] = np.min(cosine_dist, axis=1)
    
    return min_distances


def compute_cosine_distance_statistics(min_distances: np.ndarray) -> Dict:
    """
    Compute statistics and histogram for cosine distances.
    
    Args:
        min_distances: Array of minimum cosine distances [N]
        
    Returns:
        Dictionary with median, percentiles, and histogram
    """
    # Compute percentiles
    median = float(np.median(min_distances))
    p10 = float(np.percentile(min_distances, 10))
    p90 = float(np.percentile(min_distances, 90))
    mean = float(np.mean(min_distances))
    std = float(np.std(min_distances))
    
    # Compute histogram (10 bins)
    hist_counts, hist_bins = np.histogram(min_distances, bins=10)
    
    return {
        'median': median,
        'mean': mean,
        'std': std,
        'p10': p10,
        'p90': p90,
        'hist_bins': hist_bins.tolist(),
        'hist_counts': hist_counts.tolist()
    }


def find_worst_memorization_cases(
    fake_paths: List,
    min_distances: np.ndarray,
    real_paths: List,
    real_features: np.ndarray,
    fake_features: np.ndarray,
    top_k: int = 16
) -> List[Dict]:
    """
    Find fake images with smallest cosine distances (worst memorization cases).
    
    Args:
        fake_paths: List of fake image paths
        min_distances: Minimum cosine distance for each fake
        real_paths: List of real image paths
        real_features: Real features [N_real, D]
        fake_features: Fake features [N_fake, D]
        top_k: Number of worst cases to return
        
    Returns:
        List of dictionaries with fake_path, distance, real_path (nearest neighbor)
    """
    # Find indices of smallest distances
    worst_indices = np.argsort(min_distances)[:top_k]
    
    # Normalize features for cosine similarity
    fake_norm = fake_features / (np.linalg.norm(fake_features, axis=1, keepdims=True) + 1e-8)
    real_norm = real_features / (np.linalg.norm(real_features, axis=1, keepdims=True) + 1e-8)
    
    worst_cases = []
    for idx in worst_indices:
        fake_path = fake_paths[idx]
        distance = min_distances[idx]
        
        # Find nearest real image
        fake_feat = fake_norm[idx]
        cosine_sim = fake_feat @ real_norm.T
        nearest_real_idx = np.argmax(cosine_sim)
        real_path = real_paths[nearest_real_idx]
        
        worst_cases.append({
            'fake_path': str(fake_path),
            'distance': float(distance),
            'nearest_real_path': str(real_path),
            'cosine_similarity': float(1.0 - distance)
        })
    
    return worst_cases


def compute_full_evaluation(
    real_loader: DataLoader,
    fake_loader: DataLoader,
    fake_paths: List,
    real_paths: List,
    device: str = 'cuda',
    cosine_eps: float = 0.1
) -> Dict:
    """
    Compute complete evaluation: MiFID, FID, and cosine distance analysis.
    
    Args:
        real_loader: DataLoader for real images
        fake_loader: DataLoader for fake images
        fake_paths: List of fake image paths (for memorization analysis)
        real_paths: List of real image paths
        device: Device for computation
        cosine_eps: Epsilon for MiFID cosine distance
        
    Returns:
        Dictionary with all metrics and analysis
    """
    # Compute MiFID and FID with features
    scores = compute_mifid_and_fid(
        real_loader=real_loader,
        fake_loader=fake_loader,
        device=device,
        feature_dim=2048,
        cosine_eps=cosine_eps,
        return_features=True
    )
    
    mifid = scores['mifid']
    fid = scores['fid']
    real_features = scores['real_features']
    fake_features = scores['fake_features']
    
    # Compute cosine distance statistics
    min_distances = compute_cosine_distances_batched(
        fake_features=fake_features,
        real_features=real_features,
        batch_size=1000
    )
    
    cosine_stats = compute_cosine_distance_statistics(min_distances)
    
    # Find worst memorization cases
    worst_cases = find_worst_memorization_cases(
        fake_paths=fake_paths,
        min_distances=min_distances,
        real_paths=real_paths,
        real_features=real_features,
        fake_features=fake_features,
        top_k=16
    )
    
    return {
        'mifid': mifid,
        'fid': fid,
        'cosine_min_distance': cosine_stats,
        'worst_memorization_cases': worst_cases
    }

