"""
Inception-V3 feature extraction and caching for real image sets.
Uses InceptionV3 pool3 (2048-dimensional) features.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# TorchMetrics provides InceptionV3 feature extractor
from torchmetrics.image.fid import FrechetInceptionDistance


class InceptionFeatureExtractor:
    """
    Wrapper for InceptionV3 feature extraction using TorchMetrics.
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        device: str = 'cuda',
        normalize: bool = False
    ):
        """
        Args:
            feature_dim: Feature dimension (2048 for pool3)
            device: Device to run on ('cuda' or 'cpu')
            normalize: Whether to normalize inputs (False for uint8 inputs)
        """
        self.feature_dim = feature_dim
        self.device = device
        self.normalize = normalize
        
        # Initialize FID metric to access its internal feature extractor
        self.fid_metric = FrechetInceptionDistance(
            feature=feature_dim,
            normalize=normalize
        ).to(device)
    
    def extract_features(self, dataloader: DataLoader, desc: str = "Extracting features") -> np.ndarray:
        """
        Extract features from a dataloader.
        
        Args:
            dataloader: DataLoader yielding uint8 image batches [B, 3, H, W]
            desc: Description for progress bar
            
        Returns:
            Feature array of shape [N, feature_dim]
        """
        features_list = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                # Move to device
                batch = batch.to(self.device)
                
                # Extract features using TorchMetrics' internal inception model
                # We'll use a trick: call update with these images, then extract the features
                # Actually, let's directly access the inception model
                inception_model = self.fid_metric.inception
                inception_model.eval()
                
                # Forward pass through Inception
                feats = inception_model(batch)
                
                # Move to CPU and store
                features_list.append(feats.cpu().numpy())
        
        # Concatenate all features
        features = np.vstack(features_list)
        
        return features
    
    def compute_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance from features.
        
        Args:
            features: Feature array [N, D]
            
        Returns:
            Tuple of (mu, sigma) where:
                mu: mean vector [D]
                sigma: covariance matrix [D, D]
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma


def compute_cache_key(image_paths: list, base_path: Path = None) -> str:
    """
    Compute a stable cache key for a set of images.
    
    Args:
        image_paths: List of image paths
        base_path: Base path for relative path computation
        
    Returns:
        Cache key string (SHA1 hash)
    """
    from eval.utils import compute_image_list_hash
    return compute_image_list_hash(image_paths, base_path)


def load_cached_stats(cache_key: str, cache_dir: Path) -> Optional[Dict]:
    """
    Load cached statistics if available.
    
    Args:
        cache_key: Cache key (hash)
        cache_dir: Cache directory path
        
    Returns:
        Dictionary with 'mu', 'sigma', 'features' (optional), 'n' or None if not cached
    """
    cache_file = cache_dir / f"{cache_key}.npz"
    
    if not cache_file.exists():
        return None
    
    try:
        data = np.load(cache_file, allow_pickle=False)
        result = {
            'mu': data['mu'],
            'sigma': data['sigma'],
            'n': int(data['n'])
        }
        
        # Load features if available
        if 'features' in data:
            result['features'] = data['features']
        
        print(f"✓ Loaded cached stats from {cache_file.name} (n={result['n']})")
        return result
        
    except Exception as e:
        print(f"⚠ Failed to load cache {cache_file.name}: {e}")
        return None


def save_cached_stats(
    cache_key: str,
    cache_dir: Path,
    mu: np.ndarray,
    sigma: np.ndarray,
    features: np.ndarray = None,
    n: int = None
):
    """
    Save statistics to cache.
    
    Args:
        cache_key: Cache key (hash)
        cache_dir: Cache directory path
        mu: Mean vector
        sigma: Covariance matrix
        features: Optional raw features for cosine distance computation
        n: Number of samples
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.npz"
    
    # Prepare data dictionary
    save_dict = {
        'mu': mu,
        'sigma': sigma,
        'n': n if n is not None else len(features) if features is not None else 0
    }
    
    # Optionally include raw features (useful for cosine distance computation)
    if features is not None:
        save_dict['features'] = features
    
    # Save as compressed numpy archive
    np.savez_compressed(cache_file, **save_dict)
    
    print(f"✓ Cached stats to {cache_file.name} (n={save_dict['n']})")


def compute_or_load_real_stats(
    dataloader: DataLoader,
    cache_key: str,
    cache_dir: Path,
    device: str = 'cuda',
    save_features: bool = True
) -> Dict:
    """
    Compute or load real image statistics with caching.
    
    Args:
        dataloader: DataLoader for real images
        cache_key: Cache key for this dataset
        cache_dir: Cache directory
        device: Device for computation
        save_features: Whether to save raw features (needed for cosine distance)
        
    Returns:
        Dictionary with 'mu', 'sigma', 'features', 'n', 'cache_key'
    """
    # Try to load from cache
    cached = load_cached_stats(cache_key, cache_dir)
    
    if cached is not None:
        return {**cached, 'cache_key': cache_key}
    
    # Compute features
    print(f"Computing real image features (cache key: {cache_key[:8]}...)")
    extractor = InceptionFeatureExtractor(feature_dim=2048, device=device, normalize=False)
    features = extractor.extract_features(dataloader, desc="Extracting real features")
    
    # Compute statistics
    mu, sigma = extractor.compute_statistics(features)
    n = len(features)
    
    # Save to cache
    save_cached_stats(
        cache_key=cache_key,
        cache_dir=cache_dir,
        mu=mu,
        sigma=sigma,
        features=features if save_features else None,
        n=n
    )
    
    return {
        'mu': mu,
        'sigma': sigma,
        'features': features,
        'n': n,
        'cache_key': cache_key
    }


def extract_fake_features(
    dataloader: DataLoader,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Extract features from fake images (no caching).
    
    Args:
        dataloader: DataLoader for fake images
        device: Device for computation
        
    Returns:
        Feature array [N, 2048]
    """
    print("Computing fake image features...")
    extractor = InceptionFeatureExtractor(feature_dim=2048, device=device, normalize=False)
    features = extractor.extract_features(dataloader, desc="Extracting fake features")
    
    return features

