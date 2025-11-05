"""
Dataset classes for loading real and fake images with appropriate transforms.
Outputs uint8 tensors [0, 255] for TorchMetrics compatibility.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class ImageFolderDataset(Dataset):
    """
    Dataset for loading images from a folder.
    Returns uint8 tensors [0, 255] resized to target size.
    """
    
    def __init__(
        self,
        image_paths: List[Path],
        img_size: int = 299,
        convert_rgb: bool = True
    ):
        """
        Args:
            image_paths: List of image file paths
            img_size: Target size for resizing (square)
            convert_rgb: If True, convert all images to RGB
        """
        self.image_paths = image_paths
        self.img_size = img_size
        self.convert_rgb = convert_rgb
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and preprocess a single image.
        
        Returns:
            Tensor of shape [3, H, W] with dtype uint8, values in [0, 255]
        """
        img_path = self.image_paths[idx]
        
        try:
            # Load image
            img = Image.open(img_path)
            
            # Convert to RGB if needed
            if self.convert_rgb and img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to target size (bilinear interpolation)
            # TF.resize expects PIL Image or Tensor
            img = TF.resize(img, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BILINEAR)
            
            # Convert to tensor: [C, H, W] in [0, 1] float
            img_tensor = TF.to_tensor(img)
            
            # Convert to uint8 [0, 255] as expected by TorchMetrics with normalize=False
            img_tensor = (img_tensor * 255).to(torch.uint8)
            
            return img_tensor
            
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")


class TFDSMonetDataset(Dataset):
    """
    Placeholder for TFDS-backed Monet dataset.
    Raises informative error if TFDS is not available.
    """
    
    def __init__(self, tfds_name: str, split: str, img_size: int = 299):
        """
        Args:
            tfds_name: TFDS dataset name (e.g., 'cycle_gan/monet2photo')
            split: Dataset split (e.g., 'monet')
            img_size: Target size for resizing
        """
        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError(
                "TensorFlow Datasets (TFDS) is not installed. "
                "To use TFDS mode, install it with: pip install tensorflow-datasets tensorflow"
            )
        
        self.tfds_name = tfds_name
        self.split = split
        self.img_size = img_size
        
        # Load dataset
        try:
            ds = tfds.load(tfds_name, split=split, as_supervised=False)
            # Convert to list for indexing
            self.data = list(ds)
        except Exception as e:
            raise RuntimeError(f"Failed to load TFDS dataset {tfds_name}/{split}: {e}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and preprocess a single TFDS image.
        
        Returns:
            Tensor of shape [3, H, W] with dtype uint8, values in [0, 255]
        """
        item = self.data[idx]
        
        # TFDS typically stores images as 'image' key
        img_array = item['image'].numpy() if hasattr(item['image'], 'numpy') else item['image']
        
        # Convert numpy array to PIL Image
        img = Image.fromarray(img_array)
        
        # Ensure RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = TF.resize(img, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BILINEAR)
        
        # Convert to tensor and scale to uint8 [0, 255]
        img_tensor = TF.to_tensor(img)
        img_tensor = (img_tensor * 255).to(torch.uint8)
        
        return img_tensor


def create_dataloader(
    image_paths: List[Path],
    batch_size: int = 64,
    num_workers: int = 8,
    pin_memory: bool = True,
    img_size: int = 299,
    shuffle: bool = False
) -> DataLoader:
    """
    Create a DataLoader for a list of image paths.
    
    Args:
        image_paths: List of image file paths
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        img_size: Target image size
        shuffle: Whether to shuffle the dataset
        
    Returns:
        DataLoader instance
    """
    dataset = ImageFolderDataset(
        image_paths=image_paths,
        img_size=img_size,
        convert_rgb=True
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return loader


def create_tfds_dataloader(
    tfds_name: str,
    split: str,
    batch_size: int = 64,
    num_workers: int = 8,
    pin_memory: bool = True,
    img_size: int = 299
) -> DataLoader:
    """
    Create a DataLoader for TFDS dataset.
    
    Args:
        tfds_name: TFDS dataset name
        split: Dataset split
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        img_size: Target image size
        
    Returns:
        DataLoader instance
    """
    dataset = TFDSMonetDataset(
        tfds_name=tfds_name,
        split=split,
        img_size=img_size
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return loader

