"""Monet dataset for training."""
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable


class MonetDataset(Dataset):
    """Dataset for Monet images (target domain)."""
    
    def __init__(
        self,
        monet_dir: str,
        transform: Optional[Callable] = None,
        extensions: tuple = ('.jpg', '.jpeg', '.png')
    ):
        self.monet_dir = Path(monet_dir)
        self.transform = transform
        
        # Collect all image paths
        self.image_paths = []
        if self.monet_dir.exists():
            for ext in extensions:
                self.image_paths.extend(sorted(self.monet_dir.glob(f'*{ext}')))
                self.image_paths.extend(sorted(self.monet_dir.glob(f'*{ext.upper()}')))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {monet_dir}")
        
        print(f"Found {len(self.image_paths)} Monet images")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img
    
    def get_image_path(self, idx: int) -> Path:
        """Return the image path for a given index."""
        return self.image_paths[idx]

