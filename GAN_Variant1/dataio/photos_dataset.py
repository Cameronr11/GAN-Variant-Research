"""Photos dataset for training."""
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable


class PhotosDataset(Dataset):
    """Dataset for photo images (source domain)."""
    
    def __init__(
        self,
        photos_dir: str,
        transform: Optional[Callable] = None,
        extensions: tuple = ('.jpg', '.jpeg', '.png')
    ):
        self.photos_dir = Path(photos_dir)
        self.transform = transform
        
        # Collect all image paths
        self.image_paths = []
        if self.photos_dir.exists():
            for ext in extensions:
                self.image_paths.extend(sorted(self.photos_dir.glob(f'*{ext}')))
                self.image_paths.extend(sorted(self.photos_dir.glob(f'*{ext.upper()}')))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {photos_dir}")
        
        print(f"Found {len(self.image_paths)} photo images")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img


class PhotosTFRecordDataset(Dataset):
    """
    Optional TFRecord dataset loader for photos.
    Falls back to JPG dataset if TFRecord reading is unavailable.
    """
    
    def __init__(
        self,
        tfrec_dir: str,
        transform: Optional[Callable] = None,
        fallback_jpg_dir: Optional[str] = None
    ):
        try:
            import tensorflow as tf
            self.tf = tf
            self.use_tfrec = True
        except ImportError:
            print("TensorFlow not available, falling back to JPG dataset")
            self.use_tfrec = False
            if fallback_jpg_dir:
                self.jpg_dataset = PhotosDataset(fallback_jpg_dir, transform)
            return
        
        self.tfrec_dir = Path(tfrec_dir)
        self.transform = transform
        
        # Collect TFRecord files
        self.tfrec_files = sorted(self.tfrec_dir.glob('*.tfrec'))
        
        if len(self.tfrec_files) == 0:
            raise ValueError(f"No TFRecord files found in {tfrec_dir}")
        
        print(f"Found {len(self.tfrec_files)} TFRecord files")
        
        # For simplicity, we'll count total records (expensive but one-time)
        self._count_records()
    
    def _count_records(self):
        """Count total number of records across all TFRecord files."""
        self.total_records = 0
        for tfrec_file in self.tfrec_files:
            for _ in self.tf.data.TFRecordDataset(str(tfrec_file)):
                self.total_records += 1
        print(f"Total records: {self.total_records}")
    
    def __len__(self) -> int:
        if not self.use_tfrec:
            return len(self.jpg_dataset)
        return self.total_records
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        if not self.use_tfrec:
            return self.jpg_dataset[idx]
        
        # This is inefficient for random access; consider using JPG for now
        raise NotImplementedError("TFRecord random access not efficiently implemented. Use JPG dataset.")

