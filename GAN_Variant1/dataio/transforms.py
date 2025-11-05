"""Transform utilities for photo↔Monet translation."""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from typing import Tuple


class RandomCropResize:
    """Random crop then resize to target size."""
    
    def __init__(self, size: int, scale: Tuple[float, float] = (0.8, 1.0)):
        self.size = size
        self.scale = scale
    
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = np.random.uniform(*self.scale)
        crop_size = int(min(w, h) * scale)
        
        i = np.random.randint(0, h - crop_size + 1)
        j = np.random.randint(0, w - crop_size + 1)
        
        img = TF.crop(img, i, j, crop_size, crop_size)
        img = TF.resize(img, [self.size, self.size], interpolation=Image.BICUBIC)
        return img


def get_train_transforms(image_size: int = 256, use_gray_world: bool = False):
    """Get training transforms for photos/Monet."""
    transforms = [
        RandomCropResize(image_size, scale=(0.85, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # to [-1, 1]
    ]
    return T.Compose(transforms)


def get_eval_transforms(image_size: int = 256):
    """Get evaluation transforms (deterministic)."""
    transforms = [
        T.Resize([image_size, image_size], interpolation=Image.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    return T.Compose(transforms)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize from [-1, 1] to [0, 1]."""
    return tensor * 0.5 + 0.5


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB tensor (B, 3, H, W) in [0, 1] to Lab.
    Simple approximation for palette prior.
    """
    # Clamp to [0, 1]
    rgb = torch.clamp(rgb, 0, 1)
    
    # RGB to XYZ (simplified D65)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    
    # Linearize (simplified)
    r = torch.where(r > 0.04045, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)
    g = torch.where(g > 0.04045, ((g + 0.055) / 1.055) ** 2.4, g / 12.92)
    b = torch.where(b > 0.04045, ((b + 0.055) / 1.055) ** 2.4, b / 12.92)
    
    # RGB to XYZ
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    
    # XYZ to Lab (D65 white point)
    x = x / 0.95047
    y = y / 1.00000
    z = z / 1.08883
    
    epsilon = 0.008856
    kappa = 903.3
    
    fx = torch.where(x > epsilon, torch.pow(x, 1/3), (kappa * x + 16) / 116)
    fy = torch.where(y > epsilon, torch.pow(y, 1/3), (kappa * y + 16) / 116)
    fz = torch.where(z > epsilon, torch.pow(z, 1/3), (kappa * z + 16) / 116)
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_lab = 200 * (fy - fz)
    
    # Stack (B, 3, H, W)
    lab = torch.stack([L, a, b_lab], dim=1)
    return lab


def get_low_freq_stats(lab: torch.Tensor, target_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract low-frequency statistics from Lab image.
    Returns mean and std of downsampled Lab.
    
    Args:
        lab: (B, 3, H, W) Lab tensor
        target_size: downsample to this size
    
    Returns:
        mean: (B, 3) mean per channel
        std: (B, 3) std per channel
    """
    # Downsample to extract low-frequency component
    lab_low = torch.nn.functional.adaptive_avg_pool2d(lab, (target_size, target_size))
    
    # Compute statistics
    mean = lab_low.mean(dim=[2, 3])  # (B, 3)
    std = lab_low.std(dim=[2, 3])    # (B, 3)
    
    return mean, std

