# src/data.py
import os, random
from typing import Dict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def _image_tf(load_size: int, crop_size: int, train: bool):
    ops = []
    if train:
        ops += [
            transforms.Resize(load_size, Image.BICUBIC),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        ops += [
            transforms.Resize(crop_size, Image.BICUBIC),
            transforms.CenterCrop(crop_size),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # to [-1, 1]
    ]
    return transforms.Compose(ops)

class ImageFolder(Dataset):
    def __init__(self, root: str, subdir: str, load_size=286, crop_size=256, train=True):
        self.dir = os.path.join(root, subdir)
        self.paths = sorted(
            p for p in (os.path.join(self.dir, f) for f in os.listdir(self.dir))
            if os.path.isfile(p) and p.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        self.tf = _image_tf(load_size, crop_size, train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img)

class UnpairedDataset(Dataset):
    """
    Returns a dict with tensors from domain A (photos) and B (Monet) for unpaired training.
    """
    def __init__(self, root: str, domain_a: str, domain_b: str, load_size=286, crop_size=256):
        self.ds_a = ImageFolder(root, domain_a, load_size, crop_size, train=True)
        self.ds_b = ImageFolder(root, domain_b, load_size, crop_size, train=True)

    def __len__(self):
        # iterate for the larger set; sample B randomly to stay unpaired
        return max(len(self.ds_a), len(self.ds_b))

    def __getitem__(self, idx) -> Dict[str, "torch.Tensor"]:
        a = self.ds_a[idx % len(self.ds_a)]
        b = self.ds_b[random.randint(0, len(self.ds_b) - 1)]
        return {"A": a, "B": b}

def make_dataloader(cfg) -> DataLoader:
    ds = UnpairedDataset(
        cfg["data"]["root"],
        cfg["data"]["domain_a"],
        cfg["data"]["domain_b"],
        cfg["data"]["load_size"],
        cfg["data"]["img_size"],
    )
    return DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
