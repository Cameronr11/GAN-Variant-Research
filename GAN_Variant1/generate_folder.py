# GAN_Variant1/generate_folder.py
"""
Dump stylized JPGs from a trained CUT++ generator checkpoint.

Usage (from GAN_Variant1/):
  python generate_folder.py \
    --ckpt GAN_Variant1/checkpoints/ckpt_final.pt \
    --photos /lustre/isaac24/scratch/crader6/GAN_Project/data/photo_jpg \
    --out    /lustre/isaac24/scratch/crader6/GAN_Project/outputs/variant1_final \
    --batch 32 --size 256 --device cuda
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any, Iterable

import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate slightly corrupted inputs


# ------------------------- dynamic import / class resolution -------------------------
def _import_generator_module():
    """Import the generator module whether you run from GAN_Variant1/ or from repo root."""
    try:
        # Running from GAN_Variant1/
        import importlib
        return importlib.import_module("models.generator_resnet_attn")
    except Exception:
        try:
            # Running from repo root with PYTHONPATH pointing at project root
            import importlib
            return importlib.import_module("GAN_Variant1.models.generator_resnet_attn")
        except Exception as e:
            raise ImportError(
                "Could not import generator module. Run from GAN_Variant1/ OR set PYTHONPATH to repo root.\n"
                "Tried 'models.generator_resnet_attn' and 'GAN_Variant1.models.generator_resnet_attn'."
            ) from e


def _resolve_generator_class(mod):
    """
    Return the generator class object from the module.
    Tries common names first, else auto-detects any nn.Module subclass with 'Generator' in the name.
    """
    import inspect

    # Common names we’ve seen in the wild
    candidate_names = [
        "ResnetGenerator", "ResNetGenerator",
        "GeneratorResnetAttn", "GeneratorResNetAttn",
        "GeneratorResNet", "CUTGenerator",
        "Generator",
    ]

    for name in candidate_names:
        if hasattr(mod, name):
            cls = getattr(mod, name)
            if isinstance(cls, type) and issubclass(cls, nn.Module):
                return cls

    # Fallback: pick any class that is an nn.Module and contains 'Generator' in its name
    candidates = []
    for name, obj in vars(mod).items():
        if isinstance(obj, type) and issubclass(obj, nn.Module) and "generator" in name.lower():
            candidates.append((name, obj))

    if len(candidates) == 1:
        return candidates[0][1]
    elif len(candidates) > 1:
        # Prefer the shortest / simplest name
        candidates.sort(key=lambda x: len(x[0]))
        return candidates[0][1]

    # Last resort: any nn.Module class in the file
    for name, obj in vars(mod).items():
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            return obj

    raise ImportError("Could not find a generator class in generator_resnet_attn.py")


def _instantiate_generator(GenClass, device="cuda"):
    """
    Instantiate the generator with only the kwargs the constructor accepts.
    We try to match the training config (ResNet-9 @256 + attention + AdaIN).
    """
    import inspect

    desired_kwargs = dict(
        input_nc=3,
        output_nc=3,
        ngf=64,
        n_blocks=9,
        use_attention=True,
        adain_gates=True,
    )

    sig = inspect.signature(GenClass)
    # keep only supported kwargs
    filtered = {k: v for k, v in desired_kwargs.items() if k in sig.parameters}

    # Some implementations use 'n_resblocks' instead of 'n_blocks'
    if "n_blocks" not in filtered and "n_resblocks" in sig.parameters:
        filtered["n_resblocks"] = desired_kwargs["n_blocks"]

    G = GenClass(**filtered).to(device)
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return G


# ------------------------------- helpers --------------------------------------
EXPECT_KEYS = (
    "ema_state_dict",
    "G_ema",
    "G_state_dict",
    "state_dict",
)

def _pick_state_dict(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Choose the most likely generator state_dict from a checkpoint dict."""
    # Priority 1: Check for EMA generator (preferred for inference)
    if "ema_G" in ckpt and isinstance(ckpt["ema_G"], dict):
        if "shadow" in ckpt["ema_G"]:
            shadow = ckpt["ema_G"]["shadow"]
            if isinstance(shadow, dict):
                print("[INFO] Loading EMA generator weights from ckpt['ema_G']['shadow']")
                return shadow
    
    # Priority 2: Check for base generator
    if "generator" in ckpt and isinstance(ckpt["generator"], dict):
        print("[INFO] Loading generator weights from ckpt['generator']")
        return ckpt["generator"]
    
    # Priority 3: Try legacy key names
    for k in EXPECT_KEYS:
        if k in ckpt and isinstance(ckpt[k], dict):
            print(f"[INFO] Loading generator weights from ckpt['{k}']")
            return ckpt[k]
    
    # If the file itself looks like a state dict (param-tensor mapping), return it
    looks_like_state = all(isinstance(v, torch.Tensor) for v in ckpt.values()) if ckpt else False
    if looks_like_state:
        print("[INFO] Checkpoint appears to be a raw state dict")
        return ckpt
    
    # As a last resort, scan nested dicts
    for v in ckpt.values():
        if isinstance(v, dict):
            sub = v
            looks_like_state = all(isinstance(x, torch.Tensor) for x in sub.values()) if sub else False
            if looks_like_state:
                print("[INFO] Found state dict in nested structure")
                return sub
    
    raise KeyError(
        f"Could not locate a generator state_dict in checkpoint. "
        f"Tried keys: ['ema_G']['shadow'], 'generator', {EXPECT_KEYS} and shallow nested dicts. "
        f"Checkpoint has keys: {list(ckpt.keys())[:10]}..."
    )


def _list_images(root: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def _build_preproc(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),                    # [0,1]
        transforms.Normalize([0.5]*3, [0.5]*3),   # -> [-1,1]
    ])


def _to_uint8(y: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1] -> uint8
    return (y.clamp(-1, 1).mul(0.5).add(0.5).mul(255).round().byte().cpu())


# ------------------------------- core -----------------------------------------
def load_generator(ckpt_path: str, device: str = "cuda") -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint {ckpt_path} is not a dict; got {type(ckpt)}")

    state = _pick_state_dict(ckpt)

    mod = _import_generator_module()
    GenClass = _resolve_generator_class(mod)
    G = _instantiate_generator(GenClass, device=device)

    missing, unexpected = G.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading G ({len(missing)}): e.g., {missing[:8]}")
    if unexpected:
        print(f"[WARN] Unexpected keys in checkpoint for G ({len(unexpected)}): e.g., {unexpected[:8]}")
    return G


@torch.inference_mode()
def stylize_folder(
    G: torch.nn.Module,
    src_dir: str,
    out_dir: str,
    device: str = "cuda",
    img_size: int = 256,
    batch: int = 16,
    limit: Optional[int] = None,
) -> None:
    src_root = Path(src_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    paths = list(_list_images(src_root))
    if limit is not None:
        paths = paths[:limit]
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found under: {src_dir}")

    tfm = _build_preproc(img_size)

    def _load_and_tensorize(ps: Iterable[Path]):
        imgs = []
        for p in ps:
            img = Image.open(p).convert("RGB")
            imgs.append(tfm(img))
        return torch.stack(imgs, 0)  # [B,3,H,W]

    pbar = tqdm(range(0, len(paths), batch), desc="Stylizing", ncols=100)
    for i in pbar:
        chunk = paths[i : i + batch]
        x = _load_and_tensorize(chunk).to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
            y = G(x)  # [-1,1]

        y_u8 = _to_uint8(y)  # [B,3,H,W] uint8 on CPU

        for in_path, y_img in zip(chunk, y_u8):
            pil = transforms.ToPILImage()(y_img)
            rel = in_path.relative_to(src_root)
            save_path = out_root / rel.with_suffix(".jpg")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            pil.save(save_path, format="JPEG", quality=95, subsampling=0, optimize=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stylize a folder of photos with a trained CUT++ generator.")
    ap.add_argument("--ckpt", required=True, help="Path to ckpt_final.pt or ckpt_stepXXXX.pt")
    ap.add_argument("--photos", required=True, help="Path to source photos folder (e.g., Kaggle photo_jpg)")
    ap.add_argument("--out", required=True, help="Output folder for generated JPGs")
    ap.add_argument("--batch", type=int, default=16, help="Batch size for inference")
    ap.add_argument("--size", type=int, default=256, help="Output resolution (and input resize)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    ap.add_argument("--limit", type=int, default=None, help="Optionally limit number of images for quick tests")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    torch.backends.cudnn.benchmark = True if args.device == "cuda" else False

    print(f"Loading generator from: {args.ckpt}")
    G = load_generator(args.ckpt, device=args.device)
    n_params = sum(p.numel() for p in G.parameters())
    print(f"Generator parameters: {n_params:,}")

    print(f"Stylizing from '{args.photos}' -> '{args.out}' "
          f"(size={args.size}, batch={args.batch}, device={args.device})")

    stylize_folder(
        G,
        src_dir=args.photos,
        out_dir=args.out,
        device=args.device,
        img_size=args.size,
        batch=args.batch,
        limit=args.limit,
    )
    print("Done.")


if __name__ == "__main__":
    main()
