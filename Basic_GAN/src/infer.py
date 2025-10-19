# src/infer.py
import os, glob, zipfile, yaml
from typing import List
import torch
from PIL import Image
from torchvision import transforms
from .models import ResnetGenerator

@torch.no_grad()
def export_zip(cfg_path: str, ckpt_path: str, input_dir: str, out_zip: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = ResnetGenerator(ngf=cfg["model"]["ngf"], n_blocks=cfg["model"]["n_blocks"]).to(device)
    state = torch.load(ckpt_path, map_location=device)

    # support both our saved format and a raw state_dict
    if "G_A2B" in state:
        G.load_state_dict(state["G_A2B"])
    else:
        G.load_state_dict(state)

    G.eval()

    # match training norm: [-1,1] ↔ [0,1]
    img_size = int(cfg["data"]["img_size"])
    tf_in = transforms.Compose([
        transforms.Resize(img_size, Image.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    to_pil = transforms.ToPILImage()

    def denorm(x: torch.Tensor) -> torch.Tensor:
        return (x.clamp_(-1, 1) * 0.5 + 0.5)

    os.makedirs(os.path.dirname(out_zip), exist_ok=True)
    tmp_dir = os.path.join(os.path.dirname(out_zip), "_tmp_export")
    os.makedirs(tmp_dir, exist_ok=True)

    paths: List[str] = sorted(glob.glob(os.path.join(input_dir, "*")))
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        x = tf_in(img).unsqueeze(0).to(device)
        y = G(x)[0].cpu()
        y = denorm(y)
        out = to_pil(y)
        out.save(os.path.join(tmp_dir, os.path.basename(p)))

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(glob.glob(os.path.join(tmp_dir, "*"))):
            z.write(p, arcname=os.path.basename(p))

    # clean up temp (optional; comment out if you want to inspect)
    for p in glob.glob(os.path.join(tmp_dir, "*")):
        try: os.remove(p)
        except: pass
    try: os.rmdir(tmp_dir)
    except: pass

    print(f"Exported {out_zip}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/baseline.yaml")
    ap.add_argument("--ckpt",   type=str, required=True)
    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--out",    type=str, default="submissions/group_X_basic.zip")
    args = ap.parse_args()
    export_zip(args.config, args.ckpt, args.input_dir, args.out)
