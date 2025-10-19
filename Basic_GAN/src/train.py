# src/train.py
import os, yaml
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from .utils import set_seed, ensure_dir
from .data import make_dataloader
from .models import ResnetGenerator, NLayerDiscriminator
from .losses import GANLoss, cycle_loss, identity_loss

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_models(cfg, device):
    G_A2B = ResnetGenerator(ngf=cfg["model"]["ngf"], n_blocks=cfg["model"]["n_blocks"]).to(device)
    G_B2A = ResnetGenerator(ngf=cfg["model"]["ngf"], n_blocks=cfg["model"]["n_blocks"]).to(device)
    D_A = NLayerDiscriminator(ndf=cfg["model"]["ndf"], spectral=cfg["model"]["spectral_norm_d"]).to(device)
    D_B = NLayerDiscriminator(ndf=cfg["model"]["ndf"], spectral=cfg["model"]["spectral_norm_d"]).to(device)
    return G_A2B, G_B2A, D_A, D_B

def lambda_rule(epoch, start_decay, total_epochs):
    if epoch < start_decay:
        return 1.0
    # linear decay → 0 at final epoch
    return max(0.0, 1.0 - float(epoch - start_decay) / float(max(1, total_epochs - start_decay)))

def train(cfg_path: str):
    cfg = load_cfg(cfg_path)
    set_seed(cfg["training"]["seed"])

    device = cfg["runtime"].get("device", "cuda" if torch.cuda.is_available() else "cpu")

    dl = make_dataloader(cfg)
    G_A2B, G_B2A, D_A, D_B = build_models(cfg, device)

    gan_mode = cfg["loss"]["gan"]
    gan_loss = GANLoss(gan_mode).to(device)

    optim_G = Adam(
        list(G_A2B.parameters()) + list(G_B2A.parameters()),
        lr=cfg["optim"]["lr_g"], betas=tuple(cfg["optim"]["betas"])
    )
    optim_D_A = Adam(D_A.parameters(), lr=cfg["optim"]["lr_d"], betas=tuple(cfg["optim"]["betas"]))
    optim_D_B = Adam(D_B.parameters(), lr=cfg["optim"]["lr_d"], betas=tuple(cfg["optim"]["betas"]))

    total_epochs = cfg["training"]["epochs"]
    start_decay = cfg["optim"]["lr_decay_after"]
    lr_lambda = lambda e: lambda_rule(e, start_decay, total_epochs)
    sched_G   = LambdaLR(optim_G,   lr_lambda)
    sched_D_A = LambdaLR(optim_D_A, lr_lambda)
    sched_D_B = LambdaLR(optim_D_B, lr_lambda)

    scaler = GradScaler(enabled=cfg["training"]["amp"])

    save_dir = cfg["training"]["save_dir"]
    ensure_dir(save_dir)

    for epoch in range(1, total_epochs + 1):
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{total_epochs}")
        for batch in pbar:
            real_A = batch["A"].to(device, non_blocking=True)  # photos
            real_B = batch["B"].to(device, non_blocking=True)  # Monet

            # ── 1) Update Generators: A→B and B→A ────────────────────────────
            optim_G.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["training"]["amp"]):
                fake_B = G_A2B(real_A)
                rec_A  = G_B2A(fake_B)

                fake_A = G_B2A(real_B)
                rec_B  = G_A2B(fake_A)

                # identity losses (feed target domain to its own generator)
                idt_B = G_A2B(real_B)
                idt_A = G_B2A(real_A)

                # GAN: generators want D(fake)=real
                loss_G_A2B = gan_loss(D_B(fake_B), True)
                loss_G_B2A = gan_loss(D_A(fake_A), True)

                # cycle + identity (baseline λ’s)
                lam_cyc = float(cfg["loss"]["lambda_cycle"])
                lam_id  = float(cfg["loss"]["lambda_identity"])
                loss_cyc = cycle_loss(rec_A, real_A, lam_cyc) + cycle_loss(rec_B, real_B, lam_cyc)
                loss_id  = identity_loss(idt_A, real_A, lam_id) + identity_loss(idt_B, real_B, lam_id)

                loss_G = loss_G_A2B + loss_G_B2A + loss_cyc + loss_id

            scaler.scale(loss_G).backward()
            scaler.step(optim_G)

            # ── 2) Update Discriminator A (real_A vs fake_A) ─────────────────
            optim_D_A.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["training"]["amp"]):
                loss_D_A_real = gan_loss(D_A(real_A), True)
                loss_D_A_fake = gan_loss(D_A(fake_A.detach()), False)
                loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)
            scaler.scale(loss_D_A).backward()
            scaler.step(optim_D_A)

            # ── 3) Update Discriminator B (real_B vs fake_B) ─────────────────
            optim_D_B.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["training"]["amp"]):
                loss_D_B_real = gan_loss(D_B(real_B), True)
                loss_D_B_fake = gan_loss(D_B(fake_B.detach()), False)
                loss_D_B = 0.5 * (loss_D_B_real + loss_D_B_fake)
            scaler.scale(loss_D_B).backward()
            scaler.step(optim_D_B)

            scaler.update()

            pbar.set_postfix({
                "G":   f"{loss_G.item():.3f}",
                "D_A": f"{loss_D_A.item():.3f}",
                "D_B": f"{loss_D_B.item():.3f}",
            })

        # epoch end: schedulers + checkpoint
        sched_G.step(); sched_D_A.step(); sched_D_B.step()
        if (epoch % cfg["training"]["save_every"] == 0) or (epoch == total_epochs):
            ck = {
                "epoch": epoch,
                "G_A2B": G_A2B.state_dict(),
                "G_B2A": G_B2A.state_dict(),
                "D_A":   D_A.state_dict(),
                "D_B":   D_B.state_dict(),
                "optim_G":   optim_G.state_dict(),
                "optim_D_A": optim_D_A.state_dict(),
                "optim_D_B": optim_D_B.state_dict(),
            }
            torch.save(ck, os.path.join(save_dir, f"ckpt_e{epoch}.pt"))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = ap.parse_args()
    train(args.config)
