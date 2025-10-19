# src/losses.py
import torch
import torch.nn as nn

class GANLoss(nn.Module):
    """
    Baseline GAN loss:
      - 'lsgan'  → MSE against {1,0}
      - 'bce'    → BCEWithLogits against {1,0}
    """
    def __init__(self, gan_mode: str = "lsgan"):
        super().__init__()
        assert gan_mode in ("lsgan", "bce")
        self.gan_mode = gan_mode
        self.crit = nn.MSELoss() if gan_mode == "lsgan" else nn.BCEWithLogitsLoss()

    @staticmethod
    def _target(pred, is_real: bool):
        return torch.ones_like(pred) if is_real else torch.zeros_like(pred)

    def forward(self, pred, is_real: bool):
        return self.crit(pred, self._target(pred, is_real))

_L1 = nn.L1Loss()

def cycle_loss(x_recon, x_src, lam: float):
    return lam * _L1(x_recon, x_src)

def identity_loss(x_id, x_src, lam: float):
    return lam * _L1(x_id, x_src)
