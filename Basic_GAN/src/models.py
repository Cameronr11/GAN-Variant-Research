# src/models.py
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# ── ResNet generator (9-block) ────────────────────────────────────────────────
class ResnetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, in_c=3, out_c=3, ngf=64, n_blocks=9):
        super().__init__()
        assert n_blocks in [6, 9], "CycleGAN baseline typically uses 6 or 9 blocks"
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_c, ngf, kernel_size=7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]
        # downsample x2
        mult = 1
        for _ in range(2):
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]
            mult *= 2

        # residual blocks
        for _ in range(n_blocks):
            layers += [ResnetBlock(ngf * mult)]

        # upsample x2
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(True),
            ]
            mult //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_c, kernel_size=7),
            nn.Tanh(),  # outputs in [-1, 1]
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ── 70×70 PatchGAN discriminator ──────────────────────────────────────────────
def _maybe_sn(module: nn.Module, use_sn: bool):
    return spectral_norm(module) if use_sn else module

class NLayerDiscriminator(nn.Module):
    """
    PatchGAN with n_layers=3 produces ≈70×70 receptive field—CycleGAN baseline.
    """
    def __init__(self, in_c=3, ndf=64, n_layers=3, spectral=False):
        super().__init__()
        kw = 4
        padw = 1

        sequence = [
            nn.Conv2d(in_c, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            conv = nn.Conv2d(ndf * nf_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False)
            sequence += [
                _maybe_sn(conv, spectral),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        conv = nn.Conv2d(ndf * nf_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False)
        sequence += [
            _maybe_sn(conv, spectral),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        # final 1-channel logits map (PatchGAN)
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.net = nn.Sequential(*sequence)

    def forward(self, x):
        return self.net(x)
