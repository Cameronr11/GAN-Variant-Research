"""Multiscale PatchGAN Discriminator with Spectral Normalization."""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator (70x70 receptive field).
    """
    
    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        # Helper function to add spectral norm
        def norm_layer(layer):
            return spectral_norm(layer) if use_spectral_norm else layer
        
        # Build discriminator
        sequence = [
            norm_layer(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                norm_layer(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            norm_layer(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Output layer (no activation for hinge loss)
        sequence += [
            norm_layer(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        ]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            out: (B, 1, H', W') patch predictions
        """
        return self.model(x)
    
    def get_intermediate_features(self, x: torch.Tensor):
        """Extract intermediate features for feature matching loss."""
        features = []
        for i, module in enumerate(self.model):
            x = module(x)
            if isinstance(module, nn.LeakyReLU):
                features.append(x)
        return features


class MultiscaleDiscriminator(nn.Module):
    """
    Multiscale PatchGAN discriminator.
    Applies discriminators at multiple scales (downsampled inputs).
    """
    
    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        num_scales: int = 3,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()
        
        for _ in range(num_scales):
            self.discriminators.append(
                PatchGANDiscriminator(input_nc, ndf, n_layers, use_spectral_norm)
            )
        
        # Downsampling
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            outputs: list of (B, 1, H', W') predictions at each scale
        """
        outputs = []
        
        for i, discriminator in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            outputs.append(discriminator(x))
        
        return outputs
    
    def get_intermediate_features(self, x: torch.Tensor):
        """Extract features at all scales for feature matching."""
        all_features = []
        
        for i, discriminator in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            features = discriminator.get_intermediate_features(x)
            all_features.append(features)
        
        return all_features

