"""ResNet-9 Generator (Baseline)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""
    
    def __init__(
        self,
        channels: int,
        padding_type: str = 'reflect',
        norm: str = 'instance',
        activation: str = 'relu',
        use_dropout: bool = False
    ):
        super().__init__()
        
        # Build conv block
        conv_block = []
        
        # First conv
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        else:
            # zero padding handled by conv
            pass
        
        conv_block += [
            nn.Conv2d(channels, channels, kernel_size=3, padding=0 if padding_type != 'zero' else 1),
            self._get_norm_layer(norm, channels),
            self._get_activation(activation)
        ]
        
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        
        # Second conv
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        
        conv_block += [
            nn.Conv2d(channels, channels, kernel_size=3, padding=0 if padding_type != 'zero' else 1),
            self._get_norm_layer(norm, channels)
        ]
        
        self.conv_block = nn.Sequential(*conv_block)
    
    def _get_norm_layer(self, norm: str, num_features: int):
        if norm == 'instance':
            return nn.InstanceNorm2d(num_features)
        elif norm == 'batch':
            return nn.BatchNorm2d(num_features)
        else:
            return nn.Identity()
    
    def _get_activation(self, activation: str):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        else:
            return nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class ResNetGenerator(nn.Module):
    """
    ResNet-9 generator with self-attention, channel attention, and AdaIN gates.
    
    Architecture:
        - Initial conv (7x7)
        - Downsample x2
        - 9 residual blocks (with attention at specified layers)
        - Upsample x2
        - Output conv (7x7, tanh)
    """
    
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        n_blocks: int = 9,
        n_downsampling: int = 2,
        padding_type: str = 'reflect',
        norm: str = 'instance',
        activation: str = 'relu',
        use_attention: bool = True,
        attn_layers: list = [3, 7],
        use_channel_attn: bool = True,
        channel_attn_layers: list = [5],
        use_style_dropout: bool = True,
        alpha_min: float = 0.4,
        alpha_max: float = 0.9
    ):
        super().__init__()
        
        
        
        # Initial convolution
        model = []
        if padding_type == 'reflect':
            model += [nn.ReflectionPad2d(3)]
        model += [
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0 if padding_type == 'reflect' else 3),
            nn.InstanceNorm2d(ngf) if norm == 'instance' else nn.Identity(),
            nn.ReLU(inplace=True)
        ]
        
        self.initial = nn.Sequential(*model)
        
        # Downsampling
        downsample_layers = []
        for i in range(n_downsampling):
            mult = 2 ** i
            downsample_layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2) if norm == 'instance' else nn.Identity(),
                nn.ReLU(inplace=True)
            ]
        self.downsample = nn.Sequential(*downsample_layers)
        
        # Residual blocks (baseline: just pure ResNet, no attention)
        mult = 2 ** n_downsampling
        res_channels = ngf * mult
        
        self.res_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.res_blocks.append(
                ResidualBlock(res_channels, padding_type, norm, activation)
            )
        
        # Upsampling
        upsample_layers = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            upsample_layers += [
                nn.ConvTranspose2d(
                    ngf * mult, ngf * mult // 2,
                    kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(ngf * mult // 2) if norm == 'instance' else nn.Identity(),
                nn.ReLU(inplace=True)
            ]
        self.upsample = nn.Sequential(*upsample_layers)
        
        # Output layer
        output_layers = []
        if padding_type == 'reflect':
            output_layers += [nn.ReflectionPad2d(3)]
        output_layers += [
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0 if padding_type == 'reflect' else 3),
            nn.Tanh()
        ]
        self.output = nn.Sequential(*output_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input images in [-1, 1]
        Returns:
            out: (B, 3, H, W) translated images in [-1, 1]
        """
        
        x = self.initial(x)
        
        
        x = self.downsample(x)
        
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        
        x = self.upsample(x)
        
        
        x = self.output(x)
        
        return x
    
    def get_feature_layers(self, x: torch.Tensor, layer_ids: list = None):
        """
        Extract intermediate features for PatchNCE loss.
        
        Args:
            x: (B, 3, H, W) input
            layer_ids: list of layer indices to extract (e.g., [0, 4, 8, 12, 16])
        Returns:
            feats: list of feature tensors
        """
        if layer_ids is None:
            layer_ids = [0, 4, 8, 12, 16]
        
        feats = []
        layer_idx = 0
        
        
        x = self.initial(x)
        if layer_idx in layer_ids:
            feats.append(x)
        layer_idx += 1
        
        
        for module in self.downsample:
            x = module(x)
            if isinstance(module, nn.ReLU):
                if layer_idx in layer_ids:
                    feats.append(x)
                layer_idx += 1
        
        
        for res_block in self.res_blocks:
            x = res_block(x)
            if layer_idx in layer_ids:
                feats.append(x)
            layer_idx += 1
        
        
        for module in self.upsample:
            x = module(x)
            if isinstance(module, nn.ReLU):
                if layer_idx in layer_ids:
                    feats.append(x)
                layer_idx += 1
        
        return feats

