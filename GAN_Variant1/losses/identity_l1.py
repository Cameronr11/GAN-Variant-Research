"""Identity loss (L1 reconstruction when input is already Monet)."""
import torch
import torch.nn as nn


def identity_loss(generator, monet_images):
    """
    Identity loss: G(monet) should ≈ monet.
    Prevents gratuitous recoloring during warmup.
    
    Args:
        generator: generator model
        monet_images: (B, 3, H, W) real Monet images in [-1, 1]
    
    Returns:
        loss: L1 loss
    """
    with torch.cuda.amp.autocast(enabled=False):  # Use FP32 for stability
        monet_reconstructed = generator(monet_images.float())
        loss = torch.mean(torch.abs(monet_reconstructed - monet_images))
    
    return loss

