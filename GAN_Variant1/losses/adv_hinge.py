"""Hinge adversarial loss for GAN training."""
import torch
import torch.nn as nn


def discriminator_hinge_loss(real_preds, fake_preds):
    """
    Hinge loss for discriminator.
    
    Args:
        real_preds: list of predictions on real images (multiscale)
        fake_preds: list of predictions on fake images (multiscale)
    
    Returns:
        loss: scalar loss
    """
    loss = 0.0
    
    
    if not isinstance(real_preds, list):
        real_preds = [real_preds]
        fake_preds = [fake_preds]
    
    for real_pred, fake_pred in zip(real_preds, fake_preds):
        # Real: maximize D(real) => minimize -D(real) => hinge: max(0, 1 - D(real))
        loss_real = torch.mean(torch.relu(1.0 - real_pred))
        
        # Fake: minimize D(fake) => hinge: max(0, 1 + D(fake))
        loss_fake = torch.mean(torch.relu(1.0 + fake_pred))
        
        loss += (loss_real + loss_fake) * 0.5
    
    
    loss = loss / len(real_preds)
    
    return loss


def generator_hinge_loss(fake_preds):
    """
    Hinge loss for generator.
    
    Args:
        fake_preds: list of predictions on fake images (multiscale)
    
    Returns:
        loss: scalar loss
    """
    loss = 0.0
    
   
    if not isinstance(fake_preds, list):
        fake_preds = [fake_preds]
    
    for fake_pred in fake_preds:
        # Generator: maximize D(G(x)) => minimize -D(G(x))
        loss += -torch.mean(fake_pred)
    
    # Average over scales
    loss = loss / len(fake_preds)
    
    return loss

