"""Checkpoint I/O and EMA utilities."""
import os
import torch
from pathlib import Path
from typing import Dict, Optional
import copy


class EMA:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self):
        return {
            'decay': self.decay,
            'shadow': self.shadow
        }
    
    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']


def save_checkpoint(
    path: str,
    step: int,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    ema_G: Optional[EMA] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    metrics: Optional[Dict] = None,
    config: Optional[Dict] = None
):
    """Save training checkpoint with EMA."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'step': step,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'opt_G': opt_G.state_dict(),
        'opt_D': opt_D.state_dict(),
        'metrics': metrics or {},
        'config': config or {}
    }
    
    if ema_G is not None:
        checkpoint['ema_G'] = ema_G.state_dict()
    
    if scaler is not None:
        checkpoint['scaler'] = scaler.state_dict()
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    opt_G: Optional[torch.optim.Optimizer] = None,
    opt_D: Optional[torch.optim.Optimizer] = None,
    ema_G: Optional[EMA] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: str = 'cuda'
) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    
    if opt_G is not None and 'opt_G' in checkpoint:
        opt_G.load_state_dict(checkpoint['opt_G'])
    
    if opt_D is not None and 'opt_D' in checkpoint:
        opt_D.load_state_dict(checkpoint['opt_D'])
    
    if ema_G is not None and 'ema_G' in checkpoint:
        ema_G.load_state_dict(checkpoint['ema_G'])
    
    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    
    return checkpoint

