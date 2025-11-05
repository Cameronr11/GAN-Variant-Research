"""AMP utilities for mixed precision training."""
import torch


class AMPContext:
    """Wrapper for AMP context and grad scaler."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.scaler = torch.amp.GradScaler('cuda', enabled=enabled)
    
    def autocast(self):
        """Return autocast context."""
        return torch.amp.autocast('cuda', enabled=self.enabled)
    
    def scale_and_step(self, loss, optimizer, scaler=None):
        """Scale loss, backward, and optimizer step."""
        if scaler is None:
            scaler = self.scaler
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    def scale_backward(self, loss):
        """Scale and backward only."""
        self.scaler.scale(loss).backward()
    
    def step_optimizer(self, optimizer, max_grad_norm=None):
        """Step optimizer with scaler and optional gradient clipping."""
        # Unscale gradients for clipping
        if max_grad_norm is not None:
            self.scaler.unscale_(optimizer)
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None],
                max_grad_norm
            )
        
        self.scaler.step(optimizer)
        self.scaler.update()

