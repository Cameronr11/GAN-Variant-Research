"""Optimizer and scheduler setup."""
import torch.optim as optim


def get_optimizer(model, opt_config):
    """
    Get optimizer from config.
    
    Args:
        model: PyTorch model
        opt_config: dict with 'lr', 'betas', 'weight_decay'
    
    Returns:
        optimizer: Adam optimizer
    """
    lr = opt_config.get('lr', 2e-4)
    betas = opt_config.get('betas', [0.5, 0.999])
    weight_decay = opt_config.get('weight_decay', 0.0)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay
    )
    
    return optimizer


def get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """
    Linear warmup then linear decay scheduler.
    
    Args:
        optimizer: optimizer
        warmup_steps: number of warmup steps
        total_steps: total training steps
    
    Returns:
        scheduler: learning rate scheduler
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

