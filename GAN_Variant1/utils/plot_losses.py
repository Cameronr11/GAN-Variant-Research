"""Plotting utilities for training losses."""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


def plot_training_losses(
    log_dir: str,
    steps: List[int],
    d_losses: List[float],
    g_losses: List[float],
    save_path: Optional[str] = None
):
    """
    Create and save a plot of discriminator and generator losses.
    
    Args:
        log_dir: Directory containing loss history
        steps: List of step numbers
        d_losses: List of discriminator losses
        g_losses: List of generator losses
        save_path: Optional custom path to save plot (default: log_dir/losses_plot.png)
    """
    if len(steps) == 0:
        print("No loss data to plot.")
        return
    
    log_path = Path(log_dir)
    if save_path is None:
        save_path = log_path / 'losses_plot.png'
    else:
        save_path = Path(save_path)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot losses
    ax.plot(steps, d_losses, label='Discriminator Loss', alpha=0.7, linewidth=1.5)
    ax.plot(steps, g_losses, label='Generator Loss', alpha=0.7, linewidth=1.5)
    
    # Formatting
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Losses: Discriminator vs Generator', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Loss plot saved to: {save_path}")

