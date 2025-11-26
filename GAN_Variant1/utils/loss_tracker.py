"""Loss tracking utility for training."""
import csv
from pathlib import Path
from typing import Dict, Optional


class LossTracker:
    """Tracks training losses and saves to CSV."""
    
    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: Directory to save loss history CSV
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / 'losses_history.csv'
        self._file = None
        self._writer = None
        self._header_written = False
    
    def start(self):
        """Initialize CSV file and write header."""
        self._file = open(self.csv_path, 'a', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=['step', 'd_loss', 'g_loss'])
        
        # Write header only if file is new (empty)
        if self.csv_path.stat().st_size == 0:
            self._writer.writeheader()
            self._header_written = True
    
    def log(self, step: int, d_loss: float, g_loss: float):
        """Log losses for a single step."""
        if self._writer is None:
            self.start()
        
        self._writer.writerow({
            'step': step,
            'd_loss': d_loss,
            'g_loss': g_loss
        })
        self._file.flush()  # Ensure data is written immediately
    
    def close(self):
        """Close the CSV file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
    
    def load_history(self) -> Dict[str, list]:
        """
        Load existing loss history from CSV.
        
        Returns:
            Dict with 'steps', 'd_losses', 'g_losses' lists
        """
        if not self.csv_path.exists():
            return {'steps': [], 'd_losses': [], 'g_losses': []}
        
        steps = []
        d_losses = []
        g_losses = []
        
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row['step']))
                d_losses.append(float(row['d_loss']))
                g_losses.append(float(row['g_loss']))
        
        return {
            'steps': steps,
            'd_losses': d_losses,
            'g_losses': g_losses
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

