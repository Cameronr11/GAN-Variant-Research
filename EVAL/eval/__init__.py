"""
Kaggle MiFID Evaluator
Standalone evaluation package for "I'm Something of a Painter Myself" competition.
"""

__version__ = "1.0.0"

from eval.mifid import compute_mifid_and_fid, compute_full_evaluation
from eval.datasets import create_dataloader, ImageFolderDataset
from eval.features import compute_or_load_real_stats, extract_fake_features
from eval.report import create_report, save_report
from eval.utils import enumerate_images, validate_image_counts

__all__ = [
    'compute_mifid_and_fid',
    'compute_full_evaluation',
    'create_dataloader',
    'ImageFolderDataset',
    'compute_or_load_real_stats',
    'extract_fake_features',
    'create_report',
    'save_report',
    'enumerate_images',
    'validate_image_counts',
]

