"""
Report generation for evaluation results.
Creates JSON reports and human-readable summaries.
"""

from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime


def create_report(
    scores: Dict,
    run_config: Dict,
    hashes: Dict,
    validation: Dict,
    worst_cases: list = None
) -> Dict[str, Any]:
    """
    Create a comprehensive evaluation report.
    
    Args:
        scores: Dictionary with 'mifid', 'fid', 'cosine_min_distance'
        run_config: Configuration used for this run
        hashes: Hash information for datasets
        validation: Validation results
        worst_cases: Optional list of worst memorization cases
        
    Returns:
        Complete report dictionary
    """
    report = {
        'run': {
            'name': run_config.get('name', 'unnamed_run'),
            'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
            'fake_dir': str(run_config.get('fake_dir', '')),
            'real_mode': run_config.get('real_mode', 'folder'),
            'real_dir_or_tfds': str(run_config.get('real_dir', '')),
            'num_fake': validation.get('num_fake', 0),
            'num_real': validation.get('num_real', 0),
            'img_size': run_config.get('img_size', 299),
            'batch_size': run_config.get('batch_size', 64),
            'num_workers': run_config.get('num_workers', 8),
            'warnings': validation.get('warnings', [])
        },
        'scores': {
            'mifid': round(scores.get('mifid', 0.0), 4),
            'fid': round(scores.get('fid', 0.0), 4),
            'cosine_min_distance': scores.get('cosine_min_distance', {})
        },
        'hashes': hashes,
        'notes': (
            "TorchMetrics MiFID/FID with InceptionV3 pool3 (2048 dims). "
            "uint8 input [0,255] resized to 299x299. "
            "MiFID = FID / M where M is memorization penalty from avg min cosine distance."
        )
    }
    
    # Add worst cases if provided
    if worst_cases:
        report['memorization_analysis'] = {
            'worst_cases': worst_cases,
            'description': 'Top-16 fake images with smallest cosine distance to real set (highest memorization risk)'
        }
    
    return report


def save_report(report: Dict, output_path: str, verbose: bool = True):
    """
    Save report as formatted JSON.
    
    Args:
        report: Report dictionary
        output_path: Path to output JSON file
        verbose: Whether to print confirmation
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    if verbose:
        print(f"\n✓ Report saved to: {output_path}")


def create_text_summary(report: Dict) -> str:
    """
    Create a human-readable text summary from a report.
    
    Args:
        report: Report dictionary
        
    Returns:
        Formatted text summary
    """
    run = report['run']
    scores = report['scores']
    cosine = scores.get('cosine_min_distance', {})
    
    lines = []
    lines.append("=" * 70)
    lines.append("KAGGLE MiFID EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Run Name:     {run['name']}")
    lines.append(f"Timestamp:    {run['timestamp_utc']}")
    lines.append("")
    
    lines.append("DATASETS")
    lines.append("-" * 70)
    lines.append(f"Real mode:    {run['real_mode']}")
    lines.append(f"Real path:    {run['real_dir_or_tfds']}")
    lines.append(f"Fake path:    {run['fake_dir']}")
    lines.append(f"Real images:  {run['num_real']:,}")
    lines.append(f"Fake images:  {run['num_fake']:,}")
    lines.append("")
    
    if run.get('warnings'):
        lines.append("WARNINGS")
        lines.append("-" * 70)
        for warning in run['warnings']:
            lines.append(f"⚠ {warning}")
        lines.append("")
    
    lines.append("PRIMARY METRICS")
    lines.append("-" * 70)
    lines.append(f"MiFID:        {scores['mifid']:.4f}  ← KAGGLE LEADERBOARD METRIC")
    lines.append(f"FID:          {scores['fid']:.4f}")
    lines.append("")
    
    lines.append("MEMORIZATION ANALYSIS (Min Cosine Distance)")
    lines.append("-" * 70)
    lines.append(f"Median:       {cosine.get('median', 0):.4f}")
    lines.append(f"Mean:         {cosine.get('mean', 0):.4f}")
    lines.append(f"Std:          {cosine.get('std', 0):.4f}")
    lines.append(f"P10:          {cosine.get('p10', 0):.4f}")
    lines.append(f"P90:          {cosine.get('p90', 0):.4f}")
    lines.append("")
    lines.append("Lower cosine distances = higher memorization risk")
    lines.append("MiFID penalizes low distances (i.e., high similarity to real set)")
    lines.append("")
    
    if 'memorization_analysis' in report:
        worst = report['memorization_analysis']['worst_cases']
        lines.append("WORST MEMORIZATION CASES (Top-5 shown)")
        lines.append("-" * 70)
        for i, case in enumerate(worst[:5], 1):
            lines.append(f"{i}. Distance: {case['distance']:.4f}")
            lines.append(f"   Fake:  {Path(case['fake_path']).name}")
            lines.append(f"   Real:  {Path(case['nearest_real_path']).name}")
        lines.append("")
    
    lines.append("CONFIGURATION")
    lines.append("-" * 70)
    lines.append(f"Image size:   {run['img_size']}x{run['img_size']}")
    lines.append(f"Batch size:   {run['batch_size']}")
    lines.append(f"Workers:      {run['num_workers']}")
    lines.append("")
    
    lines.append("NOTES")
    lines.append("-" * 70)
    lines.append(report.get('notes', ''))
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def save_text_summary(report: Dict, output_path: str, verbose: bool = True):
    """
    Save text summary to file.
    
    Args:
        report: Report dictionary
        output_path: Path to output text file
        verbose: Whether to print confirmation
    """
    summary = create_text_summary(report)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    if verbose:
        print(f"✓ Summary saved to: {output_path}")
        print("\n" + summary)


def save_worst_cases_csv(worst_cases: list, output_path: str):
    """
    Save worst memorization cases as CSV for easy inspection.
    
    Args:
        worst_cases: List of worst case dictionaries
        output_path: Path to output CSV file
    """
    import csv
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'fake_path', 'distance', 'cosine_similarity', 'nearest_real_path'])
        
        for i, case in enumerate(worst_cases, 1):
            writer.writerow([
                i,
                case['fake_path'],
                f"{case['distance']:.6f}",
                f"{case['cosine_similarity']:.6f}",
                case['nearest_real_path']
            ])
    
    print(f"✓ Worst cases CSV saved to: {output_path}")


def print_quick_summary(scores: Dict, validation: Dict):
    """
    Print a quick summary of key results to console.
    
    Args:
        scores: Scores dictionary
        validation: Validation dictionary
    """
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Fake images: {validation['num_fake']:,}")
    print(f"Real images: {validation['num_real']:,}")
    print(f"\n{'MiFID:':<20} {scores['mifid']:>10.4f}  ← Kaggle metric")
    print(f"{'FID:':<20} {scores['fid']:>10.4f}")
    
    cosine = scores.get('cosine_min_distance', {})
    if cosine:
        print(f"\nCosine Distance (memorization):")
        print(f"  Median: {cosine.get('median', 0):.4f}")
        print(f"  P10:    {cosine.get('p10', 0):.4f}")
        print(f"  P90:    {cosine.get('p90', 0):.4f}")
    
    print("=" * 60 + "\n")

