"""
CLI for offline evaluation of CycleGAN outputs.

Computes FID, KID (optional), memorization proxy, and MiFID' heuristic.
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
import torch

from src.eval_metrics import compute_fid, compute_kid, compute_memorization_proxy
from src.utils_eval import write_json, append_csv, short_hash, list_images


def load_config(config_path: str = "configs/baseline.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate CycleGAN outputs with FID, KID, and memorization proxy"
    )
    
    parser.add_argument(
        "--real_dir",
        type=str,
        default=None,
        help="Directory containing real Monet images (default: from config)"
    )
    
    parser.add_argument(
        "--gen_dir",
        type=str,
        required=True,
        help="Directory containing generated images"
    )
    
    parser.add_argument(
        "--subset_real",
        type=int,
        default=1000,
        help="Max number of real images to use (None = all)"
    )
    
    parser.add_argument(
        "--subset_gen",
        type=int,
        default=1000,
        help="Max number of generated images to use (None = all)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for embedding extraction"
    )
    
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Epsilon for MiFID' calculation (prevents division by zero)"
    )
    
    parser.add_argument(
        "--no_kid",
        action="store_true",
        help="Skip KID computation (faster)"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON file (default: results/eval_<timestamp>.json)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility when subsetting"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to config file (default: configs/baseline.yaml)"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for FID/KID computation"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation routine."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine real_dir from config if not provided
    if args.real_dir is None:
        data_root = config.get('data', {}).get('root', 'data')
        domain_b = config.get('data', {}).get('domain_b', 'monet_jpg')
        args.real_dir = os.path.join(data_root, domain_b)
    
    # Validate directories
    if not os.path.exists(args.real_dir):
        print(f"Error: real_dir does not exist: {args.real_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.gen_dir):
        print(f"Error: gen_dir does not exist: {args.gen_dir}")
        sys.exit(1)
    
    # Auto-detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate timestamp and output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out is None:
        args.out = f"results/eval_{timestamp}.json"
    
    # Count available images
    real_images = list_images(args.real_dir)
    gen_images = list_images(args.gen_dir)
    
    real_used = min(args.subset_real, len(real_images)) if args.subset_real else len(real_images)
    gen_used = min(args.subset_gen, len(gen_images)) if args.subset_gen else len(gen_images)
    
    print(f"\n{'='*80}")
    print(f"Evaluation @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"Real:       {args.real_dir}")
    print(f"            ({len(real_images)} available, using {real_used})")
    print(f"Generated:  {args.gen_dir}")
    print(f"            ({len(gen_images)} available, using {gen_used})")
    print(f"{'='*80}\n")
    
    # Create temporary directories for subsetted images if needed
    # For FID/KID, we need to work with actual directories
    # We'll create symlinks or use clean-fid's ability to work with lists
    # For simplicity, we'll use the full directories for FID/KID
    # and only subset for memorization proxy
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "real_dir": str(args.real_dir),
        "gen_dir": str(args.gen_dir),
        "counts": {
            "real_available": len(real_images),
            "gen_available": len(gen_images),
            "real_used": real_used,
            "gen_used": gen_used
        }
    }
    
    # Compute FID
    print("Computing FID...")
    try:
        fid_score = compute_fid(args.real_dir, args.gen_dir, num_workers=args.num_workers)
        results["fid"] = fid_score
        print(f"FID: {fid_score:.2f}")
    except Exception as e:
        print(f"Error computing FID: {e}")
        results["fid"] = None
    
    # Compute KID (optional)
    if not args.no_kid:
        print("\nComputing KID...")
        try:
            kid_score = compute_kid(args.real_dir, args.gen_dir, num_workers=args.num_workers)
            results["kid"] = kid_score
            print(f"KID: {kid_score:.4e}")
        except Exception as e:
            print(f"Error computing KID: {e}")
            results["kid"] = None
    else:
        results["kid"] = None
    
    # Compute memorization proxy
    print("\nComputing memorization proxy...")
    try:
        mem_proxy = compute_memorization_proxy(
            real_dir=args.real_dir,
            gen_dir=args.gen_dir,
            max_real=args.subset_real,
            max_gen=args.subset_gen,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed
        )
        results["memorization_min_cosine_avg"] = mem_proxy
        print(f"\nMemorization (avg min cosine dist): {mem_proxy:.4f}")
    except Exception as e:
        print(f"Error computing memorization proxy: {e}")
        results["memorization_min_cosine_avg"] = None
        mem_proxy = None
    
    # Compute MiFID'
    results["epsilon"] = args.epsilon
    if results.get("fid") is not None and mem_proxy is not None:
        mifid_prime = results["fid"] / max(mem_proxy, args.epsilon)
        results["mifid_prime"] = mifid_prime
    else:
        results["mifid_prime"] = None
    
    # Create config hash
    config_dict = {
        "config": config,
        "args": {
            "subset_real": args.subset_real,
            "subset_gen": args.subset_gen,
            "batch_size": args.batch_size,
            "epsilon": args.epsilon,
            "seed": args.seed
        }
    }
    results["config_hash"] = short_hash(config_dict)
    
    # Print summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"FID:                        {results.get('fid', 'N/A'):.2f}" if results.get('fid') else "FID:                        N/A")
    if results.get("kid") is not None:
        print(f"KID:                        {results['kid']:.4e}")
    else:
        print(f"KID:                        {'N/A' if not args.no_kid else 'Skipped'}")
    
    if results.get("memorization_min_cosine_avg") is not None:
        print(f"Memorization (avg min cosine dist): {results['memorization_min_cosine_avg']:.4f}")
    else:
        print(f"Memorization (avg min cosine dist): N/A")
    
    if results.get("mifid_prime") is not None:
        print(f"MiFID' (heuristic):         {results['mifid_prime']:.2f}")
        print(f"                            (FID / max(d_avg, {args.epsilon}))")
    else:
        print(f"MiFID' (heuristic):         N/A")
    
    print(f"{'='*80}\n")
    
    # Save JSON
    write_json(args.out, results)
    print(f"Saved: {args.out}")
    
    # Append to CSV log
    csv_path = Path(args.out).parent / "eval_log.csv"
    csv_row = {
        "timestamp": results["timestamp"],
        "real_dir": results["real_dir"],
        "gen_dir": results["gen_dir"],
        "real_used": results["counts"]["real_used"],
        "gen_used": results["counts"]["gen_used"],
        "fid": results.get("fid", ""),
        "kid": results.get("kid", ""),
        "memorization_avg_cosine": results.get("memorization_min_cosine_avg", ""),
        "epsilon": results["epsilon"],
        "mifid_prime": results.get("mifid_prime", ""),
        "config_hash": results["config_hash"]
    }
    
    ordered_cols = [
        "timestamp", "real_dir", "gen_dir", "real_used", "gen_used",
        "fid", "kid", "memorization_avg_cosine", "epsilon", "mifid_prime", "config_hash"
    ]
    
    append_csv(str(csv_path), csv_row, ordered_cols)
    print(f"Appended row to: {csv_path}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

