# EVAL/scripts/select_7k.py
import argparse, os, json, numpy as np, hashlib, shutil
from pathlib import Path
from eval.features import load_folder_pool3  # uses InceptionV3 pool3
from eval.utils import list_images
from sklearn.cluster import KMeans

def sha1dir(p): 
    h = hashlib.sha1()
    for f in sorted(list_images(p)):
        h.update(Path(f).name.encode())
    return h.hexdigest()[:10]

def min_cos_dists(F_fake, F_real):
    # cosine distance = 1 - cosine similarity
    Ff = F_fake / (np.linalg.norm(F_fake,axis=1,keepdims=True)+1e-8)
    Fr = F_real / (np.linalg.norm(F_real,axis=1,keepdims=True)+1e-8)
    sims = Ff @ Fr.T
    mins = 1.0 - sims.max(axis=1)
    return mins

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--cand_roots", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tau", type=float, default=0.22)    # floor on min-cos
    ap.add_argument("--k", type=int, default=128)         # KMeans buckets
    ap.add_argument("--target", type=int, default=7000)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # Load real features (once)
    R_feats, R_paths = load_folder_pool3(args.real)
    # Collect candidate features & paths
    all_feats, all_paths, all_mincos = [], [], []
    for root in args.cand_roots:
        F, P = load_folder_pool3(root)
        mins = min_cos_dists(F, R_feats)
        all_feats.append(F); all_paths += P; all_mincos.append(mins)
    F = np.vstack(all_feats); mins = np.concatenate(all_mincos)
    F = (F - F.mean(0, keepdims=True)) / (F.std(0, keepdims=True)+1e-8)

    # Filter out too-close samples (mins < tau)
    keep = mins >= args.tau
    F, mins, all_paths = F[keep], mins[keep], [p for p, m in zip(all_paths, keep) if m]

    # Match real distribution via KMeans on real; assign fake to nearest centroids.
    km = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit(
        (R_feats - R_feats.mean(0)) / (R_feats.std(0)+1e-8)
    )
    centers = km.cluster_centers_
    # Score a fake by distance to nearest real-cluster center (proxy for FID)
    dists = ((F - centers[:,None,:])**2).sum(-1).min(0)  # [N]
    # Greedy pick: sort by (dist) ascending BUT mix in mincos to avoid memorization
    score = dists - 0.05*(mins)  # tiny preference for higher min-cos
    order = np.argsort(score)

    chosen = order[:args.target]
    sel_paths = [all_paths[i] for i in chosen]

    # Write out and copy files
    outimgs = Path(args.outdir)/"images"
    outimgs.mkdir(parents=True, exist_ok=True)
    for p in sel_paths:
        shutil.copy2(p, outimgs/Path(p).name)

    with open(Path(args.outdir)/"selection_meta.json","w") as f:
        json.dump({
            "real": args.real,
            "cand_roots": args.cand_roots,
            "tau": args.tau,
            "k": args.k,
            "selected": len(sel_paths)
        }, f, indent=2)

    print(f"Selected {len(sel_paths)} images into {outimgs}")

if __name__ == "__main__":
    main()

