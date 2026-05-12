"""
Stratified train/eval split based on visual cluster IDs.

Each cluster contributes a fixed fraction to the eval set, ensuring all
visual categories are represented in evaluation.

Outputs:
  - benchmark_train.jsonl
  - benchmark_eval.jsonl
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="all_with_clusters.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_frac", type=float, default=0.10,
                        help="Fraction of each cluster to use for evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load samples
    samples = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples")

    # Group by cluster
    by_cluster = defaultdict(list)
    for s in samples:
        by_cluster[s["cluster_id"]].append(s)

    # Stratified split
    train_set, eval_set = [], []
    for cid, cluster_samples in by_cluster.items():
        random.shuffle(cluster_samples)
        n_eval = max(1, int(len(cluster_samples) * args.eval_frac))
        eval_set.extend(cluster_samples[:n_eval])
        train_set.extend(cluster_samples[n_eval:])

    random.shuffle(train_set)
    random.shuffle(eval_set)
    print(f"Train: {len(train_set)}, Eval: {len(eval_set)}")

    # Per-cluster counts
    print("\n── Per-cluster split ──")
    train_counts = Counter(s["cluster_id"] for s in train_set)
    eval_counts = Counter(s["cluster_id"] for s in eval_set)
    total_counts = Counter(s["cluster_id"] for s in samples)
    print(f"  {'cluster':>8s} {'total':>8s} {'train':>8s} {'eval':>6s} {'eval%':>6s}")
    for cid in sorted(total_counts.keys()):
        t, tr, ev = total_counts[cid], train_counts[cid], eval_counts[cid]
        print(f"  {cid:>8d} {t:>8d} {tr:>8d} {ev:>6d} {100*ev/t:>5.1f}%")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "benchmark_train.jsonl"
    eval_path = out_dir / "benchmark_eval.jsonl"
    with open(train_path, "w") as f:
        for s in train_set:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(eval_path, "w") as f:
        for s in eval_set:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Manifest
    manifest = {
        "seed": args.seed,
        "eval_frac": args.eval_frac,
        "n_total": len(samples),
        "n_train": len(train_set),
        "n_eval": len(eval_set),
        "n_clusters": len(by_cluster),
        "per_cluster_eval": {int(cid): eval_counts[cid] for cid in sorted(total_counts.keys())},
    }
    with open(out_dir / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote:\n  {train_path}\n  {eval_path}\n  {out_dir/'split_manifest.json'}")


if __name__ == "__main__":
    main()
