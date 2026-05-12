"""
Cluster texture embeddings to assign "visual category" IDs.

Uses spherical K-means (cosine distance, since embeddings are L2-normalised).
Outputs a JSONL with each sample's cluster_id, and prints per-cluster stats
including the top-N representative descriptions per cluster.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="texture_embeddings.npz")
    parser.add_argument("--jsonl", required=True, help="all_with_attrs.jsonl")
    parser.add_argument("--output", required=True, help="Output JSONL with cluster_id added")
    parser.add_argument("--k", type=int, default=20, help="Number of clusters")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect_n", type=int, default=3,
                        help="Print N representative descriptions per cluster")
    args = parser.parse_args()

    # Load embeddings
    data = np.load(args.embeddings, allow_pickle=True)
    ids = data["ids"]
    embs = data["embeddings"]
    print(f"Loaded {len(ids)} embeddings, dim={embs.shape[1]}")

    # K-means (with L2-normalised vectors, Euclidean ≈ cosine)
    print(f"Running K-means with k={args.k}...")
    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10, verbose=0)
    cluster_ids = km.fit_predict(embs)
    print(f"K-means done. inertia={km.inertia_:.2f}")

    # Build id → cluster mapping
    id_to_cluster = dict(zip(ids, cluster_ids))

    # Load JSONL and add cluster_id
    samples = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    for s in samples:
        s["cluster_id"] = int(id_to_cluster.get(s["id"], -1))

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {len(samples)} samples (with cluster_id) to {args.output}")

    # Cluster statistics
    print("\n── Cluster sizes ──────────────────────────────────────")
    cluster_counts = Counter(cluster_ids.tolist())
    for cid, n in sorted(cluster_counts.items(), key=lambda x: -x[1]):
        print(f"  cluster {cid:2d}: {n:>5d} samples ({100*n/len(ids):5.1f}%)")

    # Print representative descriptions
    print(f"\n── {args.inspect_n} sample descriptions per cluster ──")
    random.seed(args.seed)
    by_cluster = defaultdict(list)
    for s in samples:
        by_cluster[s["cluster_id"]].append(s)

    for cid in sorted(by_cluster.keys()):
        cluster_samples = by_cluster[cid]
        attr_cats = Counter(s["attributes"]["category"] for s in cluster_samples)
        attr_colors = Counter(s["attributes"]["color"] for s in cluster_samples)
        print(f"\n[Cluster {cid}] size={len(cluster_samples)}")
        print(f"  Top categories: {dict(attr_cats.most_common(3))}")
        print(f"  Top colors:     {dict(attr_colors.most_common(3))}")
        for s in random.sample(cluster_samples, min(args.inspect_n, len(cluster_samples))):
            desc = s.get("description", "")[:150]
            print(f"  • [{s['id']}] {desc}")


if __name__ == "__main__":
    main()
