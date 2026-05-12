"""
Build hard-negative galleries for each eval query.

For each query in benchmark_eval.jsonl:
  1. Find candidates from the FULL pool (train + eval) in the same cluster.
  2. Rank candidates by cosine similarity to the query's texture embedding.
  3. Take top-K most similar as distractors (excluding ground truth itself).
  4. Gallery = [ground_truth_texture, distractor_1, ..., distractor_K]

This produces a "hard" evaluation set where the model must distinguish
visually similar materials within the same visual cluster.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_jsonl", required=True, help="benchmark_eval.jsonl")
    parser.add_argument("--all_jsonl", required=True, help="all_with_clusters.jsonl")
    parser.add_argument("--embeddings", required=True, help="texture_embeddings.npz")
    parser.add_argument("--output", required=True, help="Output JSONL with hard galleries")
    parser.add_argument("--k_distractors", type=int, default=49,
                        help="Number of distractors per query (gallery size = k+1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load embeddings + lookup
    data = np.load(args.embeddings, allow_pickle=True)
    all_ids = data["ids"].tolist()
    all_embs = data["embeddings"]
    id_to_idx = {mid: i for i, mid in enumerate(all_ids)}

    # Load full pool with cluster ids
    all_samples = []
    with open(args.all_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                all_samples.append(json.loads(line))
    print(f"Loaded {len(all_samples)} samples from full pool")

    # Group by cluster
    by_cluster = defaultdict(list)
    for s in all_samples:
        by_cluster[s["cluster_id"]].append(s)

    # Load eval queries
    eval_samples = []
    with open(args.eval_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                eval_samples.append(json.loads(line))
    print(f"Loaded {len(eval_samples)} eval queries")

    # Build hard gallery for each query
    out_records = []
    skipped = 0
    for q in eval_samples:
        q_cluster = q["cluster_id"]
        q_id = q["id"]

        if q_id not in id_to_idx:
            skipped += 1
            continue
        q_emb = all_embs[id_to_idx[q_id]]  # (256,)

        # Candidate pool: same cluster, excluding the query itself
        cands = [s for s in by_cluster[q_cluster] if s["id"] != q_id]
        if len(cands) < args.k_distractors:
            # Not enough same-cluster distractors → pad with random from other clusters
            other = [s for s in all_samples if s["cluster_id"] != q_cluster and s["id"] != q_id]
            random.shuffle(other)
            cands.extend(other[: args.k_distractors - len(cands)])

        # Rank by cosine similarity (embeddings are L2-normalised)
        cand_embs = np.stack([all_embs[id_to_idx[c["id"]]] for c in cands])  # (M, 256)
        sims = cand_embs @ q_emb  # (M,)
        top_idx = np.argsort(-sims)[: args.k_distractors]
        distractors = [cands[i] for i in top_idx]

        # Build gallery list: ground truth first, then distractors
        gallery_ids = [q_id] + [d["id"] for d in distractors]
        gallery_textures = [q["texture"]] + [d["texture"] for d in distractors]

        out_records.append({
            "query_id": q_id,
            "query_image": q["image"],
            "query_description": q.get("description", ""),
            "ground_truth_id": q_id,
            "ground_truth_texture": q["texture"],
            "cluster_id": q_cluster,
            "attributes": q.get("attributes", {}),
            "gallery_ids": gallery_ids,
            "gallery_textures": gallery_textures,
            "gallery_size": len(gallery_ids),
        })

    print(f"Built {len(out_records)} hard galleries (skipped {skipped})")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}")

    # Stats
    gallery_sizes = [r["gallery_size"] for r in out_records]
    print(f"\nGallery size: min={min(gallery_sizes)}, max={max(gallery_sizes)}, "
          f"mean={np.mean(gallery_sizes):.1f}")


if __name__ == "__main__":
    main()
