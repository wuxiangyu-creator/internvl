"""
Compute RMR-Score (Render-Material Retrieval Score) — a unified metric for
cross-modal material retrieval evaluation.

RMR-Score combines three complementary aspects:
  1. Precision   — R@1 (exact top-1 hit rate)
  2. Coverage    — R@K (top-K coverage, default K=5)
  3. Ranking     — MRR  (mean reciprocal rank, penalises lower ranks)

  RMR = (w1 * R@1  +  w2 * R@K  +  w3 * MRR) / (w1 + w2 + w3)

Default weights: w1=1, w2=1, w3=1 (equal contribution).
All three sub-metrics are in [0,1], so RMR ∈ [0,1].

Additionally computes:
  - nDCG@K  (normalised discounted cumulative gain, K=10)
  - MAP@K   (mean average precision, K=10)
  - rSum    (R@1 + R@5 + R@10, the naive aggregate used in CLIP papers)

Usage:
    python compute_rmr_score.py \
        --model_name_or_path /path/to/InternVL2-1B \
        --ckpt /path/to/best_contrastive_weights.pt \
        --val_jsonl /path/to/val.jsonl \
        --image_root /path/to/images
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel


def build_transform(image_size=448):
    return transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_image(path, transform):
    return transform(Image.open(path).convert('RGB'))


# ── Metric functions ────────────────────────────────────────────────────────

def recall_at_k(ranks, k):
    """Fraction of queries whose GT appears in top-K."""
    return float((ranks <= k).mean())


def mrr(ranks):
    """Mean Reciprocal Rank."""
    return float((1.0 / ranks).mean())


def median_rank(ranks):
    return float(np.median(ranks))


def ndcg_at_k(ranks, k):
    """nDCG@K — each query has exactly one relevant item (binary relevance)."""
    gains = []
    for r in ranks:
        if r <= k:
            gains.append(1.0 / math.log2(r + 1))  # DCG for this query
        else:
            gains.append(0.0)
    ideal_dcg = 1.0 / math.log2(2)  # ideal: relevant item at rank 1
    return float(np.mean(gains) / ideal_dcg)


def map_at_k(ranks, k):
    """MAP@K — with single relevant item per query, AP = 1/rank if rank<=K."""
    aps = []
    for r in ranks:
        if r <= k:
            aps.append(1.0 / r)
        else:
            aps.append(0.0)
    return float(np.mean(aps))


def rmr_score(r1, rk, mrr_val, w1=1.0, w2=1.0, w3=1.0):
    """
    RMR-Score: unified metric combining precision, coverage, and ranking.
    All inputs should be in [0, 1].
    """
    return (w1 * r1 + w2 * rk + w3 * mrr_val) / (w1 + w2 + w3)


# ── Main ────────────────────────────────────────────────────────────────────

def compute_ranks(model, samples, image_root, transform, batch_size, device, dtype):
    """Encode all images & textures, compute per-query rank of GT texture."""
    all_img_emb, all_tex_emb = [], []

    for i in tqdm(range(0, len(samples), batch_size), desc='Encoding'):
        batch = samples[i:i + batch_size]
        imgs = torch.stack([
            load_image(os.path.join(image_root, s['image']), transform)
            for s in batch
        ]).to(device, dtype=dtype)
        texs = torch.stack([
            load_image(os.path.join(image_root, s['texture']), transform)
            for s in batch
        ]).to(device, dtype=dtype)
        with torch.no_grad():
            all_img_emb.append(model.get_cls_embedding(imgs).cpu().float())
            all_tex_emb.append(model.get_cls_embedding(texs).cpu().float())

    all_img_emb = torch.cat(all_img_emb, dim=0)
    all_tex_emb = torch.cat(all_tex_emb, dim=0)
    sim = torch.matmul(all_img_emb, all_tex_emb.T)

    N = sim.size(0)
    ranks = np.zeros(N)
    for i in range(N):
        sorted_idx = sim[i].argsort(descending=True)
        ranks[i] = (sorted_idx == i).nonzero(as_tuple=True)[0].item() + 1

    return ranks, N


def print_full_report(ranks, N, label="Model"):
    r1  = recall_at_k(ranks, 1)
    r3  = recall_at_k(ranks, 3)
    r5  = recall_at_k(ranks, 5)
    r10 = recall_at_k(ranks, 10)
    mrr_val = mrr(ranks)
    medr = median_rank(ranks)
    ndcg10 = ndcg_at_k(ranks, 10)
    map10  = map_at_k(ranks, 10)
    rsum   = r1 + r5 + r10
    rmr    = rmr_score(r1, r5, mrr_val)

    print(f"\n{'='*60}")
    print(f"  {label}  (gallery size = {N})")
    print(f"{'='*60}")
    print(f"  ── Single-item metrics ──")
    print(f"  R@1:       {r1*100:6.2f}%")
    print(f"  R@3:       {r3*100:6.2f}%")
    print(f"  R@5:       {r5*100:6.2f}%")
    print(f"  R@10:      {r10*100:6.2f}%")
    print(f"  MRR:       {mrr_val:.4f}")
    print(f"  MedR:      {medr:.0f}")
    print(f"")
    print(f"  ── Aggregate metrics ──")
    print(f"  nDCG@10:   {ndcg10:.4f}")
    print(f"  MAP@10:    {map10:.4f}")
    print(f"  rSum:      {rsum*100:.1f}  (R@1+R@5+R@10, ×100)")
    print(f"  RMR-Score: {rmr:.4f}  (ours, w1=w2=w3=1)")
    print(f"{'='*60}")

    return {
        'R@1': r1, 'R@3': r3, 'R@5': r5, 'R@10': r10,
        'MRR': mrr_val, 'MedR': medr,
        'nDCG@10': ndcg10, 'MAP@10': map10, 'rSum': rsum, 'RMR': rmr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--val_jsonl', required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load model
    print('Loading InternVL model...')
    cfg = InternVLChatConfig.from_pretrained(args.model_name_or_path)
    cfg.llm_config._attn_implementation = "eager"
    cfg.llm_config.attn_implementation = "eager"
    model = InternVLChatModel.from_pretrained(
        args.model_name_or_path, config=cfg,
        torch_dtype=dtype, attn_implementation="eager",
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.contrastive_head.load_state_dict(ckpt['contrastive_head'])
    if 'vision_model' in ckpt:
        model.vision_model.load_state_dict(ckpt['vision_model'])
        print('Loaded fine-tuned ViT weights')
    model.language_model = None
    model.mlp1 = None
    model.eval()

    transform = build_transform(args.image_size)
    samples = []
    with open(args.val_jsonl) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f'Val samples: {len(samples)}')

    ranks, N = compute_ranks(model, samples, args.image_root,
                             transform, args.batch_size, device, dtype)

    results = print_full_report(ranks, N, label="Ours (unfrozen ViT)")

    # Save results
    out_path = os.path.join(os.path.dirname(args.ckpt), 'rmr_results.json')
    with open(out_path, 'w') as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
