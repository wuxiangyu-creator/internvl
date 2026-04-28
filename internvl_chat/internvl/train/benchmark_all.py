"""
Run all baselines + our models on the val set and produce a unified results table.

Models:
  1. CLIP ViT-B/32  (zero-shot)
  2. CLIP ViT-L/14  (zero-shot)
  3. InternVL2-1B   (zero-shot, raw CLS token without contrastive head)
  4. Ours — frozen ViT   (contrastive head only)
  5. Ours — unfrozen ViT  (full fine-tune)

Usage:
    python benchmark_all.py \
        --val_jsonl /path/to/val.jsonl \
        --image_root /path/to/images \
        --internvl_base /path/to/InternVL2-1B \
        --ckpt_frozen /path/to/frozen/best_contrastive_weights.pt \
        --ckpt_unfrozen /path/to/unfrozen/best_contrastive_weights.pt
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_all_metrics(ranks, N):
    r1  = float((ranks <= 1).mean())
    r3  = float((ranks <= 3).mean())
    r5  = float((ranks <= 5).mean())
    r10 = float((ranks <= 10).mean())
    mrr_val = float((1.0 / ranks).mean())
    medr = float(np.median(ranks))

    # nDCG@10
    ideal_dcg = 1.0 / math.log2(2)
    ndcg10 = float(np.mean([1.0 / math.log2(r + 1) if r <= 10 else 0.0 for r in ranks]) / ideal_dcg)

    # MAP@10
    map10 = float(np.mean([1.0 / r if r <= 10 else 0.0 for r in ranks]))

    # rSum
    rsum = (r1 + r5 + r10) * 100

    # RMR-Score (w1=w2=w3=1, using R@5)
    rmr = (r1 + r5 + mrr_val) / 3.0

    return {
        'R@1': r1 * 100, 'R@3': r3 * 100, 'R@5': r5 * 100, 'R@10': r10 * 100,
        'MRR': mrr_val, 'MedR': medr,
        'nDCG@10': ndcg10, 'MAP@10': map10,
        'rSum': rsum, 'RMR': rmr,
    }


def compute_ranks_from_sims(sim):
    """Given (N, N) similarity matrix, compute 1-indexed rank of diagonal."""
    N = sim.shape[0]
    ranks = np.zeros(N)
    for i in range(N):
        sorted_idx = np.argsort(-sim[i])
        ranks[i] = np.where(sorted_idx == i)[0][0] + 1
    return ranks


# ── Data loading ────────────────────────────────────────────────────────────

def load_samples(jsonl_path):
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def build_transform(size=448):
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_clip_transform(size=224):
    """CLIP uses 224x224 with its own normalisation."""
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])


def encode_batch_pil(paths):
    """Load PIL images from paths."""
    return [Image.open(p).convert('RGB') for p in paths]


# ── CLIP baselines ──────────────────────────────────────────────────────────

def run_clip_baseline(model_name, samples, image_root, batch_size=64):
    """Run CLIP zero-shot baseline using torchvision/open_clip style."""
    try:
        import clip
    except ImportError:
        print(f"  Installing openai-clip...")
        os.system("pip install git+https://github.com/openai/CLIP.git --quiet")
        import clip

    device = torch.device('cuda')
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    all_img_emb, all_tex_emb = [], []

    for i in tqdm(range(0, len(samples), batch_size), desc=f'CLIP {model_name}'):
        batch = samples[i:i + batch_size]

        imgs = torch.stack([
            preprocess(Image.open(os.path.join(image_root, s['image'])).convert('RGB'))
            for s in batch
        ]).to(device)

        texs = torch.stack([
            preprocess(Image.open(os.path.join(image_root, s['texture'])).convert('RGB'))
            for s in batch
        ]).to(device)

        with torch.no_grad():
            img_features = model.encode_image(imgs)
            tex_features = model.encode_image(texs)
            # L2 normalise
            img_features = F.normalize(img_features.float(), dim=-1)
            tex_features = F.normalize(tex_features.float(), dim=-1)

        all_img_emb.append(img_features.cpu().numpy())
        all_tex_emb.append(tex_features.cpu().numpy())

    all_img_emb = np.concatenate(all_img_emb, axis=0)
    all_tex_emb = np.concatenate(all_tex_emb, axis=0)
    sim = all_img_emb @ all_tex_emb.T
    return compute_ranks_from_sims(sim), sim.shape[0]


# ── InternVL baselines ──────────────────────────────────────────────────────

def load_internvl_model(base_path, ckpt_path=None, device='cuda'):
    """Load InternVL model, optionally with contrastive weights."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel

    dtype = torch.bfloat16

    cfg = InternVLChatConfig.from_pretrained(base_path)
    cfg.llm_config._attn_implementation = "eager"
    cfg.llm_config.attn_implementation = "eager"

    model = InternVLChatModel.from_pretrained(
        base_path, config=cfg, torch_dtype=dtype, attn_implementation="eager",
    )

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.contrastive_head.load_state_dict(ckpt['contrastive_head'])
        if 'vision_model' in ckpt:
            model.vision_model.load_state_dict(ckpt['vision_model'])
            print(f'  Loaded fine-tuned ViT from {ckpt_path}')
        else:
            print(f'  Loaded head-only weights from {ckpt_path}')

    model.language_model = None
    model.mlp1 = None
    model = model.to(device).eval()
    return model, dtype


def run_internvl_contrastive(model, dtype, samples, image_root, batch_size=64, label="InternVL"):
    """Run InternVL with contrastive head (get_cls_embedding)."""
    device = next(model.parameters()).device
    transform = build_transform(448)

    all_img_emb, all_tex_emb = [], []
    for i in tqdm(range(0, len(samples), batch_size), desc=label):
        batch = samples[i:i + batch_size]
        imgs = torch.stack([
            transform(Image.open(os.path.join(image_root, s['image'])).convert('RGB'))
            for s in batch
        ]).to(device, dtype=dtype)
        texs = torch.stack([
            transform(Image.open(os.path.join(image_root, s['texture'])).convert('RGB'))
            for s in batch
        ]).to(device, dtype=dtype)
        with torch.no_grad():
            all_img_emb.append(model.get_cls_embedding(imgs).cpu().float().numpy())
            all_tex_emb.append(model.get_cls_embedding(texs).cpu().float().numpy())

    all_img_emb = np.concatenate(all_img_emb, axis=0)
    all_tex_emb = np.concatenate(all_tex_emb, axis=0)
    sim = all_img_emb @ all_tex_emb.T
    return compute_ranks_from_sims(sim), sim.shape[0]


def run_internvl_zeroshot(model, dtype, samples, image_root, batch_size=64):
    """Run InternVL zero-shot: raw ViT CLS token (no contrastive head), L2-normalised."""
    device = next(model.parameters()).device
    transform = build_transform(448)

    all_img_emb, all_tex_emb = [], []
    for i in tqdm(range(0, len(samples), batch_size), desc='InternVL2 zero-shot'):
        batch = samples[i:i + batch_size]
        imgs = torch.stack([
            transform(Image.open(os.path.join(image_root, s['image'])).convert('RGB'))
            for s in batch
        ]).to(device, dtype=dtype)
        texs = torch.stack([
            transform(Image.open(os.path.join(image_root, s['texture'])).convert('RGB'))
            for s in batch
        ]).to(device, dtype=dtype)
        with torch.no_grad():
            # Raw CLS token from ViT, bypassing contrastive head
            img_vit = model.vision_model(imgs, output_hidden_states=False, return_dict=True).last_hidden_state[:, 0, :]
            tex_vit = model.vision_model(texs, output_hidden_states=False, return_dict=True).last_hidden_state[:, 0, :]
            img_vit = F.normalize(img_vit.float(), dim=-1)
            tex_vit = F.normalize(tex_vit.float(), dim=-1)
            all_img_emb.append(img_vit.cpu().numpy())
            all_tex_emb.append(tex_vit.cpu().numpy())

    all_img_emb = np.concatenate(all_img_emb, axis=0)
    all_tex_emb = np.concatenate(all_tex_emb, axis=0)
    sim = all_img_emb @ all_tex_emb.T
    return compute_ranks_from_sims(sim), sim.shape[0]


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_jsonl', required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--internvl_base', required=True)
    parser.add_argument('--ckpt_frozen', default=None, help='Frozen ViT checkpoint (head-only)')
    parser.add_argument('--ckpt_unfrozen', required=True, help='Unfrozen ViT checkpoint')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output', default='benchmark_results.json')
    args = parser.parse_args()

    samples = load_samples(args.val_jsonl)
    print(f'Loaded {len(samples)} val samples\n')

    results = {}

    # ── 1. CLIP ViT-B/32 ────────────────────────────────────────────────────
    print('='*60)
    print('1/5  CLIP ViT-B/32')
    print('='*60)
    ranks, N = run_clip_baseline('ViT-B/32', samples, args.image_root, args.batch_size)
    results['CLIP ViT-B/32'] = compute_all_metrics(ranks, N)
    print_metrics(results['CLIP ViT-B/32'])
    torch.cuda.empty_cache()

    # ── 2. CLIP ViT-L/14 ────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('2/5  CLIP ViT-L/14')
    print('='*60)
    ranks, N = run_clip_baseline('ViT-L/14', samples, args.image_root, args.batch_size)
    results['CLIP ViT-L/14'] = compute_all_metrics(ranks, N)
    print_metrics(results['CLIP ViT-L/14'])
    torch.cuda.empty_cache()

    # ── 3. InternVL2-1B zero-shot ────────────────────────────────────────────
    print('\n' + '='*60)
    print('3/5  InternVL2-1B (zero-shot)')
    print('='*60)
    model_zs, dtype_zs = load_internvl_model(args.internvl_base, ckpt_path=None)
    ranks, N = run_internvl_zeroshot(model_zs, dtype_zs, samples, args.image_root, args.batch_size)
    results['InternVL2-1B (zero-shot)'] = compute_all_metrics(ranks, N)
    print_metrics(results['InternVL2-1B (zero-shot)'])
    del model_zs
    torch.cuda.empty_cache()

    # ── 4. Ours (frozen ViT) ─────────────────────────────────────────────────
    if args.ckpt_frozen and os.path.exists(args.ckpt_frozen):
        print('\n' + '='*60)
        print('4/5  Ours (frozen ViT)')
        print('='*60)
        model_f, dtype_f = load_internvl_model(args.internvl_base, ckpt_path=args.ckpt_frozen)
        ranks, N = run_internvl_contrastive(model_f, dtype_f, samples, args.image_root, args.batch_size, 'Ours frozen')
        results['Ours (frozen ViT)'] = compute_all_metrics(ranks, N)
        print_metrics(results['Ours (frozen ViT)'])
        del model_f
        torch.cuda.empty_cache()
    else:
        print('\n4/5  Ours (frozen ViT) — SKIPPED (no checkpoint)')

    # ── 5. Ours (unfrozen ViT) ───────────────────────────────────────────────
    print('\n' + '='*60)
    print('5/5  Ours (unfrozen ViT)')
    print('='*60)
    model_u, dtype_u = load_internvl_model(args.internvl_base, ckpt_path=args.ckpt_unfrozen)
    ranks, N = run_internvl_contrastive(model_u, dtype_u, samples, args.image_root, args.batch_size, 'Ours unfrozen')
    results['Ours (unfrozen ViT)'] = compute_all_metrics(ranks, N)
    print_metrics(results['Ours (unfrozen ViT)'])
    del model_u
    torch.cuda.empty_cache()

    # ── Summary table ────────────────────────────────────────────────────────
    print('\n\n' + '='*100)
    print(f'{"Method":<28s} {"R@1":>6s} {"R@3":>6s} {"R@5":>6s} {"R@10":>6s} '
          f'{"MRR":>6s} {"MedR":>6s} {"nDCG":>6s} {"MAP":>6s} {"rSum":>7s} {"RMR":>6s}')
    print('='*100)
    for name, m in results.items():
        print(f'{name:<28s} {m["R@1"]:6.1f} {m["R@3"]:6.1f} {m["R@5"]:6.1f} {m["R@10"]:6.1f} '
              f'{m["MRR"]:6.3f} {m["MedR"]:6.0f} {m["nDCG@10"]:6.3f} {m["MAP@10"]:6.3f} '
              f'{m["rSum"]:7.1f} {m["RMR"]:6.3f}')
    print('='*100)

    # Save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {args.output}')


def print_metrics(m):
    print(f'  R@1={m["R@1"]:.1f}%  R@5={m["R@5"]:.1f}%  R@10={m["R@10"]:.1f}%  '
          f'MRR={m["MRR"]:.3f}  MedR={m["MedR"]:.0f}  '
          f'nDCG@10={m["nDCG@10"]:.3f}  MAP@10={m["MAP@10"]:.3f}  RMR={m["RMR"]:.3f}')


if __name__ == '__main__':
    main()
