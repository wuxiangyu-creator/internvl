"""
Full benchmark for render-to-material retrieval.

Baselines:
  1. Random (analytical)
  2. CLIP ViT-B/32  (zero-shot, cosine on CLS)
  3. CLIP ViT-L/14  (zero-shot, cosine on CLS)
  4. InternVL2-1B zero-shot (CLS before contrastive head)
  5. Ours – frozen ViT  (contrastive_ckpt/)
  6. Ours – unfrozen ViT (contrastive_ckpt_unfrozen/)

Outputs a markdown table + JSON suitable for LaTeX.

Usage:
    python internvl/train/benchmark_retrieval.py \
        --val_jsonl /home/ubuntu/alan_repository/texture_material/dataset/val.jsonl \
        --image_root /home/ubuntu/alan_repository/texture_material/output \
        --batch_size 64
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ── helpers ──────────────────────────────────────────────────────────────────

def build_transform(size=448):
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def clip_transform(size=224):
    """CLIP uses 224x224 with its own normalisation."""
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])


def load_samples(jsonl_path):
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def compute_metrics(sim_matrix):
    """Given (N, N) cosine similarity, compute retrieval metrics.
       Ground truth: diagonal (query i matches gallery i)."""
    N = sim_matrix.size(0)
    ranks = []
    for i in range(N):
        sorted_idx = sim_matrix[i].argsort(descending=True)
        rank = (sorted_idx == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    ranks = torch.tensor(ranks, dtype=torch.float)
    return {
        'R@1':  (ranks <= 1).float().mean().item() * 100,
        'R@3':  (ranks <= 3).float().mean().item() * 100,
        'R@5':  (ranks <= 5).float().mean().item() * 100,
        'R@10': (ranks <= 10).float().mean().item() * 100,
        'MRR':  (1.0 / ranks).mean().item(),
        'MedR': ranks.median().item(),
    }


def batch_encode(encode_fn, paths, transform, batch_size, device, dtype):
    """Generic batch encoder: encode_fn(pixel_values) -> (B, D)."""
    all_embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc='  encoding', leave=False):
        batch_paths = paths[i:i + batch_size]
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert('RGB')
            tensors.append(transform(img))
        pixel_values = torch.stack(tensors).to(device, dtype=dtype)
        with torch.no_grad():
            emb = encode_fn(pixel_values)
        all_embs.append(emb.cpu().float())
    return torch.cat(all_embs, dim=0)


# ── baseline encoders ────────────────────────────────────────────────────────

def eval_random(N):
    """Analytical random baseline."""
    return {
        'R@1':  1.0 / N * 100,
        'R@3':  3.0 / N * 100,
        'R@5':  5.0 / N * 100,
        'R@10': 10.0 / N * 100,
        'MRR':  sum(1.0 / r for r in range(1, N + 1)) / N,
        'MedR': (N + 1) / 2.0,
    }


def eval_clip(model_name, img_paths, tex_paths, batch_size, device):
    """Evaluate a CLIP model (zero-shot, image encoder only)."""
    import clip as clip_module
    model, preprocess = clip_module.load(model_name, device=device)
    model.eval()

    size = model.visual.input_resolution

    def encode_fn(pixel_values):
        return F.normalize(model.encode_image(pixel_values), dim=-1)

    tfm = clip_transform(size)
    img_emb = batch_encode(encode_fn, img_paths, tfm, batch_size, device, torch.float16)
    tex_emb = batch_encode(encode_fn, tex_paths, tfm, batch_size, device, torch.float16)
    sim = torch.matmul(img_emb, tex_emb.T)
    return compute_metrics(sim)


def eval_internvl_zeroshot(model_path, img_paths, tex_paths, batch_size, device):
    """InternVL2-1B zero-shot: CLS token without contrastive head."""
    from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
    from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel

    cfg = InternVLChatConfig.from_pretrained(model_path)
    cfg.llm_config._attn_implementation = "eager"
    cfg.llm_config.attn_implementation = "eager"

    model = InternVLChatModel.from_pretrained(
        model_path, config=cfg, torch_dtype=torch.float16, attn_implementation="eager",
    )
    model.language_model = None
    model.mlp1 = None
    model = model.to(device).eval()

    def encode_fn(pixel_values):
        vit_out = model.vision_model(
            pixel_values=pixel_values, output_hidden_states=False, return_dict=True,
        ).last_hidden_state
        cls = vit_out[:, 0, :]
        return F.normalize(cls, dim=-1)

    tfm = build_transform(448)
    img_emb = batch_encode(encode_fn, img_paths, tfm, batch_size, device, torch.float16)
    tex_emb = batch_encode(encode_fn, tex_paths, tfm, batch_size, device, torch.float16)
    sim = torch.matmul(img_emb, tex_emb.T)
    return compute_metrics(sim)


def eval_ours(model_path, ckpt_path, img_paths, tex_paths, batch_size, device):
    """Our fine-tuned InternVL2-1B with contrastive head."""
    from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
    from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel

    cfg = InternVLChatConfig.from_pretrained(model_path)
    cfg.llm_config._attn_implementation = "eager"
    cfg.llm_config.attn_implementation = "eager"

    model = InternVLChatModel.from_pretrained(
        model_path, config=cfg, torch_dtype=torch.float16, attn_implementation="eager",
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.contrastive_head.load_state_dict(ckpt['contrastive_head'])
    if 'vision_model' in ckpt:
        model.vision_model.load_state_dict(ckpt['vision_model'])
    model.language_model = None
    model.mlp1 = None
    model = model.to(device).eval()

    def encode_fn(pixel_values):
        return model.get_cls_embedding(pixel_values)

    tfm = build_transform(448)
    img_emb = batch_encode(encode_fn, img_paths, tfm, batch_size, device, torch.float16)
    tex_emb = batch_encode(encode_fn, tex_paths, tfm, batch_size, device, torch.float16)
    sim = torch.matmul(img_emb, tex_emb.T)
    return compute_metrics(sim)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_jsonl', default='/home/ubuntu/alan_repository/texture_material/dataset/val.jsonl')
    parser.add_argument('--image_root', default='/home/ubuntu/alan_repository/texture_material/output')
    parser.add_argument('--internvl_base', default='/home/ubuntu/alan_repository/models/InternVL2-1B')
    parser.add_argument('--ckpt_frozen', default='/home/ubuntu/alan_repository/texture_material/contrastive_ckpt/best_contrastive_weights.pt')
    parser.add_argument('--ckpt_unfrozen', default='/home/ubuntu/alan_repository/texture_material/contrastive_ckpt_unfrozen/best_contrastive_weights.pt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output', default='/home/ubuntu/alan_repository/InternVL/paper/benchmark_results.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    samples = load_samples(args.val_jsonl)
    N = len(samples)
    img_paths = [os.path.join(args.image_root, s['image']) for s in samples]
    tex_paths = [os.path.join(args.image_root, s['texture']) for s in samples]
    print(f'Gallery size: {N}\n')

    results = {}

    # 1. Random
    print('[1/6] Random baseline (analytical)')
    results['Random'] = eval_random(N)

    # 2. CLIP ViT-B/32
    try:
        print('[2/6] CLIP ViT-B/32')
        results['CLIP ViT-B/32'] = eval_clip('ViT-B/32', img_paths, tex_paths, args.batch_size, device)
    except Exception as e:
        print(f'  SKIP: {e}')
        results['CLIP ViT-B/32'] = None

    # 3. CLIP ViT-L/14
    try:
        print('[3/6] CLIP ViT-L/14')
        results['CLIP ViT-L/14'] = eval_clip('ViT-L/14', img_paths, tex_paths, args.batch_size, device)
    except Exception as e:
        print(f'  SKIP: {e}')
        results['CLIP ViT-L/14'] = None

    # 4. InternVL2-1B zero-shot
    print('[4/6] InternVL2-1B zero-shot (CLS, no contrastive head)')
    results['InternVL2-1B (zero-shot)'] = eval_internvl_zeroshot(
        args.internvl_base, img_paths, tex_paths, args.batch_size, device)
    torch.cuda.empty_cache()

    # 5. Ours – frozen ViT
    if os.path.exists(args.ckpt_frozen):
        print('[5/6] Ours (frozen ViT)')
        results['Ours (frozen ViT)'] = eval_ours(
            args.internvl_base, args.ckpt_frozen, img_paths, tex_paths, args.batch_size, device)
        torch.cuda.empty_cache()
    else:
        print(f'[5/6] SKIP frozen ViT ckpt not found: {args.ckpt_frozen}')
        results['Ours (frozen ViT)'] = None

    # 6. Ours – unfrozen ViT
    print('[6/6] Ours (unfrozen ViT)')
    results['Ours (unfrozen ViT)'] = eval_ours(
        args.internvl_base, args.ckpt_unfrozen, img_paths, tex_paths, args.batch_size, device)

    # ── Print table ──────────────────────────────────────────────────────────
    header = f'{"Method":<30s} {"R@1":>6s} {"R@3":>6s} {"R@5":>6s} {"R@10":>6s} {"MRR":>6s} {"MedR":>6s}'
    sep = '-' * len(header)
    print(f'\n{sep}\n{header}\n{sep}')
    for name, m in results.items():
        if m is None:
            print(f'{name:<30s}   (skipped)')
            continue
        print(f'{name:<30s} {m["R@1"]:6.2f} {m["R@3"]:6.2f} {m["R@5"]:6.2f} '
              f'{m["R@10"]:6.2f} {m["MRR"]:6.4f} {m["MedR"]:6.1f}')
    print(sep)

    # ── Save JSON ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'gallery_size': N, 'results': results}, f, indent=2)
    print(f'\nResults saved to {args.output}')

    # ── Print LaTeX table snippet ────────────────────────────────────────────
    print('\n% --- LaTeX table (copy-paste into paper) ---')
    print(r'\begin{tabular}{lcccccr}')
    print(r'\toprule')
    print(r'Method & R@1 & R@3 & R@5 & R@10 & MRR & Med.R \\')
    print(r'\midrule')
    for name, m in results.items():
        if m is None:
            continue
        bold = name.startswith('Ours (unfrozen')
        fmt = lambda v, dec=1: (r'\textbf{' + f'{v:.{dec}f}' + '}') if bold else f'{v:.{dec}f}'
        print(f'{name} & {fmt(m["R@1"])} & {fmt(m["R@3"])} & {fmt(m["R@5"])} & '
              f'{fmt(m["R@10"])} & {fmt(m["MRR"], 3)} & {fmt(m["MedR"], 0)} \\\\')
    print(r'\bottomrule')
    print(r'\end{tabular}')


if __name__ == '__main__':
    main()
