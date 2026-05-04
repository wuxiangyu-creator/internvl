"""
Evaluate Stage 2 multimodal contrastive model.
Tests both retrieval modes:
  1. Visual-only: ViT + ContrastiveHead (same as Stage 1)
  2. Multimodal:  ViT + LLM + RetrievalHead (image + text query)
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel


def build_transform(size=448):
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def compute_metrics(ranks, N):
    r1 = float((ranks <= 1).mean())
    r3 = float((ranks <= 3).mean())
    r5 = float((ranks <= 5).mean())
    r10 = float((ranks <= 10).mean())
    mrr = float((1.0 / ranks).mean())
    medr = float(np.median(ranks))
    ideal_dcg = 1.0 / math.log2(2)
    ndcg10 = float(np.mean([1.0 / math.log2(r + 1) if r <= 10 else 0.0 for r in ranks]) / ideal_dcg)
    map10 = float(np.mean([1.0 / r if r <= 10 else 0.0 for r in ranks]))
    rmr = (r1 + r5 + mrr) / 3.0
    return {
        'R@1': r1*100, 'R@3': r3*100, 'R@5': r5*100, 'R@10': r10*100,
        'MRR': mrr, 'MedR': medr, 'nDCG@10': ndcg10, 'MAP@10': map10, 'RMR': rmr,
    }


def print_metrics(name, m):
    print(f'\n{"="*60}')
    print(f'  {name}')
    print(f'{"="*60}')
    print(f'  R@1={m["R@1"]:.1f}%  R@3={m["R@3"]:.1f}%  R@5={m["R@5"]:.1f}%  R@10={m["R@10"]:.1f}%')
    print(f'  MRR={m["MRR"]:.4f}  MedR={m["MedR"]:.0f}  nDCG@10={m["nDCG@10"]:.4f}  MAP@10={m["MAP@10"]:.4f}  RMR={m["RMR"]:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--stage1_ckpt', required=True, help='Stage 1 contrastive weights')
    parser.add_argument('--stage2_ckpt', required=True, help='Stage 2 multimodal weights')
    parser.add_argument('--val_jsonl', required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Load samples
    samples = []
    with open(args.val_jsonl) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f'Val samples: {len(samples)}')

    transform = build_transform(448)

    # ── Load model with Stage 1 weights ──────────────────────────────────
    print('Loading model...')
    cfg = InternVLChatConfig.from_pretrained(args.model_name_or_path)
    cfg.llm_config._attn_implementation = "eager"
    cfg.llm_config.attn_implementation = "eager"

    model = InternVLChatModel.from_pretrained(
        args.model_name_or_path, config=cfg,
        torch_dtype=dtype, attn_implementation="eager",
    )

    # Load Stage 1 weights (ViT + ContrastiveHead)
    ckpt1 = torch.load(args.stage1_ckpt, map_location='cpu')
    model.contrastive_head.load_state_dict(ckpt1['contrastive_head'])
    if 'vision_model' in ckpt1:
        model.vision_model.load_state_dict(ckpt1['vision_model'])
    print('Stage 1 weights loaded')

    # Load Stage 2 weights (RetrievalHead + MLP + LoRA)
    ckpt2 = torch.load(args.stage2_ckpt, map_location='cpu')
    model.retrieval_head.load_state_dict(ckpt2['retrieval_head'])
    model.mlp1.load_state_dict(ckpt2['mlp1'])
    if 'contrastive_log_temp' in ckpt2:
        model.contrastive_log_temp.data.copy_(ckpt2['contrastive_log_temp'].data)

    # Load LoRA weights
    if 'lora_weights' in ckpt2:
        lora_state = ckpt2['lora_weights']
        # First apply LoRA structure
        model.wrap_llm_lora(r=64, lora_alpha=128)
        # Then load trained LoRA weights
        model_state = model.language_model.state_dict()
        for k, v in lora_state.items():
            if k in model_state:
                model_state[k] = v
        model.language_model.load_state_dict(model_state)
        print(f'LoRA weights loaded ({len(lora_state)} tensors)')

    model = model.to(device).eval()

    # Set img_context_token_id
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    num_image_token = model.num_image_token

    # ── Test 1: Visual-only retrieval (ContrastiveHead) ──────────────────
    print('\n--- Encoding visual embeddings ---')
    all_img_emb, all_tex_emb = [], []
    for i in tqdm(range(0, len(samples), args.batch_size), desc='Visual'):
        batch = samples[i:i + args.batch_size]
        imgs = torch.stack([
            transform(Image.open(os.path.join(args.image_root, s['image'])).convert('RGB'))
            for s in batch
        ]).to(device, dtype=dtype)
        texs = torch.stack([
            transform(Image.open(os.path.join(args.image_root, s['texture'])).convert('RGB'))
            for s in batch
        ]).to(device, dtype=dtype)
        with torch.no_grad():
            all_img_emb.append(model.get_cls_embedding(imgs).cpu().float().numpy())
            all_tex_emb.append(model.get_cls_embedding(texs).cpu().float().numpy())

    all_img_emb = np.concatenate(all_img_emb, axis=0)
    all_tex_emb = np.concatenate(all_tex_emb, axis=0)
    N = len(samples)

    # Visual retrieval ranks
    sim_vis = all_img_emb @ all_tex_emb.T
    ranks_vis = np.array([np.where(np.argsort(-sim_vis[i]) == i)[0][0] + 1 for i in range(N)])
    m_vis = compute_metrics(ranks_vis, N)
    print_metrics('Visual-only (ContrastiveHead)', m_vis)

    # ── Test 2: Multimodal retrieval (RetrievalHead) ─────────────────────
    print('\n--- Encoding multimodal embeddings ---')
    all_query_emb = []
    img_ctx = '<IMG_CONTEXT>'

    for i in tqdm(range(0, len(samples), args.batch_size), desc='Multimodal'):
        batch = samples[i:i + args.batch_size]

        imgs = torch.stack([
            transform(Image.open(os.path.join(args.image_root, s['image'])).convert('RGB'))
            for s in batch
        ]).to(device, dtype=dtype)

        # Build text inputs with image context tokens
        prompts = []
        for s in batch:
            desc = s.get('description', 'a material texture')
            prompt = img_ctx * num_image_token + '\n' + desc
            prompts.append(prompt)

        encoded = tokenizer(
            prompts,
            max_length=num_image_token + 82,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            query_emb = model.get_retrieval_embedding(
                imgs, input_ids, attention_mask
            )
            all_query_emb.append(query_emb.cpu().float().numpy())

    all_query_emb = np.concatenate(all_query_emb, axis=0)

    # Multimodal retrieval ranks (query_emb vs tex_emb from ContrastiveHead)
    sim_mm = all_query_emb @ all_tex_emb.T
    ranks_mm = np.array([np.where(np.argsort(-sim_mm[i]) == i)[0][0] + 1 for i in range(N)])
    m_mm = compute_metrics(ranks_mm, N)
    print_metrics('Multimodal (RetrievalHead, image+text)', m_mm)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'{"Method":<40s} {"R@1":>6s} {"R@5":>6s} {"R@10":>6s} {"MRR":>6s} {"MedR":>5s} {"RMR":>6s}')
    print(f'{"="*70}')
    print(f'{"Visual-only (ContrastiveHead)":<40s} {m_vis["R@1"]:6.1f} {m_vis["R@5"]:6.1f} {m_vis["R@10"]:6.1f} {m_vis["MRR"]:6.3f} {m_vis["MedR"]:5.0f} {m_vis["RMR"]:6.3f}')
    print(f'{"Multimodal (RetrievalHead)":<40s} {m_mm["R@1"]:6.1f} {m_mm["R@5"]:6.1f} {m_mm["R@10"]:6.1f} {m_mm["MRR"]:6.3f} {m_mm["MedR"]:5.0f} {m_mm["RMR"]:6.3f}')
    print(f'{"="*70}')

    # Save
    results = {'visual_only': m_vis, 'multimodal': m_mm}
    out_path = os.path.join(os.path.dirname(args.stage2_ckpt), 'eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
