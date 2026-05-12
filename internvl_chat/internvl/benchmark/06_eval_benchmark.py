"""
Unified ReTR benchmark evaluation.

Evaluates a Stage 1 + Stage 2 model on three protocols:
  1. EASY (full open gallery): query → 1950 textures
  2. HARD (per-query 50-item same-cluster gallery): query → 50 visually similar textures
  3. STRATIFIED per cluster (under EASY): R@K per visual cluster

Reports for each protocol/mode:
  R@1, R@3, R@5, R@10, MRR, MedR, nDCG@10, MAP@10, RMR-Score
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel


def build_transform(size=448):
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def compute_metrics(ranks):
    """Compute retrieval metrics given per-query ranks (1-indexed)."""
    ranks = np.asarray(ranks, dtype=np.float64)
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
        "R@1": r1*100, "R@3": r3*100, "R@5": r5*100, "R@10": r10*100,
        "MRR": mrr, "MedR": medr, "nDCG@10": ndcg10, "MAP@10": map10, "RMR": rmr,
    }


def print_metrics(name, m, indent=""):
    print(f"{indent}{name}")
    print(f"{indent}  R@1={m['R@1']:5.1f}%  R@3={m['R@3']:5.1f}%  R@5={m['R@5']:5.1f}%  R@10={m['R@10']:5.1f}%")
    print(f"{indent}  MRR={m['MRR']:.4f}  MedR={m['MedR']:.0f}  nDCG@10={m['nDCG@10']:.4f}  MAP@10={m['MAP@10']:.4f}  RMR={m['RMR']:.4f}")


def load_model(args):
    device = torch.device("cuda")
    dtype = torch.bfloat16

    cfg = InternVLChatConfig.from_pretrained(args.model_name_or_path)
    cfg.llm_config._attn_implementation = "eager"
    cfg.llm_config.attn_implementation = "eager"

    model = InternVLChatModel.from_pretrained(
        args.model_name_or_path, config=cfg,
        torch_dtype=dtype, attn_implementation="eager",
    )

    ckpt1 = torch.load(args.stage1_ckpt, map_location="cpu")
    model.contrastive_head.load_state_dict(ckpt1["contrastive_head"])
    if "vision_model" in ckpt1:
        model.vision_model.load_state_dict(ckpt1["vision_model"])
    print("Stage 1 weights loaded")

    use_stage2 = args.stage2_ckpt is not None and os.path.exists(args.stage2_ckpt)
    if use_stage2:
        ckpt2 = torch.load(args.stage2_ckpt, map_location="cpu")
        model.retrieval_head.load_state_dict(ckpt2["retrieval_head"])
        model.mlp1.load_state_dict(ckpt2["mlp1"])
        if "contrastive_log_temp" in ckpt2:
            model.contrastive_log_temp.data.copy_(ckpt2["contrastive_log_temp"].data)
        if "lora_weights" in ckpt2:
            model.wrap_llm_lora(r=64, lora_alpha=128)
            model_state = model.language_model.state_dict()
            for k, v in ckpt2["lora_weights"].items():
                if k in model_state:
                    model_state[k] = v
            model.language_model.load_state_dict(model_state)
            print("Stage 2 weights loaded (with LoRA)")

    model = model.to(device).eval()
    return model, use_stage2, device, dtype


def encode_visual(model, paths, image_root, transform, device, dtype, batch_size=16, desc="enc"):
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc):
        batch = paths[i : i + batch_size]
        imgs = torch.stack([
            transform(Image.open(os.path.join(image_root, p)).convert("RGB"))
            for p in batch
        ]).to(device, dtype=dtype)
        with torch.no_grad():
            embs.append(model.get_cls_embedding(imgs).cpu().float().numpy())
    return np.concatenate(embs, axis=0)


def encode_multimodal(model, tokenizer, samples, image_root, transform, device, dtype,
                       batch_size=16, desc="enc-mm"):
    img_ctx = "<IMG_CONTEXT>"
    num_image_token = model.num_image_token
    embs = []
    for i in tqdm(range(0, len(samples), batch_size), desc=desc):
        batch = samples[i : i + batch_size]
        imgs = torch.stack([
            transform(Image.open(os.path.join(image_root, s["query_image"] if "query_image" in s else s["image"])).convert("RGB"))
            for s in batch
        ]).to(device, dtype=dtype)

        prompts = []
        for s in batch:
            desc_text = s.get("query_description") or s.get("description") or "a material texture"
            prompts.append(img_ctx * num_image_token + "\n" + desc_text)

        encoded = tokenizer(
            prompts, max_length=num_image_token + 82,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            q_emb = model.get_retrieval_embedding(
                imgs, encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
            ).cpu().float().numpy()
        embs.append(q_emb)
    return np.concatenate(embs, axis=0)


def evaluate_full_gallery(model, tokenizer, eval_samples, image_root, transform, device, dtype,
                          use_stage2, batch_size=16):
    """EASY protocol: query → full eval gallery."""
    N = len(eval_samples)
    img_paths = [s["image"] for s in eval_samples]
    tex_paths = [s["texture"] for s in eval_samples]

    print("\n=== EASY: Full gallery retrieval ===")
    img_emb = encode_visual(model, img_paths, image_root, transform, device, dtype, batch_size, "render")
    tex_emb = encode_visual(model, tex_paths, image_root, transform, device, dtype, batch_size, "texture")

    sim_vis = img_emb @ tex_emb.T
    ranks_vis = np.array([np.where(np.argsort(-sim_vis[i]) == i)[0][0] + 1 for i in range(N)])
    m_vis = compute_metrics(ranks_vis)
    print_metrics("Visual-only (Stage 1)", m_vis, "  ")

    result = {"visual": m_vis, "ranks_visual": ranks_vis.tolist()}

    if use_stage2:
        q_emb = encode_multimodal(model, tokenizer, eval_samples, image_root, transform,
                                   device, dtype, batch_size, "multimodal")
        sim_mm = q_emb @ tex_emb.T
        ranks_mm = np.array([np.where(np.argsort(-sim_mm[i]) == i)[0][0] + 1 for i in range(N)])
        m_mm = compute_metrics(ranks_mm)
        print_metrics("Multimodal (Stage 2)", m_mm, "  ")
        result["multimodal"] = m_mm
        result["ranks_multimodal"] = ranks_mm.tolist()

    return result


def evaluate_hard_gallery(model, tokenizer, hard_samples, image_root, transform, device, dtype,
                          use_stage2, batch_size=16):
    """HARD protocol: each query has its own 50-item gallery."""
    N = len(hard_samples)
    print("\n=== HARD: 50-item same-cluster gallery per query ===")

    # Encode all query renders
    q_renders = [s["query_image"] for s in hard_samples]
    q_render_emb = encode_visual(model, q_renders, image_root, transform, device, dtype,
                                  batch_size, "render")

    # Build unique-texture cache (galleries overlap, so dedupe)
    all_tex_paths = set()
    for s in hard_samples:
        all_tex_paths.update(s["gallery_textures"])
    all_tex_paths = sorted(all_tex_paths)
    tex_emb = encode_visual(model, all_tex_paths, image_root, transform, device, dtype,
                             batch_size, "texture-pool")
    path_to_idx = {p: i for i, p in enumerate(all_tex_paths)}

    # Per-query ranking
    ranks_vis = []
    for i, s in enumerate(hard_samples):
        gallery_idx = np.array([path_to_idx[p] for p in s["gallery_textures"]])
        gallery_emb = tex_emb[gallery_idx]  # (50, 256)
        sims = gallery_emb @ q_render_emb[i]  # (50,)
        # Ground truth is index 0 by construction
        rank = int(np.where(np.argsort(-sims) == 0)[0][0]) + 1
        ranks_vis.append(rank)
    m_vis = compute_metrics(ranks_vis)
    print_metrics("Visual-only (Stage 1)", m_vis, "  ")

    result = {"visual": m_vis, "ranks_visual": ranks_vis}

    if use_stage2:
        q_mm_emb = encode_multimodal(model, tokenizer, hard_samples, image_root, transform,
                                      device, dtype, batch_size, "multimodal")
        ranks_mm = []
        for i, s in enumerate(hard_samples):
            gallery_idx = np.array([path_to_idx[p] for p in s["gallery_textures"]])
            gallery_emb = tex_emb[gallery_idx]
            sims = gallery_emb @ q_mm_emb[i]
            rank = int(np.where(np.argsort(-sims) == 0)[0][0]) + 1
            ranks_mm.append(rank)
        m_mm = compute_metrics(ranks_mm)
        print_metrics("Multimodal (Stage 2)", m_mm, "  ")
        result["multimodal"] = m_mm
        result["ranks_multimodal"] = ranks_mm

    return result


def evaluate_per_cluster(eval_samples, full_result):
    """STRATIFIED protocol: break EASY metrics down by cluster."""
    print("\n=== STRATIFIED: Per-cluster breakdown (EASY) ===")
    cluster_ids = [s.get("cluster_id", -1) for s in eval_samples]
    clusters = sorted(set(cluster_ids))

    print(f"  {'cluster':>8s} {'N':>5s} | {'Vis R@1':>8s} {'Vis R@5':>8s} {'Vis RMR':>8s} | ", end="")
    if "multimodal" in full_result:
        print(f"{'MM R@1':>7s} {'MM R@5':>7s} {'MM RMR':>7s}")
    else:
        print()

    per_cluster = {}
    for cid in clusters:
        idx = [i for i, c in enumerate(cluster_ids) if c == cid]
        if len(idx) == 0:
            continue
        ranks_vis = np.array([full_result["ranks_visual"][i] for i in idx])
        m_vis = compute_metrics(ranks_vis)

        line = f"  {cid:>8d} {len(idx):>5d} | {m_vis['R@1']:>7.1f}% {m_vis['R@5']:>7.1f}% {m_vis['RMR']:>8.4f} | "
        cluster_entry = {"n": len(idx), "visual": m_vis}

        if "ranks_multimodal" in full_result:
            ranks_mm = np.array([full_result["ranks_multimodal"][i] for i in idx])
            m_mm = compute_metrics(ranks_mm)
            line += f"{m_mm['R@1']:>6.1f}% {m_mm['R@5']:>6.1f}% {m_mm['RMR']:>7.4f}"
            cluster_entry["multimodal"] = m_mm
        print(line)
        per_cluster[int(cid)] = cluster_entry

    return per_cluster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--stage2_ckpt", default=None)
    parser.add_argument("--eval_jsonl", required=True, help="benchmark_eval.jsonl")
    parser.add_argument("--hard_jsonl", required=True, help="benchmark_eval_hard.jsonl")
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", default=None, help="Optional path to save JSON results")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, use_stage2, device, dtype = load_model(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    transform = build_transform(448)

    # Load datasets
    eval_samples = []
    with open(args.eval_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                eval_samples.append(json.loads(line))

    hard_samples = []
    with open(args.hard_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                hard_samples.append(json.loads(line))

    print(f"Eval queries: {len(eval_samples)} (easy), {len(hard_samples)} (hard)")

    # Run protocols
    easy_result = evaluate_full_gallery(
        model, tokenizer, eval_samples, args.image_root, transform,
        device, dtype, use_stage2, args.batch_size,
    )
    hard_result = evaluate_hard_gallery(
        model, tokenizer, hard_samples, args.image_root, transform,
        device, dtype, use_stage2, args.batch_size,
    )
    per_cluster = evaluate_per_cluster(eval_samples, easy_result)

    # Summary
    print("\n" + "="*80)
    print("  FINAL SUMMARY")
    print("="*80)
    print(f"  {'Protocol':<25s} {'Mode':<14s} {'R@1':>6s} {'R@5':>6s} {'MRR':>7s} {'RMR':>7s}")
    print(f"  {'EASY (gallery=1950)':<25s} {'visual':<14s} "
          f"{easy_result['visual']['R@1']:>5.1f}% {easy_result['visual']['R@5']:>5.1f}% "
          f"{easy_result['visual']['MRR']:>7.4f} {easy_result['visual']['RMR']:>7.4f}")
    if "multimodal" in easy_result:
        print(f"  {'EASY (gallery=1950)':<25s} {'multimodal':<14s} "
              f"{easy_result['multimodal']['R@1']:>5.1f}% {easy_result['multimodal']['R@5']:>5.1f}% "
              f"{easy_result['multimodal']['MRR']:>7.4f} {easy_result['multimodal']['RMR']:>7.4f}")
    print(f"  {'HARD (gallery=50)':<25s} {'visual':<14s} "
          f"{hard_result['visual']['R@1']:>5.1f}% {hard_result['visual']['R@5']:>5.1f}% "
          f"{hard_result['visual']['MRR']:>7.4f} {hard_result['visual']['RMR']:>7.4f}")
    if "multimodal" in hard_result:
        print(f"  {'HARD (gallery=50)':<25s} {'multimodal':<14s} "
              f"{hard_result['multimodal']['R@1']:>5.1f}% {hard_result['multimodal']['R@5']:>5.1f}% "
              f"{hard_result['multimodal']['MRR']:>7.4f} {hard_result['multimodal']['RMR']:>7.4f}")

    # Save
    if args.output:
        # Strip rank arrays to keep file small
        easy_save = {k: v for k, v in easy_result.items() if not k.startswith("ranks_")}
        hard_save = {k: v for k, v in hard_result.items() if not k.startswith("ranks_")}
        results = {
            "easy": easy_save,
            "hard": hard_save,
            "per_cluster_easy": per_cluster,
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
