"""
Unified evaluation protocol for ReTR-Bench.

Evaluates two retrieval modes:
  - visual:     ViT + ContrastiveHead     (image-only query)
  - multimodal: ViT + LLM + RetrievalHead (image+text query)

Under two gallery settings:
  - easy: full eval gallery (1,950 textures)
  - hard: per-query 50-item gallery (49 same-cluster distractors + GT)

Reports R@K, MRR, MedR, nDCG@10, MAP@10, RMR-Score, and per-cluster breakdown.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
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


def compute_metrics(ranks: np.ndarray) -> dict:
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
    return dict(R1=r1*100, R3=r3*100, R5=r5*100, R10=r10*100,
                MRR=mrr, MedR=medr, nDCG10=ndcg10, MAP10=map10, RMR=rmr)


def fmt_row(name: str, m: dict, gallery_size: int) -> str:
    return (f"  {name:<32s} | gallery={gallery_size:>5d} | "
            f"R@1={m['R1']:5.1f} R@5={m['R5']:5.1f} R@10={m['R10']:5.1f} "
            f"MRR={m['MRR']:.3f} MedR={m['MedR']:2.0f} RMR={m['RMR']:.3f}")


@torch.no_grad()
def encode_visual(model, paths, image_root, transform, device, dtype, batch_size=32, desc=""):
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc, disable=False):
        batch = paths[i : i + batch_size]
        imgs = torch.stack([
            transform(Image.open(os.path.join(image_root, p)).convert("RGB"))
            for p in batch
        ]).to(device, dtype=dtype)
        embs.append(model.get_cls_embedding(imgs).cpu().float().numpy())
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def encode_multimodal(model, tokenizer, num_image_token, samples, image_root,
                      transform, device, dtype, batch_size=16, desc=""):
    img_ctx = "<IMG_CONTEXT>"
    all_q = []
    for i in tqdm(range(0, len(samples), batch_size), desc=desc):
        batch = samples[i : i + batch_size]
        imgs = torch.stack([
            transform(Image.open(os.path.join(image_root, s["image"])).convert("RGB"))
            for s in batch
        ]).to(device, dtype=dtype)
        prompts = [img_ctx * num_image_token + "\n" + (s.get("description") or "a material texture")
                   for s in batch]
        encoded = tokenizer(prompts, max_length=num_image_token + 82,
                            padding="max_length", truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attn_mask = encoded["attention_mask"].to(device)
        q = model.get_retrieval_embedding(imgs, input_ids, attn_mask)
        all_q.append(q.cpu().float().numpy())
    return np.concatenate(all_q, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--stage2_ckpt", required=True)
    parser.add_argument("--eval_jsonl", required=True, help="benchmark_eval.jsonl")
    parser.add_argument("--hard_jsonl", required=True, help="benchmark_eval_hard.jsonl")
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", required=True, help="Output JSON file for all metrics")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # ─── Load eval data ────────────────────────────────────────────────
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
    print(f"Eval queries: {len(eval_samples)}, Hard queries: {len(hard_samples)}")

    # ─── Load model (Stage 1 + Stage 2 weights) ────────────────────────
    print("Loading model...")
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
    print("Stage 1 loaded")

    ckpt2 = torch.load(args.stage2_ckpt, map_location="cpu")
    model.retrieval_head.load_state_dict(ckpt2["retrieval_head"])
    model.mlp1.load_state_dict(ckpt2["mlp1"])
    if "contrastive_log_temp" in ckpt2:
        model.contrastive_log_temp.data.copy_(ckpt2["contrastive_log_temp"].data)
    if "lora_weights" in ckpt2:
        lora_state = ckpt2["lora_weights"]
        model.wrap_llm_lora(r=64, lora_alpha=128)
        model_state = model.language_model.state_dict()
        for k, v in lora_state.items():
            if k in model_state:
                model_state[k] = v
        model.language_model.load_state_dict(model_state)
        print(f"LoRA loaded ({len(lora_state)} tensors)")

    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    num_image_token = model.num_image_token
    transform = build_transform(448)

    # ─── EASY: full gallery ────────────────────────────────────────────
    print("\n══ EASY mode (full gallery) ══")
    eval_img_paths = [s["image"] for s in eval_samples]
    eval_tex_paths = [s["texture"] for s in eval_samples]

    img_emb = encode_visual(model, eval_img_paths, args.image_root, transform,
                             device, dtype, args.batch_size, "visual-img")
    tex_emb = encode_visual(model, eval_tex_paths, args.image_root, transform,
                             device, dtype, args.batch_size, "visual-tex")
    mm_emb = encode_multimodal(model, tokenizer, num_image_token, eval_samples,
                                args.image_root, transform, device, dtype,
                                args.batch_size, "multimodal-q")
    N = len(eval_samples)

    sim_vis_easy = img_emb @ tex_emb.T
    ranks_vis_easy = np.array([np.where(np.argsort(-sim_vis_easy[i]) == i)[0][0] + 1
                                for i in range(N)])
    m_vis_easy = compute_metrics(ranks_vis_easy)

    sim_mm_easy = mm_emb @ tex_emb.T
    ranks_mm_easy = np.array([np.where(np.argsort(-sim_mm_easy[i]) == i)[0][0] + 1
                               for i in range(N)])
    m_mm_easy = compute_metrics(ranks_mm_easy)

    print(fmt_row("Stage 1 visual (easy)", m_vis_easy, N))
    print(fmt_row("Stage 2 multimodal (easy)", m_mm_easy, N))

    # Per-cluster breakdown for EASY
    cluster_breakdown = defaultdict(lambda: {"ranks_vis": [], "ranks_mm": []})
    for i, s in enumerate(eval_samples):
        c = s["cluster_id"]
        cluster_breakdown[c]["ranks_vis"].append(ranks_vis_easy[i])
        cluster_breakdown[c]["ranks_mm"].append(ranks_mm_easy[i])
    per_cluster = {}
    for c, d in cluster_breakdown.items():
        per_cluster[int(c)] = {
            "n": len(d["ranks_vis"]),
            "visual": compute_metrics(np.array(d["ranks_vis"])),
            "multimodal": compute_metrics(np.array(d["ranks_mm"])),
        }

    # ─── HARD: per-query 50-item gallery ───────────────────────────────
    print("\n══ HARD mode (per-query 50-item gallery, same-cluster distractors) ══")
    # Build texture encoder cache for HARD gallery
    # Note: hard_samples have a gallery of 50 textures per query;
    # texture paths may overlap across queries — cache embeddings.
    print("Caching unique gallery texture embeddings...")
    unique_tex = set()
    for h in hard_samples:
        for tp in h["gallery_textures"]:
            unique_tex.add(tp)
    unique_tex = list(unique_tex)
    print(f"Unique gallery textures: {len(unique_tex)}")
    tex_emb_cache_arr = encode_visual(model, unique_tex, args.image_root, transform,
                                       device, dtype, args.batch_size, "hard-tex-cache")
    tex_to_emb = {tp: tex_emb_cache_arr[i] for i, tp in enumerate(unique_tex)}

    # Visual ranks (use img_emb already computed for eval_samples, but ensure order matches hard_samples)
    # Build query_id → idx map from eval_samples
    qid_to_visual_emb = {s["id"]: img_emb[i] for i, s in enumerate(eval_samples)}
    qid_to_mm_emb     = {s["id"]: mm_emb[i]  for i, s in enumerate(eval_samples)}

    ranks_vis_hard, ranks_mm_hard = [], []
    cluster_breakdown_hard = defaultdict(lambda: {"ranks_vis": [], "ranks_mm": []})
    for h in hard_samples:
        qid = h["query_id"]
        q_vis = qid_to_visual_emb[qid]
        q_mm  = qid_to_mm_emb[qid]
        gallery_embs = np.stack([tex_to_emb[tp] for tp in h["gallery_textures"]])  # (50, 256)
        # GT is at index 0 (by construction in build_hard_negatives)
        sim_v = gallery_embs @ q_vis
        sim_m = gallery_embs @ q_mm
        rank_v = int(np.where(np.argsort(-sim_v) == 0)[0][0]) + 1
        rank_m = int(np.where(np.argsort(-sim_m) == 0)[0][0]) + 1
        ranks_vis_hard.append(rank_v)
        ranks_mm_hard.append(rank_m)
        cluster_breakdown_hard[h["cluster_id"]]["ranks_vis"].append(rank_v)
        cluster_breakdown_hard[h["cluster_id"]]["ranks_mm"].append(rank_m)

    ranks_vis_hard = np.array(ranks_vis_hard)
    ranks_mm_hard = np.array(ranks_mm_hard)
    m_vis_hard = compute_metrics(ranks_vis_hard)
    m_mm_hard = compute_metrics(ranks_mm_hard)

    print(fmt_row("Stage 1 visual (hard)", m_vis_hard, 50))
    print(fmt_row("Stage 2 multimodal (hard)", m_mm_hard, 50))

    per_cluster_hard = {}
    for c, d in cluster_breakdown_hard.items():
        per_cluster_hard[int(c)] = {
            "n": len(d["ranks_vis"]),
            "visual": compute_metrics(np.array(d["ranks_vis"])),
            "multimodal": compute_metrics(np.array(d["ranks_mm"])),
        }

    # ─── Save ──────────────────────────────────────────────────────────
    out = {
        "n_eval": N,
        "n_hard": len(hard_samples),
        "gallery_size_easy": N,
        "gallery_size_hard": 50,
        "easy": {
            "stage1_visual":    m_vis_easy,
            "stage2_multimodal": m_mm_easy,
            "per_cluster": per_cluster,
        },
        "hard": {
            "stage1_visual":    m_vis_hard,
            "stage2_multimodal": m_mm_hard,
            "per_cluster": per_cluster_hard,
        },
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # ─── Summary table ────────────────────────────────────────────────
    print("\n╔════════════════════════════════════════════════════════════════════╗")
    print("║                ReTR-Bench results summary                          ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"{'Setting':<30s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} {'MRR':>6s} {'RMR':>6s}")
    for name, m, gs in [
        ("Stage 1 visual (easy, N=1950)", m_vis_easy, N),
        ("Stage 2 multimodal (easy)",     m_mm_easy,  N),
        ("Stage 1 visual (hard, gal=50)", m_vis_hard, 50),
        ("Stage 2 multimodal (hard)",     m_mm_hard,  50),
    ]:
        print(f"{name:<30s} {m['R1']:>6.1f} {m['R5']:>6.1f} {m['R10']:>6.1f} "
              f"{m['MRR']:>6.3f} {m['RMR']:>6.3f}")


if __name__ == "__main__":
    main()
