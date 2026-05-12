"""
Unified ablation script — runs three ablations in one model load:

  A1.2  Head ablation:          raw ViT CLS (1024-d) vs trained head (256-d)
  A2.1  Gallery size scaling:   {500, 1000, 1950, 5000, full} gallery
  A2.2  Caption robustness:     full / truncated-10tok / generic / shuffled

All ablations reuse the in-domain benchmark eval set (1950 samples).
"""

import argparse
import json
import math
import os
import random
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
    ranks = np.array(ranks)
    if len(ranks) == 0:
        return None
    r1 = float((ranks <= 1).mean())
    r3 = float((ranks <= 3).mean())
    r5 = float((ranks <= 5).mean())
    r10 = float((ranks <= 10).mean())
    mrr = float((1.0 / ranks).mean())
    medr = float(np.median(ranks))
    ideal_dcg = 1.0 / math.log2(2)
    ndcg10 = float(np.mean([1.0 / math.log2(r + 1) if r <= 10 else 0.0 for r in ranks]) / ideal_dcg)
    rmr = (r1 + r5 + mrr) / 3.0
    return dict(R1=r1*100, R3=r3*100, R5=r5*100, R10=r10*100, MRR=mrr,
                MedR=medr, nDCG10=ndcg10, RMR=rmr)


@torch.no_grad()
def encode_visual_full(model, paths, root, transform, device, dtype, batch_size, desc,
                       use_head=True):
    """Encode images. If use_head=False, return raw ViT CLS (L2-normalised)."""
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc):
        batch = paths[i : i + batch_size]
        imgs = torch.stack([
            transform(Image.open(os.path.join(root, p)).convert("RGB"))
            for p in batch
        ]).to(device, dtype=dtype)

        if use_head:
            e = model.get_cls_embedding(imgs)
        else:
            vit_out = model.vision_model(
                pixel_values=imgs,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state  # (B, N+1, C)
            cls = vit_out[:, 0, :]
            e = F.normalize(cls, dim=-1)
        embs.append(e.cpu().float().numpy())
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def encode_multimodal_with_text(model, tokenizer, num_image_token, samples, root,
                                 transform, device, dtype, batch_size, desc,
                                 text_fn):
    """text_fn(sample_dict) → str — controls caption strategy."""
    img_ctx = "<IMG_CONTEXT>"
    all_q = []
    for i in tqdm(range(0, len(samples), batch_size), desc=desc):
        batch = samples[i : i + batch_size]
        imgs = torch.stack([
            transform(Image.open(os.path.join(root, s["image"])).convert("RGB"))
            for s in batch
        ]).to(device, dtype=dtype)
        prompts = [img_ctx * num_image_token + "\n" + text_fn(s) for s in batch]
        enc = tokenizer(prompts, max_length=num_image_token + 82,
                        padding="max_length", truncation=True, return_tensors="pt")
        q = model.get_retrieval_embedding(imgs, enc["input_ids"].to(device),
                                          enc["attention_mask"].to(device))
        all_q.append(q.cpu().float().numpy())
    return np.concatenate(all_q, axis=0)


def retrieval_ranks(query_emb: np.ndarray, gallery_emb: np.ndarray,
                    gt_indices: np.ndarray) -> np.ndarray:
    """For each query i, return rank of gt_indices[i] in gallery similarity ranking."""
    sim = query_emb @ gallery_emb.T  # (Nq, Ng)
    ranks = []
    for i in range(len(query_emb)):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == gt_indices[i])[0][0]) + 1
        ranks.append(rank)
    return np.array(ranks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--stage2_ckpt", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--train_jsonl", required=True,
                        help="benchmark_train.jsonl, used for extending gallery size")
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # ── Load eval + train samples ────────────────────────────────────
    with open(args.eval_jsonl) as f:
        eval_samples = [json.loads(line) for line in f if line.strip()]
    with open(args.train_jsonl) as f:
        train_samples = [json.loads(line) for line in f if line.strip()]
    print(f"Eval: {len(eval_samples)}, Train (for gallery extension): {len(train_samples)}")

    # ── Load model ───────────────────────────────────────────────────
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
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    num_image_token = model.num_image_token
    transform = build_transform(448)

    eval_img_paths = [s["image"] for s in eval_samples]
    eval_tex_paths = [s["texture"] for s in eval_samples]
    N = len(eval_samples)

    results = {}

    # ── A1.2 Head ablation ───────────────────────────────────────────
    print("\n══ A1.2: Head Ablation (raw ViT CLS vs trained head) ══")
    # With head (256-d) — current default
    img_emb_h = encode_visual_full(model, eval_img_paths, args.image_root, transform,
                                    device, dtype, args.batch_size, "img-w-head", use_head=True)
    tex_emb_h = encode_visual_full(model, eval_tex_paths, args.image_root, transform,
                                    device, dtype, args.batch_size, "tex-w-head", use_head=True)
    ranks_h = retrieval_ranks(img_emb_h, tex_emb_h, np.arange(N))
    m_with_head = compute_metrics(ranks_h)

    # Without head (raw 1024-d ViT CLS, L2-normalised)
    img_emb_n = encode_visual_full(model, eval_img_paths, args.image_root, transform,
                                    device, dtype, args.batch_size, "img-no-head", use_head=False)
    tex_emb_n = encode_visual_full(model, eval_tex_paths, args.image_root, transform,
                                    device, dtype, args.batch_size, "tex-no-head", use_head=False)
    ranks_n = retrieval_ranks(img_emb_n, tex_emb_n, np.arange(N))
    m_no_head = compute_metrics(ranks_n)

    print(f"  With head (256-d):     R@1={m_with_head['R1']:5.1f}  R@5={m_with_head['R5']:5.1f}  RMR={m_with_head['RMR']:.3f}")
    print(f"  Without head (1024-d): R@1={m_no_head['R1']:5.1f}  R@5={m_no_head['R5']:5.1f}  RMR={m_no_head['RMR']:.3f}")
    results["A1.2_head_ablation"] = {
        "with_head_256d":    m_with_head,
        "without_head_1024d": m_no_head,
    }

    # ── A2.1 Gallery size scaling ────────────────────────────────────
    print("\n══ A2.1: Gallery Size Scaling ══")
    # Build extended gallery by adding random train samples
    sizes = [500, 1000, 1950, 5000, 10000, len(eval_samples) + len(train_samples)]
    # Encode train textures for extended gallery
    train_tex_paths = [s["texture"] for s in train_samples]
    print("Encoding train textures (one-time, for gallery extension)...")
    train_tex_emb = encode_visual_full(model, train_tex_paths, args.image_root, transform,
                                        device, dtype, args.batch_size, "train-tex", use_head=True)

    # Combined: [eval_tex_emb (gt at positions 0..N-1), train_tex_emb]
    full_gallery = np.concatenate([tex_emb_h, train_tex_emb], axis=0)
    # GT for each query i is still position i in the gallery
    gt_indices = np.arange(N)

    a21_results = {}
    for gs in sizes:
        if gs <= N:
            # Sub-gallery: GT + random distractors from eval (excluding self)
            ranks_list = []
            for i in range(N):
                # Sample gs-1 distractors from eval set (excluding i)
                cand_idx = [j for j in range(N) if j != i]
                random.shuffle(cand_idx)
                cand_idx = cand_idx[:gs - 1]
                gallery_idx = [i] + cand_idx  # GT at position 0
                gallery_embs = tex_emb_h[gallery_idx]
                sim = gallery_embs @ img_emb_h[i]
                order = np.argsort(-sim)
                rank = int(np.where(order == 0)[0][0]) + 1
                ranks_list.append(rank)
            m = compute_metrics(np.array(ranks_list))
        else:
            # Extended: use eval + train_tex_emb as gallery
            n_extra = min(gs - N, len(train_tex_emb))
            extra_idx = np.random.choice(len(train_tex_emb), size=n_extra, replace=False)
            ext_gallery = np.concatenate([tex_emb_h, train_tex_emb[extra_idx]], axis=0)
            ranks_list = retrieval_ranks(img_emb_h, ext_gallery, np.arange(N))
            m = compute_metrics(ranks_list)
        a21_results[str(gs)] = m
        print(f"  gallery={gs:>6d}  R@1={m['R1']:5.1f}  R@5={m['R5']:5.1f}  R@10={m['R10']:5.1f}  RMR={m['RMR']:.3f}")
    results["A2.1_gallery_size"] = a21_results

    # ── A2.2 Caption robustness ──────────────────────────────────────
    print("\n══ A2.2: Caption Quality Robustness (Stage 2 multimodal) ══")
    # Full caption
    text_full = lambda s: s.get("description") or "a material texture"
    # Truncated (first ~10 words)
    def text_truncated(s):
        d = (s.get("description") or "").split()
        return " ".join(d[:10]) if d else "a material texture"
    # Generic
    text_generic = lambda s: "a material texture"
    # Shuffled (assign random other sample's description)
    shuffled_descs = [s.get("description", "") for s in eval_samples]
    random.shuffle(shuffled_descs)
    text_shuffled = lambda s, _idx=[0]: shuffled_descs[_idx[0] % len(shuffled_descs)]
    # Use sample id to produce a stable shuffled assignment
    id_to_shuffled = {s["id"]: shuffled_descs[i] for i, s in enumerate(eval_samples)}
    text_shuffled = lambda s: id_to_shuffled.get(s["id"], "a material texture")

    variants = {
        "full":      text_full,
        "truncated": text_truncated,
        "generic":   text_generic,
        "shuffled":  text_shuffled,
    }

    a22_results = {}
    for name, fn in variants.items():
        q_emb = encode_multimodal_with_text(
            model, tokenizer, num_image_token, eval_samples, args.image_root,
            transform, device, dtype, args.batch_size, f"mm-{name}", fn
        )
        ranks_v = retrieval_ranks(q_emb, tex_emb_h, np.arange(N))
        m = compute_metrics(ranks_v)
        a22_results[name] = m
        print(f"  caption='{name:>10s}'  R@1={m['R1']:5.1f}  R@5={m['R5']:5.1f}  R@10={m['R10']:5.1f}  RMR={m['RMR']:.3f}")
    results["A2.2_caption_quality"] = a22_results

    # ── Save all ─────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll ablation results saved to {args.output}")


if __name__ == "__main__":
    main()
