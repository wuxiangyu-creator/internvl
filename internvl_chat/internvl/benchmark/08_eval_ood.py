"""
Evaluate ReTR on the Out-Of-Distribution (OOD) benchmark.

These samples come from texture categories never seen during training:
  UGC_Poster, SceneInteraction, UGC_terrain, UrbanInteractiveObjects,
  Smallthing, UGC_LOGO, Rock, Traffic

Reports overall metrics + per-category breakdown for both retrieval modes.
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


def compute_metrics(ranks):
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
    map10 = float(np.mean([1.0 / r if r <= 10 else 0.0 for r in ranks]))
    rmr = (r1 + r5 + mrr) / 3.0
    return dict(R1=r1*100, R3=r3*100, R5=r5*100, R10=r10*100,
                MRR=mrr, MedR=medr, nDCG10=ndcg10, MAP10=map10, RMR=rmr)


@torch.no_grad()
def encode_visual(model, paths, root, transform, device, dtype, batch_size, desc):
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc):
        batch = paths[i : i + batch_size]
        imgs = torch.stack([
            transform(Image.open(os.path.join(root, p)).convert("RGB"))
            for p in batch
        ]).to(device, dtype=dtype)
        embs.append(model.get_cls_embedding(imgs).cpu().float().numpy())
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def encode_multimodal(model, tokenizer, num_image_token, samples, root,
                      transform, device, dtype, batch_size, desc):
    img_ctx = "<IMG_CONTEXT>"
    all_q = []
    for i in tqdm(range(0, len(samples), batch_size), desc=desc):
        batch = samples[i : i + batch_size]
        imgs = torch.stack([
            transform(Image.open(os.path.join(root, s["image"])).convert("RGB"))
            for s in batch
        ]).to(device, dtype=dtype)
        prompts = [img_ctx * num_image_token + "\n" + (s.get("description") or "a material texture")
                   for s in batch]
        enc = tokenizer(prompts, max_length=num_image_token + 82,
                        padding="max_length", truncation=True, return_tensors="pt")
        q = model.get_retrieval_embedding(imgs, enc["input_ids"].to(device),
                                          enc["attention_mask"].to(device))
        all_q.append(q.cpu().float().numpy())
    return np.concatenate(all_q, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--stage2_ckpt", required=True)
    parser.add_argument("--ood_jsonl", required=True)
    parser.add_argument("--image_root", required=True,
                        help="Root for OOD fig.png/texture.png files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    samples = []
    with open(args.ood_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"OOD samples: {len(samples)}")

    # ── Load model ────────────────────────────────────────────────────
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

    # ── Encode ───────────────────────────────────────────────────────
    img_paths = [s["image"] for s in samples]
    tex_paths = [s["texture"] for s in samples]

    print("\nEncoding OOD images...")
    img_emb = encode_visual(model, img_paths, args.image_root, transform,
                             device, dtype, args.batch_size, "ood-img")
    tex_emb = encode_visual(model, tex_paths, args.image_root, transform,
                             device, dtype, args.batch_size, "ood-tex")
    mm_emb = encode_multimodal(model, tokenizer, num_image_token, samples,
                                args.image_root, transform, device, dtype,
                                args.batch_size, "ood-mm")
    N = len(samples)

    # ── OOD-only gallery: gallery = all OOD textures ──────────────────
    sim_vis = img_emb @ tex_emb.T
    sim_mm = mm_emb @ tex_emb.T
    ranks_vis = np.array([np.where(np.argsort(-sim_vis[i]) == i)[0][0] + 1 for i in range(N)])
    ranks_mm = np.array([np.where(np.argsort(-sim_mm[i]) == i)[0][0] + 1 for i in range(N)])

    m_vis = compute_metrics(ranks_vis)
    m_mm = compute_metrics(ranks_mm)

    print("\n══ OOD evaluation results (gallery = all OOD textures) ══")
    print(f"  Stage 1 visual:     R@1={m_vis['R1']:5.1f}  R@5={m_vis['R5']:5.1f}  "
          f"R@10={m_vis['R10']:5.1f}  MRR={m_vis['MRR']:.3f}  RMR={m_vis['RMR']:.3f}")
    print(f"  Stage 2 multimodal: R@1={m_mm['R1']:5.1f}   R@5={m_mm['R5']:5.1f}  "
          f"R@10={m_mm['R10']:5.1f}  MRR={m_mm['MRR']:.3f}  RMR={m_mm['RMR']:.3f}")

    # Per-OOD-category breakdown
    per_cat_v = defaultdict(list)
    per_cat_m = defaultdict(list)
    for i, s in enumerate(samples):
        per_cat_v[s["ood_category"]].append(ranks_vis[i])
        per_cat_m[s["ood_category"]].append(ranks_mm[i])

    print(f"\n══ Per OOD-category breakdown ══")
    print(f"{'Category':<30s} {'n':>4s} | {'V R@1':>6s} {'V R@5':>6s} {'V RMR':>6s} | "
          f"{'M R@1':>6s} {'M R@5':>6s} {'M RMR':>6s}")
    per_cat = {}
    for cat in sorted(per_cat_v.keys()):
        rv = np.array(per_cat_v[cat])
        rm = np.array(per_cat_m[cat])
        mv = compute_metrics(rv)
        mm = compute_metrics(rm)
        per_cat[cat] = {"n": len(rv), "visual": mv, "multimodal": mm}
        print(f"{cat:<30s} {len(rv):>4d} | {mv['R1']:>6.1f} {mv['R5']:>6.1f} {mv['RMR']:>6.3f} | "
              f"{mm['R1']:>6.1f} {mm['R5']:>6.1f} {mm['RMR']:>6.3f}")

    # Save
    out = {
        "n_ood": N,
        "ood_gallery_size": N,
        "stage1_visual": m_vis,
        "stage2_multimodal": m_mm,
        "per_category": per_cat,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
