"""
Evaluate ReTR (Stage 1 + Stage 2) plus all modern baselines on the
cross-dataset OOD benchmark (AmbientCG materials).

Reports retrieval metrics on a per-method basis.
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


def compute_metrics(ranks):
    ranks = np.asarray(ranks)
    r1 = float((ranks <= 1).mean()) * 100
    r5 = float((ranks <= 5).mean()) * 100
    r10 = float((ranks <= 10).mean()) * 100
    mrr = float((1.0 / ranks).mean())
    medr = float(np.median(ranks))
    rmr = (r1 / 100 + r5 / 100 + mrr) / 3.0
    return dict(R1=r1, R5=r5, R10=r10, MRR=mrr, MedR=medr, RMR=rmr)


def retrieval_ranks(qe, ge):
    sim = qe @ ge.T
    N = len(qe)
    return np.array([np.where(np.argsort(-sim[i]) == i)[0][0] + 1 for i in range(N)])


def build_retr_transform(size=448):
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def encode_retr_visual(model, paths, root, transform, device, dtype, batch_size, desc):
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
def encode_retr_multimodal(model, tokenizer, num_image_token, samples, root,
                            transform, device, dtype, batch_size, desc):
    img_ctx = "<IMG_CONTEXT>"
    out = []
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
        out.append(q.cpu().float().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def encode_hf(paths, root, model, processor, device, batch_size, desc):
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc):
        batch = paths[i : i + batch_size]
        imgs = [Image.open(os.path.join(root, p)).convert("RGB") for p in batch]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        out = model.get_image_features(**inputs) if hasattr(model, "get_image_features") \
              else model(**inputs).last_hidden_state[:, 0, :]
        embs.append(F.normalize(out.float(), dim=-1).cpu().numpy())
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def encode_open_clip(paths, root, model, preprocess, device, batch_size, desc):
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc):
        batch = paths[i : i + batch_size]
        imgs = torch.stack([
            preprocess(Image.open(os.path.join(root, p)).convert("RGB"))
            for p in batch
        ]).to(device)
        out = model.encode_image(imgs)
        embs.append(F.normalize(out.float(), dim=-1).cpu().numpy())
    return np.concatenate(embs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--stage2_ckpt", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda")
    samples = []
    with open(args.eval_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    img_paths = [s["image"] for s in samples]
    tex_paths = [s["texture"] for s in samples]
    print(f"Cross-dataset eval samples: {len(samples)}")

    results = {}

    # ── ReTR Stage 1 + Stage 2 ────────────────────────────────────────
    print("\n══ Loading ReTR (Stage 1 + Stage 2) ══")
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
    ckpt2 = torch.load(args.stage2_ckpt, map_location="cpu")
    model.retrieval_head.load_state_dict(ckpt2["retrieval_head"])
    model.mlp1.load_state_dict(ckpt2["mlp1"])
    if "contrastive_log_temp" in ckpt2:
        model.contrastive_log_temp.data.copy_(ckpt2["contrastive_log_temp"].data)
    if "lora_weights" in ckpt2:
        lora_state = ckpt2["lora_weights"]
        model.wrap_llm_lora(r=64, lora_alpha=128)
        ms = model.language_model.state_dict()
        for k, v in lora_state.items():
            if k in ms:
                ms[k] = v
        model.language_model.load_state_dict(ms)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    num_image_token = model.num_image_token
    transform = build_retr_transform(448)

    print("\n══ ReTR Stage 1 visual ══")
    img_emb = encode_retr_visual(model, img_paths, args.image_root, transform,
                                  device, dtype, args.batch_size, "retr-img")
    tex_emb = encode_retr_visual(model, tex_paths, args.image_root, transform,
                                  device, dtype, args.batch_size, "retr-tex")
    ranks = retrieval_ranks(img_emb, tex_emb)
    results["ReTR Stage 1 (visual)"] = compute_metrics(ranks)

    print("\n══ ReTR Stage 2 multimodal ══")
    mm_emb = encode_retr_multimodal(model, tokenizer, num_image_token, samples,
                                      args.image_root, transform, device, dtype,
                                      args.batch_size, "retr-mm")
    ranks_mm = retrieval_ranks(mm_emb, tex_emb)
    results["ReTR Stage 2 (multimodal)"] = compute_metrics(ranks_mm)

    del model
    torch.cuda.empty_cache()

    # ── Modern Baselines ────────────────────────────────────────────
    from transformers import AutoModel, AutoImageProcessor

    print("\n══ SigLIP-SO400M-384 ══")
    siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384",
                                       torch_dtype=torch.float32).to(device).eval()
    siglip_proc = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    s_img = encode_hf(img_paths, args.image_root, siglip, siglip_proc, device, args.batch_size, "siglip-img")
    s_tex = encode_hf(tex_paths, args.image_root, siglip, siglip_proc, device, args.batch_size, "siglip-tex")
    results["SigLIP-SO400M-384"] = compute_metrics(retrieval_ranks(s_img, s_tex))
    del siglip
    torch.cuda.empty_cache()

    print("\n══ DINOv2-large ══")
    dino = AutoModel.from_pretrained("facebook/dinov2-large",
                                     torch_dtype=torch.float32).to(device).eval()
    dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    d_img = encode_hf(img_paths, args.image_root, dino, dino_proc, device, args.batch_size, "dino-img")
    d_tex = encode_hf(tex_paths, args.image_root, dino, dino_proc, device, args.batch_size, "dino-tex")
    results["DINOv2-large"] = compute_metrics(retrieval_ranks(d_img, d_tex))
    del dino
    torch.cuda.empty_cache()

    print("\n══ OpenCLIP ViT-L/14 LAION-2B ══")
    import open_clip
    oc, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
    oc = oc.to(device).eval()
    o_img = encode_open_clip(img_paths, args.image_root, oc, pre, device, args.batch_size, "oc-img")
    o_tex = encode_open_clip(tex_paths, args.image_root, oc, pre, device, args.batch_size, "oc-tex")
    results["OpenCLIP-ViT-L/14-LAION2B"] = compute_metrics(retrieval_ranks(o_img, o_tex))
    del oc
    torch.cuda.empty_cache()

    print("\n══ CLIP ViT-L/14 (original) ══")
    clip_l = AutoModel.from_pretrained("openai/clip-vit-large-patch14",
                                        torch_dtype=torch.float32).to(device).eval()
    clip_proc = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    c_img = encode_hf(img_paths, args.image_root, clip_l, clip_proc, device, args.batch_size, "clip-img")
    c_tex = encode_hf(tex_paths, args.image_root, clip_l, clip_proc, device, args.batch_size, "clip-tex")
    results["CLIP-ViT-L/14"] = compute_metrics(retrieval_ranks(c_img, c_tex))

    # Summary
    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║       Cross-Dataset Generalization (AmbientCG, N=450)             ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"{'Method':<32s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} {'MRR':>6s} {'RMR':>6s}")
    print("─" * 70)
    for name, m in results.items():
        print(f"{name:<32s} {m['R1']:>6.1f} {m['R5']:>6.1f} {m['R10']:>6.1f} "
              f"{m['MRR']:>6.3f} {m['RMR']:>6.3f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
