"""
Modern VLM baseline evaluation on ReTR-Bench.

Adds SigLIP, DINOv2, and OpenCLIP results to extend Table 2 beyond
the 2021-vintage CLIP baselines.

All models are used \emph{zero-shot}: we encode both fig.png (query) and
texture.png (gallery) with the same image encoder, then do cosine retrieval.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def compute_metrics(ranks):
    ranks = np.asarray(ranks)
    r1 = float((ranks <= 1).mean()) * 100
    r5 = float((ranks <= 5).mean()) * 100
    r10 = float((ranks <= 10).mean()) * 100
    mrr = float((1.0 / ranks).mean())
    medr = float(np.median(ranks))
    ideal_dcg = 1.0 / math.log2(2)
    ndcg10 = float(np.mean([1.0 / math.log2(r + 1) if r <= 10 else 0.0
                            for r in ranks]) / ideal_dcg)
    rmr = (r1 / 100 + r5 / 100 + mrr) / 3.0
    return dict(R1=r1, R5=r5, R10=r10, MRR=mrr, MedR=medr, nDCG10=ndcg10, RMR=rmr)


@torch.no_grad()
def encode_with_processor(paths, root, model, processor, device, batch_size, desc):
    """Encode using HuggingFace transformers API (SigLIP, DINOv2)."""
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc):
        batch = paths[i : i + batch_size]
        imgs = [Image.open(os.path.join(root, p)).convert("RGB") for p in batch]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        out = model.get_image_features(**inputs) if hasattr(model, "get_image_features") \
              else model(**inputs).last_hidden_state[:, 0, :]
        out = F.normalize(out.float(), dim=-1)
        embs.append(out.cpu().numpy())
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def encode_open_clip(paths, root, model, preprocess, device, batch_size, desc):
    """Encode using open_clip API."""
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc):
        batch = paths[i : i + batch_size]
        imgs = torch.stack([
            preprocess(Image.open(os.path.join(root, p)).convert("RGB"))
            for p in batch
        ]).to(device)
        out = model.encode_image(imgs)
        out = F.normalize(out.float(), dim=-1)
        embs.append(out.cpu().numpy())
    return np.concatenate(embs, axis=0)


def retrieval_metrics(img_emb, tex_emb):
    """1:1 retrieval — GT for query i is at index i in gallery."""
    sim = img_emb @ tex_emb.T
    N = len(img_emb)
    ranks = np.array([
        np.where(np.argsort(-sim[i]) == i)[0][0] + 1 for i in range(N)
    ])
    return compute_metrics(ranks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--models", nargs="+",
                        default=["siglip", "dinov2_base", "dinov2_large", "openclip_vit_l"])
    args = parser.parse_args()

    device = torch.device("cuda")

    samples = []
    with open(args.eval_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Eval samples: {len(samples)}")
    img_paths = [s["image"] for s in samples]
    tex_paths = [s["texture"] for s in samples]

    results = {}

    # ── SigLIP ────────────────────────────────────────────────────────
    if "siglip" in args.models:
        print("\n══ SigLIP-SO400M-patch14-384 ══")
        from transformers import AutoModel, AutoImageProcessor
        siglip_id = "google/siglip-so400m-patch14-384"
        model = AutoModel.from_pretrained(siglip_id, torch_dtype=torch.float32).to(device).eval()
        # Use image processor only (avoid tokenizer / sentencepiece issues)
        processor = AutoImageProcessor.from_pretrained(siglip_id)
        img_emb = encode_with_processor(img_paths, args.image_root, model, processor,
                                         device, args.batch_size, "siglip-img")
        tex_emb = encode_with_processor(tex_paths, args.image_root, model, processor,
                                         device, args.batch_size, "siglip-tex")
        results["SigLIP-SO400M-384"] = retrieval_metrics(img_emb, tex_emb)
        del model
        torch.cuda.empty_cache()
        m = results["SigLIP-SO400M-384"]
        print(f"  R@1={m['R1']:5.1f}  R@5={m['R5']:5.1f}  R@10={m['R10']:5.1f}  "
              f"MRR={m['MRR']:.3f}  RMR={m['RMR']:.3f}")

    # ── DINOv2 base ──────────────────────────────────────────────────
    if "dinov2_base" in args.models:
        print("\n══ DINOv2-base (768-d) ══")
        from transformers import AutoModel, AutoImageProcessor
        model = AutoModel.from_pretrained("facebook/dinov2-base", torch_dtype=torch.float32).to(device).eval()
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        img_emb = encode_with_processor(img_paths, args.image_root, model, processor,
                                         device, args.batch_size, "dinov2b-img")
        tex_emb = encode_with_processor(tex_paths, args.image_root, model, processor,
                                         device, args.batch_size, "dinov2b-tex")
        results["DINOv2-base"] = retrieval_metrics(img_emb, tex_emb)
        del model
        torch.cuda.empty_cache()
        m = results["DINOv2-base"]
        print(f"  R@1={m['R1']:5.1f}  R@5={m['R5']:5.1f}  R@10={m['R10']:5.1f}  "
              f"MRR={m['MRR']:.3f}  RMR={m['RMR']:.3f}")

    # ── DINOv2 large ─────────────────────────────────────────────────
    if "dinov2_large" in args.models:
        print("\n══ DINOv2-large (1024-d) ══")
        from transformers import AutoModel, AutoImageProcessor
        model = AutoModel.from_pretrained("facebook/dinov2-large", torch_dtype=torch.float32).to(device).eval()
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        img_emb = encode_with_processor(img_paths, args.image_root, model, processor,
                                         device, args.batch_size, "dinov2l-img")
        tex_emb = encode_with_processor(tex_paths, args.image_root, model, processor,
                                         device, args.batch_size, "dinov2l-tex")
        results["DINOv2-large"] = retrieval_metrics(img_emb, tex_emb)
        del model
        torch.cuda.empty_cache()
        m = results["DINOv2-large"]
        print(f"  R@1={m['R1']:5.1f}  R@5={m['R5']:5.1f}  R@10={m['R10']:5.1f}  "
              f"MRR={m['MRR']:.3f}  RMR={m['RMR']:.3f}")

    # ── OpenCLIP ViT-L/14 (LAION-2B) ─────────────────────────────────
    if "openclip_vit_l" in args.models:
        print("\n══ OpenCLIP ViT-L/14 (LAION-2B) ══")
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k"
        )
        model = model.to(device).eval()
        img_emb = encode_open_clip(img_paths, args.image_root, model, preprocess,
                                    device, args.batch_size, "openclip-img")
        tex_emb = encode_open_clip(tex_paths, args.image_root, model, preprocess,
                                    device, args.batch_size, "openclip-tex")
        results["OpenCLIP-ViT-L/14-LAION2B"] = retrieval_metrics(img_emb, tex_emb)
        del model
        torch.cuda.empty_cache()
        m = results["OpenCLIP-ViT-L/14-LAION2B"]
        print(f"  R@1={m['R1']:5.1f}  R@5={m['R5']:5.1f}  R@10={m['R10']:5.1f}  "
              f"MRR={m['MRR']:.3f}  RMR={m['RMR']:.3f}")

    # Summary table
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║              Modern VLM baselines on ReTR-Bench                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
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
