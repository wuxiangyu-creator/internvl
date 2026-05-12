"""
Encode all texture images with the fine-tuned Stage 1 ViT + ContrastiveHead.

Output: a single .npz file with
  - ids:        (N,) array of material id strings
  - embeddings: (N, 256) float32 array, L2-normalised
  - texture_paths: (N,) array of relative paths

Used downstream for:
  - K-means clustering to assign "visual category" IDs
  - Hard-negative mining (nearest-neighbour lookup)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel


def build_transform(size=448):
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--jsonl", required=True, help="JSONL with all (id, texture, ...) records")
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", required=True, help="Output .npz file")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Load samples
    samples = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Encoding {len(samples)} texture images...")

    # Build model
    print("Loading model...")
    cfg = InternVLChatConfig.from_pretrained(args.model_name_or_path)
    cfg.llm_config._attn_implementation = "eager"
    cfg.llm_config.attn_implementation = "eager"
    model = InternVLChatModel.from_pretrained(
        args.model_name_or_path, config=cfg,
        torch_dtype=dtype, attn_implementation="eager",
    )
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    model.contrastive_head.load_state_dict(ckpt["contrastive_head"])
    if "vision_model" in ckpt:
        model.vision_model.load_state_dict(ckpt["vision_model"])
    model = model.to(device).eval()
    print("Stage 1 weights loaded")

    transform = build_transform(448)

    # Encode in batches
    all_embs = []
    t0 = time.time()
    for i in tqdm(range(0, len(samples), args.batch_size), desc="Encoding"):
        batch = samples[i : i + args.batch_size]
        imgs = torch.stack([
            transform(Image.open(os.path.join(args.image_root, s["texture"])).convert("RGB"))
            for s in batch
        ]).to(device, dtype=dtype)
        with torch.no_grad():
            embs = model.get_cls_embedding(imgs).cpu().float().numpy()
        all_embs.append(embs)

    all_embs = np.concatenate(all_embs, axis=0)
    ids = np.array([s["id"] for s in samples])
    texture_paths = np.array([s["texture"] for s in samples])

    print(f"Encoded {len(all_embs)} textures in {time.time()-t0:.1f}s")
    print(f"Embedding shape: {all_embs.shape}, dtype: {all_embs.dtype}")

    np.savez_compressed(args.output, ids=ids, embeddings=all_embs, texture_paths=texture_paths)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
