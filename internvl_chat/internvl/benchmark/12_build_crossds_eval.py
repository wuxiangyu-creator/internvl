"""
Build a cross-dataset evaluation set from AmbientCG materials.

Uses the same render-synthesis pipeline (crop + perspective + color jitter)
on each AmbientCG color map to produce (render, texture) pairs.

This is a *true* OOD benchmark: the textures are from an entirely different
texture library that was not used during training.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def random_perspective_warp(img, strength=0.12):
    w, h = img.size
    d = int(min(w, h) * strength)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (random.randint(0, d),     random.randint(0, d)),
        (w - random.randint(0, d), random.randint(0, d)),
        (w - random.randint(0, d), h - random.randint(0, d)),
        (random.randint(0, d),     h - random.randint(0, d)),
    ]
    matrix = []
    for (x, y), (X, Y) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.matrix(matrix, dtype=np.float64)
    B = []
    for (X, Y) in src:
        B += [X, Y]
    B = np.array(B, dtype=np.float64)
    coeffs = np.array(np.linalg.lstsq(A, B, rcond=None)[0]).flatten()
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def color_jitter(img):
    rgb = img.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(random.uniform(0.6, 1.4))
    rgb = ImageEnhance.Contrast(rgb).enhance(random.uniform(0.7, 1.3))
    rgb = ImageEnhance.Color(rgb).enhance(random.uniform(0.6, 1.5))
    if random.random() < 0.3:
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    elif random.random() < 0.3:
        rgb = rgb.filter(ImageFilter.SHARPEN)
    return rgb


def random_crop_resize(img, crop_min=0.6):
    w, h = img.size
    scale = random.uniform(crop_min, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    x0 = random.randint(0, w - nw)
    y0 = random.randint(0, h - nh)
    return img.crop((x0, y0, x0 + nw, y0 + nh)).resize((w, h), Image.LANCZOS)


def synthesise_fig(material_img, target_size=512):
    img = material_img.copy().resize((target_size, target_size), Image.LANCZOS)
    img = random_crop_resize(img, crop_min=0.65)
    img = random_perspective_warp(img, strength=random.uniform(0.05, 0.18))
    img = color_jitter(img)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="ambientcg_manifest.json")
    parser.add_argument("--source_dir", required=True,
                        help="Directory containing AmbientCG color maps")
    parser.add_argument("--output_dir", required=True,
                        help="Output dir for OOD sample folders")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--augments_per_texture", type=int, default=3)
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--id_start", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.manifest) as f:
        manifest = json.load(f)
    print(f"Manifest entries: {len(manifest)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    next_id = args.id_start
    skipped = 0

    for item in manifest:
        aid = item["asset_id"]
        category = item.get("category") or "ambientcg"
        tex_path = Path(args.source_dir) / item["texture_path"]
        if not tex_path.is_file():
            print(f"  [SKIP] {aid}: file missing {tex_path}")
            skipped += 1
            continue
        try:
            tex_img = Image.open(tex_path).convert("RGB")
        except Exception as e:
            print(f"  [SKIP] {aid}: open failed: {e}")
            skipped += 1
            continue

        tex_resized = tex_img.resize((args.target_size, args.target_size), Image.LANCZOS)
        for aug_i in range(args.augments_per_texture):
            sid = str(next_id)
            pair_dir = output_dir / sid
            pair_dir.mkdir(parents=True, exist_ok=True)
            try:
                fig = synthesise_fig(tex_img, target_size=args.target_size)
                fig.save(pair_dir / "fig.png")
                tex_resized.save(pair_dir / "texture.png")
                records.append({
                    "id": sid,
                    "image": f"{sid}/fig.png",
                    "texture": f"{sid}/texture.png",
                    "source_dataset": "AmbientCG",
                    "asset_id": aid,
                    "ambientcg_category": category,
                })
                next_id += 1
            except Exception as e:
                print(f"  [WARN] {aid} aug{aug_i}: {e}")
                skipped += 1

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nGenerated {len(records)} cross-dataset samples (skipped {skipped})")


if __name__ == "__main__":
    main()
