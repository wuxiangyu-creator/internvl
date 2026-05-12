"""
Build an OOD (Out-Of-Distribution) evaluation set from texture directories
that were NOT used during training.

Source directories (all from /home/ubuntu/alan_repository/SceneObject/):
  UGC_Poster, SceneInteraction, UGC_terrain, UrbanInteractiveObjects,
  Smallthing, UGC_LOGO, Rock, Traffic

For each texture file we:
  1. Generate `augments_per_texture` augmented "render" images (same pipeline
     as generate_pairs.py: crop + perspective warp + color jitter)
  2. Save (fig.png, texture.png) pair in a new sample directory
  3. Emit a JSONL record { id, image, texture, ood_category }

Output:
  - {output_dir}/{ood_id}/fig.png        — augmented query
  - {output_dir}/{ood_id}/texture.png    — clean texture (reference)
  - {output_jsonl}                       — JSONL with all OOD samples
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Force this to be a self-contained script (don't depend on generate_pairs.py imports)


# ── augmentation pipeline (copied from generate_pairs.py for stand-alone use) ─
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


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", default="/home/ubuntu/alan_repository/SceneObject")
    parser.add_argument("--ood_dirs", nargs="+", default=[
        "UGC_Poster", "SceneInteraction", "UGC_terrain",
        "UrbanInteractiveObjects", "Smallthing", "UGC_LOGO", "Rock", "Traffic",
    ])
    parser.add_argument("--output_dir", required=True,
                        help="Output dir for OOD sample folders")
    parser.add_argument("--output_jsonl", required=True,
                        help="Output JSONL listing all OOD samples")
    parser.add_argument("--augments_per_texture", type=int, default=3)
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id_start", type=int, default=100000,
                        help="Start ID (must not collide with in-domain IDs)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all textures
    all_textures = []
    for d in args.ood_dirs:
        src = Path(args.source_root) / d
        if not src.is_dir():
            print(f"  [SKIP] {src} does not exist")
            continue
        for p in src.rglob("*"):
            if p.suffix.lower() in {".png", ".tga", ".jpg", ".jpeg"} and p.is_file():
                all_textures.append((d, p))
    print(f"Found {len(all_textures)} OOD textures across {len(args.ood_dirs)} directories")

    # Generate
    records = []
    next_id = args.id_start
    skipped = 0

    for ood_dir, tex_path in all_textures:
        try:
            tex_img = Image.open(tex_path).convert("RGBA")
        except Exception as e:
            print(f"  [SKIP] Cannot open {tex_path}: {e}")
            skipped += 1
            continue

        # Save clean texture (resized to target_size)
        tex_rgb = tex_img.convert("RGB").resize((args.target_size, args.target_size), Image.LANCZOS)

        # Each augmented sample gets its own folder
        for aug_i in range(args.augments_per_texture):
            sid = str(next_id)
            pair_dir = output_dir / sid
            pair_dir.mkdir(parents=True, exist_ok=True)

            try:
                fig = synthesise_fig(tex_img, target_size=args.target_size)
                fig.convert("RGB").save(pair_dir / "fig.png")
                tex_rgb.save(pair_dir / "texture.png")

                records.append({
                    "id": sid,
                    "image": f"{sid}/fig.png",
                    "texture": f"{sid}/texture.png",
                    "ood_category": ood_dir,
                    "source_texture": str(tex_path.relative_to(args.source_root)),
                })
                next_id += 1
            except Exception as e:
                print(f"  [WARN] aug {aug_i} for {tex_path.name}: {e}")
                next_id += 1
                skipped += 1

    # Save JSONL
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nGenerated {len(records)} OOD samples (skipped {skipped})")
    print(f"  Output dir:   {output_dir}")
    print(f"  Output JSONL: {args.output_jsonl}")

    # Per-category breakdown
    from collections import Counter
    cat_counter = Counter(r["ood_category"] for r in records)
    print(f"\nPer-category breakdown:")
    for cat, n in cat_counter.most_common():
        print(f"  {cat:>30s}: {n}")


if __name__ == "__main__":
    main()
