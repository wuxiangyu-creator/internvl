"""
Visualize retrieval cases: for each query, show the top-K retrieved textures.
Outputs a directory with side-by-side comparison images and a summary.

Usage:
    python internvl/train/eval_retrieval_cases.py \
        --model_name_or_path /home/ubuntu/alan_repository/models/InternVL2-1B \
        --ckpt /home/ubuntu/alan_repository/texture_material/contrastive_ckpt_unfrozen/best_contrastive_weights.pt \
        --val_jsonl /home/ubuntu/alan_repository/texture_material/dataset/val.jsonl \
        --image_root /home/ubuntu/alan_repository/texture_material/output \
        --output_dir /home/ubuntu/alan_repository/texture_material/retrieval_cases \
        --num_cases 20 --topk 5
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel


def build_transform(image_size=448):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_image(path, transform):
    img = Image.open(path).convert('RGB')
    return transform(img)


def make_comparison_image(query_path, gt_texture_path, retrieved_paths, retrieved_scores, gt_rank, cell_size=256):
    """Create a side-by-side comparison image:
       [Query | GT Texture | Retrieved#1 | Retrieved#2 | ... | Retrieved#K]
    """
    topk = len(retrieved_paths)
    n_cols = 2 + topk  # query + gt + topk
    padding = 4
    header_h = 30

    w = n_cols * (cell_size + padding) + padding
    h = cell_size + header_h + padding * 2

    canvas = Image.new('RGB', (w, h), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
        font_small = font

    def paste_image(img_path, col, label, border_color=None):
        x = padding + col * (cell_size + padding)
        y = header_h + padding

        img = Image.open(img_path).convert('RGB').resize((cell_size, cell_size))

        if border_color:
            border_w = 4
            bordered = Image.new('RGB', (cell_size, cell_size), border_color)
            bordered.paste(img.resize((cell_size - border_w * 2, cell_size - border_w * 2)), (border_w, border_w))
            canvas.paste(bordered, (x, y))
        else:
            canvas.paste(img, (x, y))

        draw.text((x + 4, 4), label, fill='white', font=font_small)

    # Column 0: Query (rendered image)
    paste_image(query_path, 0, "Query (fig.png)", border_color=(100, 100, 255))

    # Column 1: Ground truth texture
    paste_image(gt_texture_path, 1, f"GT (rank={gt_rank})", border_color=(0, 200, 0))

    # Columns 2+: Retrieved textures
    for k, (rpath, rscore) in enumerate(zip(retrieved_paths, retrieved_scores)):
        is_correct = (rpath == gt_texture_path)
        border = (0, 255, 0) if is_correct else None
        label = f"Top-{k+1} ({rscore:.3f})"
        if is_correct:
            label += " HIT"
        paste_image(rpath, 2 + k, label, border_color=border)

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--val_jsonl', required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--output_dir', default='/home/ubuntu/alan_repository/texture_material/retrieval_cases')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_cases', type=int, default=20, help='Number of cases to visualize')
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load model
    print('Loading model...')
    model = InternVLChatModel.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype, use_flash_attn=False, low_cpu_mem_usage=False,
    ).to(device)

    print(f'Loading contrastive weights from {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location=device)
    model.contrastive_head.load_state_dict(ckpt['contrastive_head'])
    model.contrastive_log_temp.data.copy_(ckpt['contrastive_log_temp'].data)
    if 'vision_model' in ckpt:
        model.vision_model.load_state_dict(ckpt['vision_model'])
        print('Loaded fine-tuned ViT weights')
    model.eval()

    # Load val data
    transform = build_transform(args.image_size)
    samples = []
    with open(args.val_jsonl) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f'Total val samples: {len(samples)}')

    # Encode all
    all_img_emb, all_tex_emb = [], []
    for i in tqdm(range(0, len(samples), args.batch_size), desc='Encoding'):
        batch = samples[i:i + args.batch_size]
        imgs = torch.stack([
            load_image(os.path.join(args.image_root, s['image']), transform) for s in batch
        ]).to(device, dtype=dtype)
        texs = torch.stack([
            load_image(os.path.join(args.image_root, s['texture']), transform) for s in batch
        ]).to(device, dtype=dtype)
        with torch.no_grad():
            all_img_emb.append(model.get_cls_embedding(imgs).cpu().float())
            all_tex_emb.append(model.get_cls_embedding(texs).cpu().float())

    all_img_emb = torch.cat(all_img_emb, dim=0)
    all_tex_emb = torch.cat(all_tex_emb, dim=0)
    N = all_img_emb.size(0)

    sim = torch.matmul(all_img_emb, all_tex_emb.T)

    # Compute ranks for all
    ranks = []
    for i in range(N):
        sorted_idx = sim[i].argsort(descending=True)
        rank = (sorted_idx == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    # Select cases: mix of hits and misses
    hits = [i for i, r in enumerate(ranks) if r == 1]
    misses = [i for i, r in enumerate(ranks) if r > 1]
    n_hit_cases = min(args.num_cases // 2, len(hits))
    n_miss_cases = min(args.num_cases - n_hit_cases, len(misses))

    import random
    random.seed(42)
    selected = random.sample(hits, n_hit_cases) + random.sample(misses, n_miss_cases)
    selected.sort(key=lambda i: ranks[i])

    # Generate visualizations
    os.makedirs(args.output_dir, exist_ok=True)

    summary_lines = []
    summary_lines.append(f"Retrieval Cases (gallery={N}, topk={args.topk})")
    summary_lines.append(f"{'='*80}")
    summary_lines.append("")

    for case_idx, qi in enumerate(selected):
        sample = samples[qi]
        rank = ranks[qi]

        query_path = os.path.join(args.image_root, sample['image'])
        gt_tex_path = os.path.join(args.image_root, sample['texture'])

        # Top-K retrieved
        topk_indices = sim[qi].argsort(descending=True)[:args.topk].tolist()
        topk_scores = [sim[qi][j].item() for j in topk_indices]
        topk_paths = [os.path.join(args.image_root, samples[j]['texture']) for j in topk_indices]

        # Save comparison image
        tag = "HIT" if rank == 1 else f"MISS_rank{rank}"
        out_name = f"case_{case_idx:03d}_{tag}_{sample['id']}.png"
        out_path = os.path.join(args.output_dir, out_name)

        comp_img = make_comparison_image(
            query_path, gt_tex_path, topk_paths, topk_scores, rank
        )
        comp_img.save(out_path)

        # Summary text
        summary_lines.append(f"Case {case_idx}: id={sample['id']}  rank={rank}  {'HIT' if rank==1 else 'MISS'}")
        summary_lines.append(f"  Query:      {query_path}")
        summary_lines.append(f"  GT Texture: {gt_tex_path}")
        for k, (tp, ts, ti) in enumerate(zip(topk_paths, topk_scores, topk_indices)):
            hit_mark = " <-- CORRECT" if ti == qi else ""
            summary_lines.append(f"  Top-{k+1}: sim={ts:.4f}  {tp}{hit_mark}")
        summary_lines.append(f"  Output:     {out_path}")
        summary_lines.append("")

    # Write summary
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    print(f"\nSaved {len(selected)} cases to: {args.output_dir}")
    print(f"Summary: {summary_path}")
    print("\n".join(summary_lines))


if __name__ == '__main__':
    main()
