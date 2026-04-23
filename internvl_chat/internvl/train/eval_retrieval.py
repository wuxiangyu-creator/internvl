"""
Evaluate contrastive retrieval: given a rendered image, retrieve the matching texture.

Metrics:
  - Recall@1, @5, @10
  - Mean Reciprocal Rank (MRR)

Usage:
    python internvl/train/eval_retrieval.py \
        --model_name_or_path /home/ubuntu/alan_repository/models/InternVL2-1B \
        --ckpt /home/ubuntu/alan_repository/texture_material/contrastive_ckpt/best_contrastive_weights.pt \
        --val_jsonl /home/ubuntu/alan_repository/texture_material/dataset/val.jsonl \
        --image_root /home/ubuntu/alan_repository/texture_material/output \
        --batch_size 64
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--ckpt', required=True, help='Path to best_contrastive_weights.pt')
    parser.add_argument('--val_jsonl', required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_samples', type=int, default=0, help='0 = use all val samples')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load model
    print('Loading model...')
    model = InternVLChatModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        use_flash_attn=False,
        low_cpu_mem_usage=False,
    ).to(device)

    # Load contrastive weights (and optionally fine-tuned ViT)
    print(f'Loading contrastive weights from {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location=device)
    model.contrastive_head.load_state_dict(ckpt['contrastive_head'])
    model.contrastive_log_temp.data.copy_(ckpt['contrastive_log_temp'].data)
    if 'vision_model' in ckpt:
        model.vision_model.load_state_dict(ckpt['vision_model'])
        print('Loaded fine-tuned ViT weights')
    else:
        print('Using original ViT weights (head-only checkpoint)')
    model.eval()

    temp = model.contrastive_log_temp.exp().item()
    print(f'Temperature: {temp:.4f}')

    # Load val data
    transform = build_transform(args.image_size)
    samples = []
    with open(args.val_jsonl) as f:
        for line in f:
            item = json.loads(line)
            samples.append(item)

    if args.num_samples > 0:
        samples = samples[:args.num_samples]
    print(f'Evaluating on {len(samples)} samples')

    # Encode all images and textures
    all_img_emb = []
    all_tex_emb = []

    for i in tqdm(range(0, len(samples), args.batch_size), desc='Encoding'):
        batch = samples[i:i + args.batch_size]

        imgs = torch.stack([
            load_image(os.path.join(args.image_root, s['image']), transform)
            for s in batch
        ]).to(device, dtype=dtype)

        texs = torch.stack([
            load_image(os.path.join(args.image_root, s['texture']), transform)
            for s in batch
        ]).to(device, dtype=dtype)

        with torch.no_grad():
            img_emb = model.get_cls_embedding(imgs)
            tex_emb = model.get_cls_embedding(texs)

        all_img_emb.append(img_emb.cpu().float())
        all_tex_emb.append(tex_emb.cpu().float())

    all_img_emb = torch.cat(all_img_emb, dim=0)  # (N, D)
    all_tex_emb = torch.cat(all_tex_emb, dim=0)  # (N, D)

    print(f'Embeddings: img={all_img_emb.shape}, tex={all_tex_emb.shape}')

    # Compute similarity matrix: (N, N)
    sim = torch.matmul(all_img_emb, all_tex_emb.T)  # cosine sim (already normalized)

    N = sim.size(0)
    # For each image, rank textures by similarity
    ranks = []
    for i in range(N):
        row = sim[i]
        sorted_indices = row.argsort(descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed
        ranks.append(rank)

    ranks = torch.tensor(ranks, dtype=torch.float)

    r1 = (ranks <= 1).float().mean().item() * 100
    r5 = (ranks <= 5).float().mean().item() * 100
    r10 = (ranks <= 10).float().mean().item() * 100
    mrr = (1.0 / ranks).mean().item()
    median_rank = ranks.median().item()

    print(f'\n{"="*50}')
    print(f'  Retrieval Results (gallery size = {N})')
    print(f'{"="*50}')
    print(f'  Recall@1:    {r1:.2f}%')
    print(f'  Recall@5:    {r5:.2f}%')
    print(f'  Recall@10:   {r10:.2f}%')
    print(f'  MRR:         {mrr:.4f}')
    print(f'  Median Rank: {median_rank:.0f}')
    print(f'{"="*50}')

    # Show some examples
    print(f'\nTop-5 examples (first 10 queries):')
    for i in range(min(10, N)):
        row = sim[i]
        top5 = row.argsort(descending=True)[:5].tolist()
        correct = i
        marker = '✓' if top5[0] == correct else '✗'
        print(f'  Query {samples[i]["id"]}: top5={top5}, gt={correct} {marker}')


if __name__ == '__main__':
    main()
