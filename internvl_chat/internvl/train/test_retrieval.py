"""
Retrieval evaluation: given a rendered image, retrieve the correct texture
from the full val set. Reports Recall@1/5/10/20.

Loads only ViT + ContrastiveHead (no LLM) so it works on any InternVL size.
"""

import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from internvl.model.internvl_chat.modeling_intern_vit import InternVisionModel
from internvl.model.internvl_chat import InternVLChatConfig


class ContrastiveHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ViTContrastiveModel(nn.Module):
    def __init__(self, vision_model, vit_hidden_size, proj_dim=256):
        super().__init__()
        self.vision_model = vision_model
        self.contrastive_head = ContrastiveHead(in_dim=vit_hidden_size, proj_dim=proj_dim)
        self.contrastive_log_temp = nn.Parameter(torch.tensor(-2.6593))

    def get_cls_embedding(self, pixel_values):
        vit_out = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
        cls = vit_out[:, 0, :]
        return self.contrastive_head(cls)


def build_transform(image_size=448):
    return transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def encode_images(model, image_paths, transform, device, dtype, batch_size=64):
    all_embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = [transform(Image.open(p).convert('RGB')) for p in batch_paths]
        pixel_values = torch.stack(imgs).to(device, dtype=dtype)
        emb = model.get_cls_embedding(pixel_values)
        all_embs.append(emb.float().cpu())
    return torch.cat(all_embs, dim=0)


def load_model(model_path, ckpt_path, device, dtype):
    """Load ViT + ContrastiveHead only (no LLM)."""
    config = InternVLChatConfig.from_pretrained(model_path)
    vit_hidden_size = config.vision_config.hidden_size
    config.vision_config.use_flash_attn = False

    vision_model = InternVisionModel(config.vision_config)

    # Load ViT weights from base model
    from safetensors.torch import load_file as safe_load
    ckpt_files = sorted(glob.glob(os.path.join(model_path, '*.safetensors')))
    if not ckpt_files:
        ckpt_files = sorted(glob.glob(os.path.join(model_path, 'pytorch_model*.bin')))

    vit_state = {}
    for f in ckpt_files:
        if f.endswith('.safetensors'):
            state = safe_load(f)
        else:
            state = torch.load(f, map_location='cpu')
        for k, v in state.items():
            if k.startswith('vision_model.'):
                vit_state[k[len('vision_model.'):]] = v
        del state
    vision_model.load_state_dict(vit_state, strict=False)
    del vit_state

    model = ViTContrastiveModel(vision_model, vit_hidden_size).to(device, dtype=dtype)

    # Load fine-tuned checkpoint
    print(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)
    model.contrastive_head.load_state_dict(ckpt['contrastive_head'])
    model.contrastive_log_temp.data.copy_(ckpt['contrastive_log_temp'])
    if 'vision_model' in ckpt:
        model.vision_model.load_state_dict(ckpt['vision_model'])
        print('Loaded fine-tuned ViT weights')
    model.eval()

    temp = model.contrastive_log_temp.exp().item()
    print(f'Temperature: {temp:.4f}')
    vram_mb = torch.cuda.memory_allocated() / 1024 ** 2
    print(f'VRAM usage: {vram_mb:.0f} MB')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/home/ubuntu/alan_repository/models/InternVL2-1B')
    parser.add_argument('--ckpt', default='/home/ubuntu/alan_repository/texture_material/contrastive_ckpt_unfrozen/best_contrastive_weights.pt')
    parser.add_argument('--val_jsonl', default='/home/ubuntu/alan_repository/texture_material/dataset/val.jsonl')
    parser.add_argument('--image_root', default='/home/ubuntu/alan_repository/texture_material/output')
    args = parser.parse_args()

    device = torch.device('cuda')
    dtype = torch.bfloat16
    transform = build_transform(448)

    model = load_model(args.model_path, args.ckpt, device, dtype)

    # Load val data
    samples = []
    with open(args.val_jsonl) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f'Val samples: {len(samples)}')

    image_paths = [os.path.join(args.image_root, s['image']) for s in samples]
    texture_paths = [os.path.join(args.image_root, s['texture']) for s in samples]

    print('Encoding rendered images...')
    img_embs = encode_images(model, image_paths, transform, device, dtype)
    print('Encoding textures...')
    tex_embs = encode_images(model, texture_paths, transform, device, dtype)

    img_embs = F.normalize(img_embs, dim=-1)
    tex_embs = F.normalize(tex_embs, dim=-1)
    sim = img_embs @ tex_embs.T

    N = sim.size(0)
    ranks = sim.argsort(dim=1, descending=True)
    gt = torch.arange(N).unsqueeze(1)

    for k in [1, 5, 10, 20]:
        topk = ranks[:, :k]
        recall = (topk == gt).any(dim=1).float().mean().item() * 100
        print(f'Recall@{k}: {recall:.2f}%')

    print('\n--- Examples (image -> top-3 texture matches) ---')
    for i in range(min(10, N)):
        top3_idx = ranks[i, :3].tolist()
        mark = 'OK' if ranks[i, 0].item() == i else 'MISS'
        print(f'  [{mark}] image {samples[i]["id"]} -> '
              f'top3 textures: {[samples[j]["id"] for j in top3_idx]}')


if __name__ == '__main__':
    main()
