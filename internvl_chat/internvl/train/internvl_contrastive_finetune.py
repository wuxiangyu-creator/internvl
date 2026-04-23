"""
Contrastive fine-tuning for InternVL: image-texture alignment.

Only loads ViT + ContrastiveHead (skips LLM entirely to save VRAM).
This allows training with larger InternVL models (e.g. 8B) on a single GPU.

At deployment time, load the full InternVLChatModel and apply the fine-tuned
ViT + contrastive weights on top.

Usage:
    python internvl/train/internvl_contrastive_finetune.py \
        --model_name_or_path /path/to/InternVL2-8B \
        --train_jsonl /path/to/train.jsonl \
        --val_jsonl   /path/to/val.jsonl \
        --image_root  /path/to/images \
        --output_dir  ./output/contrastive \
        --num_train_epochs 5 \
        --per_device_train_batch_size 64 \
        --learning_rate 1e-4 \
        --freeze_vit False
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from internvl.model.internvl_chat.modeling_intern_vit import InternVisionModel
from internvl.model.internvl_chat import InternVLChatConfig

from dataset_contrastive import ContrastivePairDataset


class ContrastiveHead(nn.Module):
    """Projection head for image-texture contrastive learning."""

    def __init__(self, in_dim: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ViTContrastiveModel(nn.Module):
    """Lightweight wrapper: ViT + ContrastiveHead (no LLM)."""

    def __init__(self, vision_model, vit_hidden_size, proj_dim=256):
        super().__init__()
        self.vision_model = vision_model
        self.contrastive_head = ContrastiveHead(in_dim=vit_hidden_size, proj_dim=proj_dim)
        self.contrastive_log_temp = nn.Parameter(torch.tensor(-2.6593))  # log(0.07)

    def get_cls_embedding(self, pixel_values):
        vit_out = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state  # (B, N+1, C)
        cls = vit_out[:, 0, :]  # CLS token
        return self.contrastive_head(cls)

    def contrastive_forward(self, pixel_values, texture_pixel_values):
        img_emb = self.get_cls_embedding(pixel_values)
        tex_emb = self.get_cls_embedding(texture_pixel_values)

        log_temp = self.contrastive_log_temp.clamp(-4.6, 2.3)
        temp = log_temp.exp()
        logits = torch.matmul(img_emb, tex_emb.T) / temp
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--train_jsonl', required=True)
    parser.add_argument('--val_jsonl', required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--per_device_train_batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Accumulate gradients to simulate larger batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--freeze_vit', type=lambda x: x.lower() != 'false', default=True)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--bf16', action='store_true', default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32

    # --- Only load ViT from the InternVL config (skip LLM) ---
    print(f'Loading ViT from {args.model_name_or_path} (skipping LLM to save VRAM)')
    config = InternVLChatConfig.from_pretrained(args.model_name_or_path)
    vit_hidden_size = config.vision_config.hidden_size
    print(f'ViT hidden_size: {vit_hidden_size}')

    # Build ViT
    config.vision_config.use_flash_attn = False
    vision_model = InternVisionModel(config.vision_config)

    # Load only ViT weights from the full checkpoint
    from safetensors.torch import load_file as safe_load
    import glob

    ckpt_files = sorted(glob.glob(os.path.join(args.model_name_or_path, '*.safetensors')))
    if not ckpt_files:
        ckpt_files = sorted(glob.glob(os.path.join(args.model_name_or_path, 'pytorch_model*.bin')))

    vit_state = {}
    for f in ckpt_files:
        if f.endswith('.safetensors'):
            state = safe_load(f)
        else:
            state = torch.load(f, map_location='cpu')
        for k, v in state.items():
            if k.startswith('vision_model.'):
                new_key = k[len('vision_model.'):]
                vit_state[new_key] = v
        del state

    info = vision_model.load_state_dict(vit_state, strict=False)
    print(f'ViT weights loaded: {len(vit_state)} tensors')
    if info.missing_keys:
        print(f'  Missing (expected for pos_embed etc): {info.missing_keys[:5]}...')
    del vit_state

    # Build lightweight model
    model = ViTContrastiveModel(vision_model, vit_hidden_size).to(device, dtype=dtype)

    # Explicitly set log_temp
    with torch.no_grad():
        model.contrastive_log_temp.fill_(-2.6593)

    # Freeze/unfreeze ViT
    if args.freeze_vit:
        for param in model.vision_model.parameters():
            param.requires_grad = False
        print('ViT frozen — training contrastive_head + contrastive_temp only')
    else:
        print('ViT unfrozen — training ViT + contrastive_head + contrastive_temp')

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable params: {trainable:,} / {total:,}')

    vram_mb = torch.cuda.memory_allocated() / 1024 ** 2
    print(f'VRAM after model load: {vram_mb:.0f} MB')

    # Datasets
    train_dataset = ContrastivePairDataset(args.train_jsonl, args.image_root, args.image_size)
    val_dataset = ContrastivePairDataset(args.val_jsonl, args.image_root, args.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.per_device_train_batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, weight_decay=0.01
    )
    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.num_train_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    print(f'Effective batch size: {effective_batch}, total optimizer steps: {total_steps}')

    global_step = 0
    best_val_loss = float('inf')
    temp_warmup_epochs = 2

    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0.0

        model.contrastive_log_temp.requires_grad = (epoch >= temp_warmup_epochs)
        if epoch == temp_warmup_epochs:
            print(f'Epoch {epoch+1}: unfreezing contrastive_log_temp')

        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values'].to(device, dtype=dtype)
            texture_pixel_values = batch['texture_pixel_values'].to(device, dtype=dtype)

            loss = model.contrastive_forward(pixel_values, texture_pixel_values)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item() * args.gradient_accumulation_steps

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    model.contrastive_log_temp.clamp_(-4.6, 2.3)

                global_step += 1

                if global_step % 10 == 0:
                    tau = model.contrastive_log_temp.exp().item()
                    print(f'Epoch {epoch+1} step {global_step} | '
                          f'loss={loss.item() * args.gradient_accumulation_steps:.4f} temp={tau:.4f}')

                if global_step % args.save_steps == 0:
                    ckpt = os.path.join(args.output_dir, f'step-{global_step}')
                    os.makedirs(ckpt, exist_ok=True)
                    save_dict = {
                        'contrastive_head': model.contrastive_head.state_dict(),
                        'contrastive_log_temp': model.contrastive_log_temp,
                    }
                    if not args.freeze_vit:
                        save_dict['vision_model'] = model.vision_model.state_dict()
                    torch.save(save_dict, os.path.join(ckpt, 'contrastive_weights.pt'))
                    print(f'Saved checkpoint: {ckpt}')

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device, dtype=dtype)
                texture_pixel_values = batch['texture_pixel_values'].to(device, dtype=dtype)
                loss = model.contrastive_forward(pixel_values, texture_pixel_values)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f'=== Epoch {epoch+1}/{args.num_train_epochs} | '
              f'train_loss={avg_loss:.4f} val_loss={val_loss:.4f} ===')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, 'best_contrastive_weights.pt')
            save_dict = {
                'contrastive_head': model.contrastive_head.state_dict(),
                'contrastive_log_temp': model.contrastive_log_temp,
            }
            if not args.freeze_vit:
                save_dict['vision_model'] = model.vision_model.state_dict()
            torch.save(save_dict, best_path)
            print(f'Best model saved: {best_path}')

    print('Training complete.')


if __name__ == '__main__':
    main()
