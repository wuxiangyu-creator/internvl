"""
Multimodal contrastive fine-tuning for InternVL2:
  Path A — ViT + ContrastiveHead    (image ↔ texture, visual only)
  Path B — ViT + LLM + RetrievalHead (image+text ↔ texture, multimodal)

  L_total = α · L_visual + β · L_multimodal

LLM uses LoRA to keep VRAM manageable on a single GPU.

Usage:
    python internvl_multimodal_contrastive.py \
        --model_name_or_path /path/to/InternVL2-1B \
        --contrastive_ckpt /path/to/best_contrastive_weights.pt \
        --train_jsonl /path/to/train_with_desc.jsonl \
        --val_jsonl /path/to/val_with_desc.jsonl \
        --image_root /path/to/images \
        --output_dir ./output/multimodal_contrastive \
        --num_train_epochs 5 \
        --per_device_train_batch_size 8 \
        --learning_rate 5e-5 \
        --lora_r 64 \
        --alpha 1.0 --beta 1.0
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel

from dataset_multimodal_contrastive import MultimodalContrastiveDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--contrastive_ckpt', default=None,
                        help='Path to stage-1 contrastive weights (ViT + ContrastiveHead)')
    parser.add_argument('--train_jsonl', required=True)
    parser.add_argument('--val_jsonl', required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for L_visual')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight for L_multimodal')
    parser.add_argument('--warmup_steps', type=int, default=50)
    parser.add_argument('--save_steps', type=int, default=200)
    parser.add_argument('--max_text_len', type=int, default=80)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda')
    dtype = torch.bfloat16

    # ── 1. Load full InternVL2 model ────────────────────────────────────────
    print(f'Loading InternVL2 from {args.model_name_or_path}')
    cfg = InternVLChatConfig.from_pretrained(args.model_name_or_path)
    cfg.llm_config._attn_implementation = "eager"
    cfg.llm_config.attn_implementation = "eager"

    model = InternVLChatModel.from_pretrained(
        args.model_name_or_path, config=cfg,
        torch_dtype=dtype, attn_implementation="eager",
    )

    # ── 2. Load stage-1 contrastive weights (ViT + ContrastiveHead) ─────────
    if args.contrastive_ckpt and os.path.exists(args.contrastive_ckpt):
        print(f'Loading stage-1 contrastive weights from {args.contrastive_ckpt}')
        ckpt = torch.load(args.contrastive_ckpt, map_location='cpu')
        model.contrastive_head.load_state_dict(ckpt['contrastive_head'])
        if 'vision_model' in ckpt:
            model.vision_model.load_state_dict(ckpt['vision_model'])
            print('  Loaded fine-tuned ViT weights from stage 1')
        if 'contrastive_log_temp' in ckpt:
            model.contrastive_log_temp.data.copy_(ckpt['contrastive_log_temp'].data)
        del ckpt

    # ── 3. Freeze / LoRA setup ──────────────────────────────────────────────
    # Freeze ViT (already fine-tuned in stage 1)
    for param in model.vision_model.parameters():
        param.requires_grad = False
    print('ViT frozen (using stage-1 weights)')

    # Freeze ContrastiveHead (already trained, shared embedding space anchor)
    for param in model.contrastive_head.parameters():
        param.requires_grad = False
    print('ContrastiveHead frozen (anchor for shared space)')

    # MLP projector: unfreeze (needs to adapt for LLM path)
    for param in model.mlp1.parameters():
        param.requires_grad = True
    print('MLP projector unfrozen')

    # LLM: apply LoRA
    model.wrap_llm_lora(r=args.lora_r, lora_alpha=args.lora_alpha)
    print(f'LLM LoRA applied (r={args.lora_r}, alpha={args.lora_alpha})')

    # RetrievalHead: train from scratch
    for param in model.retrieval_head.parameters():
        param.requires_grad = True
    print('RetrievalHead trainable')

    # Temperature: trainable
    model.contrastive_log_temp.requires_grad = True

    model = model.to(device)

    # Print param summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)')
    vram_mb = torch.cuda.memory_allocated() / 1024 ** 2
    print(f'VRAM after model load: {vram_mb:.0f} MB')

    # ── 4. Tokenizer + Dataset ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Set img_context_token_id on model
    img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    model.img_context_token_id = img_context_token_id

    # Compute num_image_token from model config
    num_image_token = model.num_image_token
    print(f'num_image_token: {num_image_token}, img_context_token_id: {img_context_token_id}')

    train_dataset = MultimodalContrastiveDataset(
        args.train_jsonl, args.image_root, tokenizer,
        args.image_size, args.max_text_len,
        num_image_token=num_image_token,
    )
    val_dataset = MultimodalContrastiveDataset(
        args.val_jsonl, args.image_root, tokenizer,
        args.image_size, args.max_text_len,
        num_image_token=num_image_token,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.per_device_train_batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ── 5. Optimizer + Scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, weight_decay=0.01,
    )

    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.num_train_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    print(f'Effective batch: {effective_batch}, total steps: {total_steps}')

    # ── 6. Training loop ────────────────────────────────────────────────────
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(args.num_train_epochs):
        model.train()
        # Keep frozen modules in eval mode
        model.vision_model.eval()
        model.contrastive_head.eval()

        total_loss = 0.0
        total_loss_v = 0.0
        total_loss_m = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values'].to(device, dtype=dtype)
            texture_pixel_values = batch['texture_pixel_values'].to(device, dtype=dtype)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            result = model.multimodal_contrastive_forward(
                pixel_values=pixel_values,
                texture_pixel_values=texture_pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                alpha=args.alpha,
                beta=args.beta,
            )

            loss = result['loss'] / args.gradient_accumulation_steps
            loss.backward()

            total_loss += result['loss'].item()
            total_loss_v += result['loss_visual'].item()
            total_loss_m += result['loss_multimodal'].item()

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
                    n = args.gradient_accumulation_steps
                    print(f'Epoch {epoch+1} step {global_step} | '
                          f'loss={total_loss/(batch_idx+1):.4f} '
                          f'L_vis={total_loss_v/(batch_idx+1):.4f} '
                          f'L_mm={total_loss_m/(batch_idx+1):.4f} '
                          f'tau={tau:.4f}')

                if global_step % args.save_steps == 0:
                    save_checkpoint(model, args, global_step, prefix='step')

        avg_loss = total_loss / len(train_loader)

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_loss_v = 0.0
        val_loss_m = 0.0

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device, dtype=dtype)
                texture_pixel_values = batch['texture_pixel_values'].to(device, dtype=dtype)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                result = model.multimodal_contrastive_forward(
                    pixel_values=pixel_values,
                    texture_pixel_values=texture_pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    alpha=args.alpha,
                    beta=args.beta,
                )
                val_loss += result['loss'].item()
                val_loss_v += result['loss_visual'].item()
                val_loss_m += result['loss_multimodal'].item()

        val_loss /= len(val_loader)
        val_loss_v /= len(val_loader)
        val_loss_m /= len(val_loader)

        print(f'=== Epoch {epoch+1}/{args.num_train_epochs} | '
              f'train={avg_loss:.4f} val={val_loss:.4f} '
              f'(vis={val_loss_v:.4f} mm={val_loss_m:.4f}) ===')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, args, epoch + 1, prefix='best')
            print(f'  Best model saved (val_loss={val_loss:.4f})')

    print('Training complete.')


def save_checkpoint(model, args, step, prefix='step'):
    """Save trainable weights: RetrievalHead, MLP, LoRA, temperature."""
    path = os.path.join(args.output_dir, f'{prefix}-{step}')
    os.makedirs(path, exist_ok=True)

    save_dict = {
        'retrieval_head': model.retrieval_head.state_dict(),
        'contrastive_head': model.contrastive_head.state_dict(),
        'contrastive_log_temp': model.contrastive_log_temp,
        'mlp1': model.mlp1.state_dict(),
    }

    # Save LoRA weights from LLM
    lora_state = {}
    for name, param in model.language_model.named_parameters():
        if 'lora' in name:
            lora_state[name] = param.data.cpu()
    save_dict['lora_weights'] = lora_state

    torch.save(save_dict, os.path.join(path, 'multimodal_contrastive_weights.pt'))
    print(f'  Saved checkpoint: {path}')


if __name__ == '__main__':
    main()
