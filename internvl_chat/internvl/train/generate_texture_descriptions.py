"""
Generate text descriptions for each texture using InternVL2-1B's chat ability.
Produces a JSONL with {"id": ..., "image": ..., "texture": ..., "description": ...}

Usage:
    python generate_texture_descriptions.py \
        --model_path /home/ubuntu/alan_repository/models/InternVL2-1B \
        --input_jsonl /home/ubuntu/alan_repository/texture_material/dataset/train.jsonl \
        --image_root /home/ubuntu/alan_repository/texture_material/output \
        --output_jsonl /home/ubuntu/alan_repository/texture_material/dataset/train_with_desc.jsonl \
        --batch_size 8
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
from transformers import AutoTokenizer


PROMPT = "Describe this material texture in one sentence. Focus on: material type, color, surface roughness, pattern, and visual appearance."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--output_jsonl', required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Load full model (with LLM) for captioning
    print('Loading InternVL2 for captioning...')
    cfg = InternVLChatConfig.from_pretrained(args.model_path)
    cfg.llm_config._attn_implementation = "eager"
    cfg.llm_config.attn_implementation = "eager"

    model = InternVLChatModel.from_pretrained(
        args.model_path, config=cfg,
        torch_dtype=dtype, attn_implementation="eager",
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    generation_config = dict(max_new_tokens=80, do_sample=False)

    # Load samples
    samples = []
    with open(args.input_jsonl) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f'Loaded {len(samples)} samples')

    # Check for existing progress (resume support)
    done_ids = set()
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl) as f:
            for line in f:
                obj = json.loads(line)
                done_ids.add(str(obj['id']))
        print(f'Resuming: {len(done_ids)} already done')

    remaining = [s for s in samples if str(s['id']) not in done_ids]
    print(f'Remaining: {len(remaining)} samples')

    # Process one by one (chat doesn't support true batching easily)
    transform = transforms.Compose([
        transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with open(args.output_jsonl, 'a') as fout:
        for sample in tqdm(remaining, desc='Generating descriptions'):
            tex_path = os.path.join(args.image_root, sample['texture'])

            try:
                img = Image.open(tex_path).convert('RGB')
                pixel_values = transform(img).unsqueeze(0).to(device, dtype=dtype)

                question = f'<image>\n{PROMPT}'
                with torch.no_grad():
                    response = model.chat(
                        tokenizer, pixel_values, question,
                        generation_config,
                        num_patches_list=[1],
                    )

                out = {
                    'id': sample['id'],
                    'image': sample['image'],
                    'texture': sample['texture'],
                    'description': response.strip(),
                }
                fout.write(json.dumps(out) + '\n')
                fout.flush()

            except Exception as e:
                print(f'Error on {sample["id"]}: {e}')
                # Write with empty description so we don't retry
                out = {
                    'id': sample['id'],
                    'image': sample['image'],
                    'texture': sample['texture'],
                    'description': '',
                }
                fout.write(json.dumps(out) + '\n')
                fout.flush()

    print(f'Done. Output: {args.output_jsonl}')


if __name__ == '__main__':
    main()
