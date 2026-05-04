"""
Dataset for multimodal contrastive learning: (rendered_image, texture, text_description).

Each JSONL line:
    {"id": "123", "image": "123/fig.png", "texture": "123/texture.png", "description": "rough dark oak..."}

Returns:
    pixel_values           — rendered image for ViT
    texture_pixel_values   — texture image for ViT
    input_ids              — tokenized "<image>\n{description}" for LLM
    attention_mask         — attention mask for LLM
"""

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_transform(image_size=448):
    return transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class MultimodalContrastiveDataset(Dataset):
    """
    Triplet dataset: (rendered_image, texture, text_description).
    The text is wrapped as "<image>\\n{description}" so the LLM receives
    visual tokens followed by the text prompt.
    """

    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        tokenizer,
        image_size: int = 448,
        max_text_len: int = 80,
        img_context_token: str = '<IMG_CONTEXT>',
        num_image_token: int = 256,
    ):
        self.image_root = image_root
        self.transform = build_transform(image_size)
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.img_context_token = img_context_token
        self.num_image_token = num_image_token

        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    # Skip samples without description
                    if obj.get('description', '').strip():
                        self.samples.append(obj)

        print(f'MultimodalContrastiveDataset: {len(self.samples)} samples from {jsonl_path}')

    def _build_text_input(self, description: str):
        """Build input_ids for: <IMG_CONTEXT> * num_image_token + \\n + description.

        This mimics InternVL's chat format where visual tokens are inserted
        at <IMG_CONTEXT> positions, followed by the text.
        """
        # Build the prompt string with image context tokens
        img_tokens = self.img_context_token * self.num_image_token
        prompt = f'{img_tokens}\n{description}'

        encoded = self.tokenizer(
            prompt,
            max_length=self.num_image_token + self.max_text_len + 2,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = os.path.join(self.image_root, item['image'])
        tex_path = os.path.join(self.image_root, item['texture'])

        image = Image.open(img_path).convert('RGB')
        texture = Image.open(tex_path).convert('RGB')

        pixel_values = self.transform(image)
        texture_pixel_values = self.transform(texture)

        input_ids, attention_mask = self._build_text_input(item['description'])

        return {
            'pixel_values': pixel_values,
            'texture_pixel_values': texture_pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
