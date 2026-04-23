import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_transform(image_size=448):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class ContrastivePairDataset(Dataset):
    """
    Dataset for image-texture contrastive learning.

    Each JSONL line:
        {"id": "123", "image": "123/fig.png", "texture": "123/texture.png"}

    Returns (pixel_values, texture_pixel_values) pairs where both are
    processed by the same ViT transform.
    """

    def __init__(self, jsonl_path: str, image_root: str, image_size: int = 448):
        self.image_root = image_root
        self.transform = build_transform(image_size)
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

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

        return {
            'pixel_values': pixel_values,
            'texture_pixel_values': texture_pixel_values,
        }
