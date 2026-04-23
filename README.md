<div align="center">

# From Render to Material: Cross-Modal Alignment of a Vision-Language Model for Texture Retrieval

**WU XIANGYU**
*Nanyang Technological University — MIAO Lab*

[![Paper](https://img.shields.io/badge/paper-PDF-red)](paper/main.pdf)
[![License](https://img.shields.io/badge/license-MIT-blue)]()

</div>

---

## Abstract

Retrieving a physically-based rendering (PBR) material texture from a large library given only a rendered image of a 3D object is a challenging **cross-modal alignment** problem: the query is a composite scene image that conflates geometry, lighting, and material appearance, while the gallery contains flat, uniformly-lit texture maps. We address this *render-to-material* retrieval task by contrastively fine-tuning the visual backbone of **InternVL2-1B**, a state-of-the-art vision-language model (VLM). The core contribution is a **multimodal alignment** procedure that maps rendered images and raw PBR texture maps into a shared 256-dimensional embedding space via a symmetric InfoNCE loss with learned temperature. We construct a paired dataset of **19,568** (render, texture) pairs. On the full validation gallery of 1,956 textures, the aligned model achieves **Recall@1 = 30%**, **Recall@3 = 70%**, and **Recall@5 = 80%**, demonstrating that a pre-trained VLM backbone can be efficiently adapted to bridge the domain gap between rendered scenes and material maps.

---

## Overview

The fundamental challenge is **cross-modal alignment**: a rendered scene image and a flat texture map depict the *same material* but look visually very different due to geometry, lighting, and shading effects. Standard image similarity metrics fail because they cannot disentangle material appearance from scene-level factors.

<div align="center">

```
  QUERY MODALITY                    GALLERY MODALITY
  (Rendered Scene)                  (Flat Texture Map)
        │                                  │
        ▼                                  ▼
  InternVL2 ViT                    InternVL2 ViT   ← shared weights
        │                                  │
      CLS token                          CLS token
        │                                  │
        ▼                                  ▼
  ContrastiveHead                  ContrastiveHead  ← shared weights
  (MLP: 1024→1024→256)             (MLP: 1024→1024→256)
        │                                  │
        ▼                                  ▼
    u ∈ S²⁵⁵  ─── InfoNCE loss ───  v ∈ S²⁵⁵
```

</div>

Both modalities are processed by **the same shared ViT + ContrastiveHead**. The symmetric InfoNCE loss performs alignment: matching (render, texture) pairs are pulled together on the unit hypersphere, while non-matching pairs are pushed apart.

---

## Key Results

| Model | R@1 | R@3 | R@5 |
|---|:---:|:---:|:---:|
| Random baseline | 0.05% | 0.15% | 0.26% |
| InternVL2-1B (zero-shot, no alignment) | — | — | — |
| **Ours — contrastively aligned ViT** | **30%** | **70%** | **80%** |

Evaluation gallery: **1,956** unique PBR material textures.

The large R@1→R@3 jump (+40%) indicates the model consistently aligns the correct texture to the top-3, with near-identical scores among visually similar material variants.

---

## Dataset

Each sample is a pair **(fig, texture)** — two different *views* of the same material:

| Image type | Description | Role in retrieval |
|---|---|---|
| **`fig.png`** | Rendered 3D object with PBR material under standard lighting | Query |
| **`texture.png`** | Flat PBR albedo/diffuse map under neutral illumination | Gallery item |

```
output/
└── {material_id}/
    ├── fig.png          # rendered scene image  (query modality)
    └── texture.png      # flat PBR texture map  (gallery modality)
```

JSONL format:
```json
{"id": "1277", "image": "1277/fig.png", "texture": "1277/texture.png"}
```

| Split | # Pairs |
|---|---|
| Train | 17,612 |
| Validation | 1,956 |
| **Total** | **19,568** |

---

## Method

### Multimodal Alignment Framework

We fine-tune InternVL2-1B's visual backbone with a **contrastive alignment objective**, discarding the language model entirely. The pipeline has two stages:

**Encoding:** Both modalities share identical processing:
$$h(\mathbf{x}) = \ell_2\!\left(\mathbf{W}_2\,\text{ReLU}(\mathbf{W}_1\,\text{CLS}(\mathbf{x}))\right) \in \mathbb{R}^{256}$$

where $\mathbf{W}_1 \in \mathbb{R}^{1024 \times 1024}$, $\mathbf{W}_2 \in \mathbb{R}^{1024 \times 256}$, and $\ell_2$ normalisation places embeddings on the unit hypersphere.

**Alignment loss:** For a mini-batch of $B$ pairs $\{(\mathbf{f}_i, \mathbf{t}_i)\}$:
$$\mathcal{L}_{\text{align}} = \frac{1}{2}\left[\ell(\mathbf{U},\mathbf{V}) + \ell(\mathbf{V},\mathbf{U})\right]$$

$$\ell(\mathbf{A},\mathbf{B}) = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\mathbf{a}_i^\top\mathbf{b}_i / \tau)}{\sum_{j=1}^{B}\exp(\mathbf{a}_i^\top\mathbf{b}_j / \tau)}$$

The temperature $\tau = \exp(\phi)$ is **learned jointly** with the model, initialised at $\phi_0 = \log(0.07)$ and clamped to $[e^{-4.6}, e^{2.3}]$ to prevent collapse. It is frozen for the first two epochs for stability.

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Base model | InternVL2-1B (ViT-300M, $d=1024$) |
| Projection dimension | 256 |
| Optimiser | AdamW (lr=1e-4, wd=0.01) |
| LR schedule | Cosine annealing |
| Gradient clipping | max-norm 1.0 |
| Effective batch size | 64 |
| Epochs | 5 |
| ViT | Unfrozen (full backbone alignment) |
| Temperature $\tau_0$ | 0.07 |

### Inference

At query time the rendered image is encoded to a 256-dim vector. The gallery of texture embeddings is pre-computed and stored in a **FAISS `IndexFlatIP`** index. Retrieval is a single inner-product search — $O(N)$ for gallery size $N$.

---

## Repository Structure

```
InternVL/
├── internvl_chat/
│   └── internvl/
│       ├── model/internvl_chat/
│       │   ├── modeling_internvl_chat.py     # + ContrastiveHead, get_cls_embedding
│       │   ├── modeling_intern_vit.py        # + CLS token output
│       │   └── configuration_internvl_chat.py
│       └── train/
│           ├── internvl_contrastive_finetune.py   # alignment training script
│           ├── dataset_contrastive.py              # ContrastivePairDataset
│           ├── eval_retrieval.py                   # Recall@K / MRR
│           ├── eval_retrieval_cases.py             # qualitative case analysis
│           └── test_retrieval.py                   # quick sanity check
├── paper/
│   ├── main.tex    # LaTeX source
│   └── main.pdf    # compiled paper
└── TEXTURE_RETRIEVAL_Interview_analysis.md
```

---

## Quick Start

### Install

```bash
pip install torch torchvision transformers einops timm faiss-cpu
```

### Train (Contrastive Alignment)

```bash
cd internvl_chat/internvl/train
python internvl_contrastive_finetune.py \
    --model_name_or_path /path/to/InternVL2-1B \
    --train_jsonl /path/to/train.jsonl \
    --val_jsonl   /path/to/val.jsonl \
    --image_root  /path/to/images \
    --output_dir  ./contrastive_ckpt \
    --num_train_epochs 5 \
    --per_device_train_batch_size 64 \
    --learning_rate 1e-4 \
    --freeze_vit False
```

### Evaluate

```bash
python eval_retrieval.py \
    --model_name_or_path /path/to/InternVL2-1B \
    --ckpt_path ./contrastive_ckpt/best_contrastive_weights.pt \
    --val_jsonl /path/to/val.jsonl \
    --image_root /path/to/images
```

---

## Analysis of Alignment Quality

**Success cases:** Materials with distinctive visual signatures (unique colour + texture pattern) align accurately — the correct gallery item reaches rank 1 with a clear margin (e.g., query `15662`: score 0.873 vs 0.872 for the nearest neighbour).

**Failure modes:**
- *Same-family confusion* — Variants of the same material (e.g., same wood grain, slightly different hue) produce near-identical embeddings; rank differences < 0.001.
- *Geometry–material entanglement* — Complex geometry (highlights, shadows) can dominate the embedding, causing the model to align to texture colour rather than material identity.

---

## Citation

```bibtex
@article{retr2025,
  title   = {From Render to Material: Cross-Modal Alignment of a Vision-Language Model for Texture Retrieval},
  author  = {Anonymous},
  year    = {2025},
}
```

---

## Acknowledgements

This work builds upon [InternVL](https://github.com/OpenGVLab/InternVL) (OpenGVLab). The contrastive alignment framework is inspired by [CLIP](https://arxiv.org/abs/2103.00020) and [SimCLR](https://arxiv.org/abs/2002.05709).
