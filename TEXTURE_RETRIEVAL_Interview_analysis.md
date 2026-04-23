# InternVL Texture Retrieval — 项目说明

## 目标

给定一张 3D 游戏资产的**渲染图**（fig.png），从贴图库中**检索**出对应的 **UV 材质贴图**（texture.png）。

## 架构

```
                InternVL2-1B (939M params)
┌──────────────────────────────────────────────────┐
│                                                  │
│  ┌─────────────┐    ┌──────┐    ┌─────────────┐  │
│  │  InternViT   │    │ MLP1 │    │  InternLM2  │  │
│  │  (300M)      │───→│bridge│───→│  (1.8B LLM) │  │  ← 原有 VQA 路径
│  │              │    └──────┘    └─────────────┘  │
│  │  ViT-300M    │                                 │
│  │              │    ┌──────────────────────┐     │
│  │              │───→│  ContrastiveHead     │     │  ← 新增 检索路径
│  │              │    │  (2层MLP, 768→256)   │     │
│  └─────────────┘    └──────────────────────┘     │
│                      + contrastive_log_temp       │
│                        (可学习温度标量)             │
└──────────────────────────────────────────────────┘
```

两条路径**共享同一个 ViT**，互不干扰：
- **VQA 路径**：ViT → MLP1 bridge → LLM → 文本输出（原有功能）
- **检索路径**：ViT → ContrastiveHead → 256维 embedding → 余弦相似度检索（新增功能）

### 检索路径逐步解释

```
1. ViT（Vision Transformer，视觉编码器）
   - 输入：一张 448×448 的图片
   - 把图片切成 32×32 = 1024 个 patch（每个 14×14 像素）
   - 加上 1 个 CLS token（代表整张图的全局语义）
   - 经过 Transformer 编码
   - 输出：1025 个 768 维向量

2. CLS token 提取
   - 只取第 0 个位置的向量（CLS token）
   - 这一个 768 维向量浓缩了整张图片的语义信息
   - 类比 BERT 里的 [CLS] 用于句子级分类

3. ContrastiveHead（投影头，2 层 MLP）
   - Linear(768→768) + ReLU + Linear(768→256)
   - 把 768 维"通用视觉特征"投影到 256 维"材质检索专用空间"
   - 过滤掉跟材质无关的信息（光照、背景、形状等），只保留材质相关信号

4. L2 归一化
   - 把 256 维向量缩放到单位长度（向量长度 = 1）
   - 归一化后，向量内积 = 余弦相似度，计算更简单

5. 余弦相似度检索
   - 一张渲染图的 embedding (1, 256) × 所有贴图的 embedding (N, 256)^T = (1, N)
   - 得到这张渲染图和 N 张贴图的相似度分数
   - 按分数从高到低排序，top-1 就是最匹配的贴图
```

**类比**：类似用 Google 搜图——你给一张照片（渲染图），搜索引擎在图库（贴图库）里找最像的返回给你。区别是 Google 用文本匹配，我们用 ViT 提取的视觉 embedding 匹配。

## 模型组件

| 组件 | 参数量 | 作用 | 训练状态 |
|------|--------|------|---------|
| InternViT-300M | 304M | 视觉编码器，提取图像特征 | fine-tuned（unfrozen） |
| ContrastiveHead | ~400K | 投影头，768维→256维检索空间 | 从零训练 |
| contrastive_log_temp | 1 | InfoNCE 温度参数，存储 log(τ)=-2.66，实际温度 τ=exp(-2.66)=0.07 | 从零训练 |
| MLP1 bridge | ~4M | ViT→LLM 的桥接层 | 冻结（未改动） |
| InternLM2-1.8B | 630M | 语言模型 | 冻结（训练时未加载） |

## 我们做了什么 / 没做什么

### ViT = Vision Transformer

把图片切成 14×14 像素的小块（patch），每个 patch 当作一个 token 输入 Transformer 编码器，输出图像特征向量。和 NLP 里的 Transformer 结构一样，只是输入从文字变成了图片 patch。

### 训练范围

```
InternVL2-1B 完整结构:

┌────────────────┐     ┌──────────┐     ┌────────────────┐
│  InternViT-300M │ ──→ │ MLP1     │ ──→ │ InternLM2-1.8B │ ──→ 文本输出
│  (视觉编码器)    │     │ (bridge) │     │ (语言模型)      │
└────────────────┘     └──────────┘     └────────────────┘
       ↓
┌────────────────┐
│ ContrastiveHead │ ──→ 256维 embedding ──→ 检索
│ (投影头)         │
└────────────────┘
```

| 组件 | 做了什么 | 是否属于 MLLM 微调 |
|------|---------|-------------------|
| InternViT-300M | **微调了**（unfrozen, 权重被更新） | 否，ViT 微调 ≠ MLLM 微调 |
| ContrastiveHead | **从零训练**（新增组件） | 否，是对比学习的投影头 |
| contrastive_log_temp | **从零训练**（1个标量） | 否 |
| MLP1 bridge | 没动（冻结） | — |
| InternLM2-1.8B | **完全没动**（训练时甚至没加载到 GPU） | — |

### 这算什么任务？

| | 我们做的 | MLLM 微调（没做） |
|---|---|---|
| 任务类型 | Contrastive Learning（对比学习） | SFT / RLHF |
| 训练目标 | 拉近 image↔texture embedding 距离 | 让 LLM 生成正确的文本回答 |
| 损失函数 | InfoNCE loss | Cross-Entropy (next token prediction) |
| 训练了什么 | ViT + 投影头 | ViT + bridge + LLM（全链路） |
| 输出形式 | 256维向量（用于检索） | 自然语言文本 |
| 数据格式 | `{image, texture}` 图片对 | `{image, conversations}` 对话数据 |

### 如果要做 MLLM 微调？

那是另一个任务——让模型看一张渲染图，用自然语言描述材质：

```
输入: <image> 描述这个 3D 物体的材质纹理
输出: "该物体表面为深蓝色金属材质，带有磨砂质感和细微划痕纹理..."
```

需要：
- 对话格式的训练数据（conversations）
- SFT 训练脚本（`internvl_chat_finetune.py`）
- 训练 ViT + MLP1 bridge + LLM 全链路
- 显存需求远大于当前方案（需要加载完整 LLM）

## 温度参数 τ 详解

### 什么是温度

InfoNCE loss 公式：

```
                    exp(sim(img, tex_正确) / τ)
loss = -log ─────────────────────────────────────────
              Σ_j  exp(sim(img, tex_j) / τ)
```

τ（温度）控制 softmax 分布的"尖锐程度"——即模型对相似度差异的敏感度。

### 温度的效果（具体数字）

假设 batch_size=4，img_0 和 4 张 texture 的余弦相似度为 `[0.80, 0.75, 0.70, 0.30]`：

**τ=0.07（小温度）：**
```
logits = [0.80, 0.75, 0.70, 0.30] / 0.07 = [11.4, 10.7, 10.0, 4.3]
softmax = [0.66, 0.27, 0.07, 0.00]
→ 模型非常确信"就是第0个"，差距被放大
```

**τ=0.3（适中温度）：**
```
logits = [0.80, 0.75, 0.70, 0.30] / 0.3 = [2.67, 2.50, 2.33, 1.00]
softmax = [0.33, 0.28, 0.24, 0.15]
→ 有区分度但不极端，每个负样本都还有"存在感"
```

**τ=10（大温度）：**
```
logits = [0.80, 0.75, 0.70, 0.30] / 10 = [0.08, 0.075, 0.07, 0.03]
softmax = [0.27, 0.26, 0.25, 0.25]
→ 几乎均匀分布，模型觉得"都差不多"
```

### 温度与 exploration/exploitation

| | 低温度（τ→0） | 高温度（τ→∞） |
|---|---|---|
| softmax 分布 | 尖锐（接近 one-hot） | 平坦（接近均匀） |
| 行为 | exploitation — 只关注最相似的 | exploration — 每个候选都考虑 |
| 类比 LLM | temperature=0，确定性输出 | temperature=2，随机多样输出 |

### 温度对梯度的影响

cross_entropy 梯度 = `softmax(i) - label(i)`：

**τ=0.07 训练几步后**（正确 pair 的 sim 被推到 0.95）：
```
logits = [0.95, 0.70, 0.65, ...] / 0.07 = [13.6, 10.0, 9.3, ...]
softmax = [0.97, 0.02, 0.01, ...]

正确答案梯度 = 0.97 - 1.0 = -0.03  ← 几乎为零，学不动了
其余负样本梯度 ≈ 0.00              ← 完全没有信号
→ entropy collapse：分布坍缩成 one-hot，训练停滞
```

**τ=0.3 同样情况**：
```
logits = [0.95, 0.70, 0.65, ...] / 0.3 = [3.17, 2.33, 2.17, ...]
softmax = [0.42, 0.18, 0.15, 0.05, ...]

正确答案梯度 = 0.42 - 1.0 = -0.58  ← 梯度充足
负样本梯度 = 0.18, 0.15, 0.05      ← 多个负样本都贡献梯度
→ 训练继续有效推进
```

### 温度与 batch_size 的关系

| batch_size | 推荐 τ | 原因 |
|---|---|---|
| 32768（CLIP） | 0.07 | 负样本极多，即使 softmax 尖锐也总能撞到 hard negatives |
| 256 | 0.07~0.1 | 负样本较多，τ=0.07 基本安全 |
| **32（本项目）** | **0.1~0.5** | 只有 31 个负样本，τ 太小会导致有效负样本只剩 1~2 个 |

### 为什么存 log(τ) 而不是 τ

温度必须恒正（τ > 0），如果直接用 `nn.Parameter(0.07)`，梯度更新可能把它推成负数。
存 log(τ) 的话，实际温度 = exp(log_τ)，而 exp(任何实数) 恒正：

```python
self.contrastive_log_temp = nn.Parameter(torch.tensor(-2.6593))  # log(0.07)

# 使用时
temp = self.contrastive_log_temp.exp()  # 恒正，无需额外约束
```

### 本项目的温度分析

我们使用 τ=0.07 + batch_size=32，理论上有 entropy collapse 风险。

**实际训练结果**：没有 collapse，loss 持续下降 4 个 epoch：
```
Epoch 1: val_loss = 1.072
Epoch 3: val_loss = 0.319
Epoch 5: val_loss = 0.249  ← 仍在降，但趋于平缓
```

**原因**：数据中天然存在大量 hard negatives（同系列模型共享极相似的贴图），
即使 softmax 很尖锐，一个 batch 里也大概率撞上多个高相似度的负样本，梯度信号不会完全消失。

**但 Epoch 4→5 val_loss 几乎不降（0.251→0.249）**，可能是轻微 collapse 的迹象——
模型已过度自信，剩余 hard cases 的梯度不足以继续推动学习。
调大 τ 到 0.2~0.3 可能还能再压低 val_loss，但预计收益有限。

## 推理流程

```python
# 1. 加载模型
model = InternVLChatModel.from_pretrained("InternVL2-1B")

# 2. 加载训练好的 contrastive 权重
ckpt = torch.load("best_contrastive_weights.pt")
model.contrastive_head.load_state_dict(ckpt['contrastive_head'])
model.vision_model.load_state_dict(ckpt['vision_model'])  # fine-tuned ViT

# 3. 编码
img_emb = model.get_cls_embedding(rendered_image)   # (1, 256)
tex_emb = model.get_cls_embedding(texture_image)     # (1, 256)

# 4. 检索：余弦相似度排序
similarity = torch.matmul(img_emb, all_tex_embeddings.T)  # (1, N)
top_k = similarity.argsort(descending=True)[:5]            # top-5 最相似
```

## 数据流（Tensor Shape 变化）

以 batch_size=32 为例，完整数据流：

```
输入: 32 张 fig.png（渲染图）+ 32 张 texture.png（贴图）
      ↓
transforms: PIL Image → (3, 448, 448)       ← RGB 3通道, 448×448 像素
      ↓
batch 拼接: (32, 3, 448, 448)               ← 32 张图
      ↓
┌──────────────────────────────────────────────────────────────────┐
│ ViT (vision_model)                                               │
│                                                                  │
│   图片切 patch: 448 ÷ 14 = 32 patches/边 → 32×32 = 1024 patches │
│   加上 1 个 CLS token → 共 1025 个 token                         │
│   每个 token 768 维                                               │
│                                                                  │
│   输出: (32, 1025, 768)                                          │
│          B   N+1    C                                            │
└──────────────────────────────────────────────────────────────────┘
      ↓
cls = vit_out[:, 0, :]              ← 只取第 0 个位置（CLS token）
      ↓
(32, 768)                            ← 每张图 → 一个 768 维向量
      ↓
┌──────────────────────────────────────────────────────────────────┐
│ ContrastiveHead（2 层 MLP）                                       │
│                                                                  │
│   Linear(768, 768)    (32, 768) → (32, 768)    ← 第1个全连接层   │
│         ↓                                                        │
│   ReLU                (32, 768) → (32, 768)    ← 非线性激活      │
│         ↓                                                        │
│   Linear(768, 256)    (32, 768) → (32, 256)    ← 第2个全连接层   │
│         ↓                                                        │
│   L2 normalize        (32, 256) → (32, 256)    ← 每个向量长度=1  │
└──────────────────────────────────────────────────────────────────┘
      ↓
img_emb: (32, 256)                   ← 32 张渲染图的 embedding
tex_emb: (32, 256)                   ← 32 张贴图的 embedding（同样流程）
      ↓
┌──────────────────────────────────────────────────────────────────┐
│ 相似度矩阵                                                       │
│                                                                  │
│   sim = img_emb @ tex_emb.T                                      │
│         (32, 256) × (256, 32) = (32, 32)                         │
│                                                                  │
│   sim[i][j] = 第 i 张渲染图和第 j 张贴图的余弦相似度              │
│   对角线 sim[i][i] 是正确配对                                     │
│                                                                  │
│   示例（B=4 简化）:                                               │
│            tex_0  tex_1  tex_2  tex_3                             │
│   img_0  [ 0.95,  0.12,  0.30,  0.08]  ← img_0↔tex_0 最高 ✓    │
│   img_1  [ 0.10,  0.92,  0.15,  0.20]  ← img_1↔tex_1 最高 ✓    │
│   img_2  [ 0.28,  0.18,  0.97,  0.05]  ← img_2↔tex_2 最高 ✓    │
│   img_3  [ 0.07,  0.22,  0.03,  0.96]  ← img_3↔tex_3 最高 ✓    │
└──────────────────────────────────────────────────────────────────┘
```

## 评估指标

评估时构建完整的相似度矩阵 (N, N)，N=1956（验证集大小）。
对每个 query（渲染图），把所有 N 张贴图按相似度从高到低排序，看正确贴图排在第几位（rank）。

### Recall@K

**在 top-K 结果里找到正确答案的比例。**

```
Recall@K = 正确答案排在前 K 名的 query 数量 / 总 query 数量

示例（5 个 query 的 rank）:
  query_0: rank=1   ← 正确答案排第 1
  query_1: rank=3   ← 正确答案排第 3
  query_2: rank=1   ← 正确答案排第 1
  query_3: rank=12  ← 正确答案排第 12
  query_4: rank=5   ← 正确答案排第 5

Recall@1  = 2/5 = 40%    ← rank≤1 的有 query_0, query_2
Recall@5  = 4/5 = 80%    ← rank≤5 的有 query_0, query_1, query_2, query_4
Recall@10 = 4/5 = 80%    ← rank≤10 的同上
```

| 指标 | 含义 | 本项目参考值 |
|------|------|-------------|
| Recall@1 | 第一名就是正确答案的概率 | 越高越好，>50% 可用 |
| Recall@5 | 前 5 名里包含正确答案 | >80% 表示用户翻一页就能找到 |
| Recall@10 | 前 10 名里包含正确答案 | >90% 表示检索基本可靠 |

### MRR（Mean Reciprocal Rank）

**正确答案排名的倒数的平均值。**

```
MRR = mean(1 / rank)

示例:
  query_0: rank=1  → 1/1 = 1.0
  query_1: rank=3  → 1/3 = 0.33
  query_2: rank=1  → 1/1 = 1.0
  query_3: rank=12 → 1/12 = 0.083
  query_4: rank=5  → 1/5 = 0.2

MRR = (1.0 + 0.33 + 1.0 + 0.083 + 0.2) / 5 = 0.523
```

MRR 的特点：rank=1 贡献 1.0，rank=2 贡献 0.5，rank=10 只贡献 0.1。
**越接近 1.0 越好**，意味着大部分 query 的正确答案都在第 1 名。

### Median Rank

**所有 query 的 rank 的中位数。**

```
ranks = [1, 3, 1, 12, 5] → 排序 → [1, 1, 3, 5, 12] → 中位数 = 3
```

Median Rank = 3 表示一半的 query 在前 3 名就能找到正确答案。**越小越好，理想值 = 1**。

### 其他常见检索指标（本项目未使用）

| 指标 | 含义 | 适用场景 |
|------|------|---------|
| mAP (mean Average Precision) | 每个 query 可能有多个正确答案时的综合指标 | 图片搜索（一张 query 可能匹配多张图） |
| NDCG (Normalized Discounted Cumulative Gain) | 考虑正确答案的相关性分级 | 搜索排序（结果有"很相关/一般相关/不相关"） |
| Precision@K | top-K 里有多少个是正确的 | 一个 query 有多个正确答案时 |
| R-Precision | 正确答案数=R 时，top-R 里命中多少 | 不同 query 正确答案数量不同时 |

本项目每个 query 只有 1 个正确答案（1 张渲染图对应 1 张贴图），所以 **Recall@K + MRR + Median Rank** 已经足够。

## 关键方法（modeling_internvl_chat.py）

| 方法 | 位置 | 用途 |
|------|------|------|
| `get_cls_embedding()` | L315 | ViT 编码 → CLS token → ContrastiveHead → 256维归一化向量 |
| `contrastive_forward()` | L325 | 训练用，计算 InfoNCE 对比损失 |
| `ContrastiveHead` | L33 | 2层MLP: Linear(768,768) → ReLU → Linear(768,256) → L2 normalize |

## 数据

| 项目 | 路径 |
|------|------|
| 渲染数据 | `/home/ubuntu/alan_repository/texture_material/output/` |
| 训练集 | `/home/ubuntu/alan_repository/texture_material/dataset/train.jsonl` (17612 pairs) |
| 验证集 | `/home/ubuntu/alan_repository/texture_material/dataset/val.jsonl` (1956 pairs) |
| 基础模型 | `/home/ubuntu/alan_repository/models/InternVL2-1B` |
| 最佳权重 | `/home/ubuntu/alan_repository/texture_material/contrastive_ckpt_unfrozen/best_contrastive_weights.pt` (583MB) |

数据格式（每行 JSONL）：
```json
{"id": "2230", "image": "2230/fig.png", "texture": "2230/texture.png"}
```

## 训练

```bash
cd /home/ubuntu/alan_repository/InternVL/internvl_chat

python internvl/train/internvl_contrastive_finetune.py \
    --model_name_or_path /home/ubuntu/alan_repository/models/InternVL2-1B \
    --train_jsonl /home/ubuntu/alan_repository/texture_material/dataset/train.jsonl \
    --val_jsonl   /home/ubuntu/alan_repository/texture_material/dataset/val.jsonl \
    --image_root  /home/ubuntu/alan_repository/texture_material/output \
    --output_dir  /home/ubuntu/alan_repository/texture_material/contrastive_ckpt_unfrozen \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-4 \
    --freeze_vit False
```

训练结果（unfrozen ViT, bs=32, lr=1e-4, 5 epochs）：

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 1.695 | 1.072 |
| 2 | 0.714 | 0.574 |
| 3 | 0.339 | 0.319 |
| 4 | 0.176 | 0.251 |
| 5 | 0.143 | 0.249 |

## 评估

```bash
# 指标评估
python internvl/train/eval_retrieval.py \
    --model_name_or_path /home/ubuntu/alan_repository/models/InternVL2-1B \
    --ckpt /home/ubuntu/alan_repository/texture_material/contrastive_ckpt_unfrozen/best_contrastive_weights.pt

# Case 可视化（输出对比图到 retrieval_cases/）
python internvl/train/eval_retrieval_cases.py \
    --model_name_or_path /home/ubuntu/alan_repository/models/InternVL2-1B \
    --ckpt /home/ubuntu/alan_repository/texture_material/contrastive_ckpt_unfrozen/best_contrastive_weights.pt \
    --num_cases 20 --topk 5
```

## 文件结构

```
InternVL/internvl_chat/internvl/
├── model/internvl_chat/
│   └── modeling_internvl_chat.py    # 模型定义（新增 ContrastiveHead + get_cls_embedding + contrastive_forward）
├── train/
│   ├── internvl_contrastive_finetune.py  # 对比学习训练脚本
│   ├── dataset_contrastive.py            # 数据集（读 fig.png + texture.png pairs）
│   ├── eval_retrieval.py                 # Recall@K 评估
│   └── eval_retrieval_cases.py           # Case 可视化
```

## 环境

- GPU: NVIDIA L20 (46GB VRAM)
- Conda: `qwen` 环境
- 依赖: `torch`, `transformers`, `timm`, `einops`, `pillow`
- 训练显存: ~14GB（ViT-only 加载，跳过 LLM）
