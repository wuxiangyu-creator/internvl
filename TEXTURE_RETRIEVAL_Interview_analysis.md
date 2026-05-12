# ReTR — Render-to-Material Texture Retrieval 项目说明

## 目标

给定一张 3D 游戏资产的**渲染图**（fig.png），从贴图库中**检索**出对应的 **PBR 材质贴图**（texture.png）。
支持两种检索模式：
- **纯视觉检索**：上传渲染图 → 返回最相似材质
- **对话式检索**：上传渲染图 + 自然语言描述（如"更粗糙一点的木材"）→ 返回语义匹配材质

## 整体架构（两阶段训练）

```
                        InternVL2-1B (975M params)
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  ┌─────────────┐                                                   │
│  │  InternViT   │    ┌──────────────────┐                          │
│  │  (300M)      │───→│ ContrastiveHead  │──→ 256维 embedding       │  ← Stage 1: 视觉对齐
│  │  ViT-300M    │    │ (1024→1024→256)  │    (纯视觉检索)           │
│  │              │    └──────────────────┘                          │
│  │              │                                                   │
│  │              │    ┌──────┐    ┌────────────┐    ┌─────────────┐ │
│  │              │───→│ MLP1 │───→│ Qwen2 LLM  │───→│RetrievalHead│ │  ← Stage 2: 多模态对齐
│  └─────────────┘    │bridge│    │ (558M)+全    │    │(896→448→256)│ │
│                     └──────┘    │参数/LoRA     │    └─────────────┘ │
│                                 └────────────┘     ↓               │
│  文本描述 ──→ Tokenizer ──→ text tokens ──────────→ │  256维 embedding│
│                                                     (对话式检索)    │
│                     + contrastive_log_temp (共享温度参数)            │
└────────────────────────────────────────────────────────────────────┘
                                    ↓
                          共享 256 维嵌入空间
                    ┌─────────────────────────┐
                    │  img_emb   ● (渲染图)     │
                    │  tex_emb   ● (贴图)       │  ←── 三种向量可互相检索
                    │  query_emb ● (图+文融合)   │
                    └─────────────────────────┘
                                    ↓
                          FAISS IndexFlatIP 检索
```

## 对比学习范式说明（为什么 Stage 1 用共享 ViT）

### 对比学习的本质

对比学习不是定义在"图像+文本"上的，它的核心是：

```
正样本对 (positive pair)  →  embedding 拉近
负样本对 (negative pair)  →  embedding 推远
```

**只要能定义"什么算正对、什么算负对"，就能做对比学习**。具体用什么模态、几个编码器都是工程选择，不是定义。

### 三种典型对比学习范式对比

| 范式 | 输入 | 编码器 | 正对来源 | 代表方法 |
|---|---|---|---|---|
| **多模态** | image + text | **不同**（ViT + TextTransformer） | 配对的图文 | CLIP, ALIGN |
| **同模态自监督** | image_view1 + image_view2 | **共享**（同一 ViT） | 同一图的两个数据增强 | SimCLR, MoCo, BYOL |
| **同模态跨域**（本项目 Stage 1） | render + texture | **共享**（同一 ViT） | 同一材质的渲染图与贴图 | ReTR Stage 1 |

### 为什么 CLIP 必须双塔？

```
image: (3, 224, 224)        ← 像素值，连续浮点数
text:  [101, 2003, ...]     ← token id，离散整数
```

输入空间根本不同 — ViT 没法处理 token，TextTransformer 没法处理像素。所以 CLIP 必须有两个不同架构的编码器。

### 为什么 SimCLR / ReTR Stage 1 可以单塔？

我们 Stage 1 的两个输入：

```
render_image:  448×448 RGB  ← 渲染图（有光照、阴影、几何）
texture_image: 448×448 RGB  ← 贴图（无光照、纯材质）
```

**两个都是 RGB 图像 — 同模态**，但属于两个不同的"域"：
- render 域：包含 lighting、geometry、view angle 信息
- texture 域：纯粹的 albedo / normal / roughness 等材质属性

正对：同一个 material id 的 (render, texture)
负对：batch 内不同 material id 的所有 (render_i, texture_j)

这是 **SimCLR 范式的扩展**，只是把"两个数据增强 view"换成了"两个语义相关但视觉不同的域 view"。学术上称为 **cross-domain same-modality contrastive learning** 或 **siamese contrastive learning**。

### 共享 ViT 的三大优势

**1. 强迫学到"光照/几何不变"的材质特征**

这是设计上的关键！如果用两个不同的 ViT：
```
ViT_render  专门看渲染图 → 可能依赖光影线索
ViT_texture 专门看贴图   → 可能依赖纹理周期性
```
两个特征空间会发散，学不到"材质本质"。

共享 ViT 后：
```
同一个 ViT 必须能在 render 上看出"这是哑光木材"
        也能在 texture 上看出"这是哑光木材"
```
**唯一的解就是学到与光照/几何无关的材质本质特征** — 正好是我们想要的。

**2. 参数量减半，训练更稳定**

300M ViT 而不是 600M，单卡 L20 能开更大 batch（更多负样本，InfoNCE 效果更好）。

**3. 推理时 gallery 和 query 共享空间**

因为两路是同一个 ViT，推理时：
- 离线：用 ViT 编码 N 个 texture，存进 FAISS（gallery）
- 在线：用**同一个 ViT** 编码 query render，直接做内积搜索

如果两路 ViT 不同，gallery 向量和 query 向量根本不在同一空间，无法比较。

### "这还是对比学习吗？" — 是，而且更纯粹

InfoNCE loss 的形式完全没变：
```python
logits = emb_a @ emb_b.T / τ   # B×B 相似度矩阵
labels = arange(B)              # 对角线是正对
loss = (CE(logits, labels) + CE(logits.T, labels)) / 2
```

这个公式不关心 emb_a 和 emb_b 来自哪个编码器、哪种模态。它只关心：
- 矩阵对角线的相似度要高（正对）
- 非对角线相似度要低（负对）

所以无论 CLIP（双塔跨模态）、SimCLR（单塔同模态自增强）、还是 ReTR Stage 1（单塔同模态跨域），**都是对比学习** — 只是"正对怎么构造"和"用几个编码器"在变。

### 三种范式架构对比图

```
       CLIP                      SimCLR                ReTR Stage 1
  ─────────────              ─────────────            ─────────────
  image     text              img_aug1   img_aug2     render    texture
    │        │                   │           │           │         │
    ▼        ▼                   ▼           ▼           ▼         ▼
  ┌───┐   ┌─────┐              ┌───┐       ┌───┐       ┌───┐     ┌───┐
  │ViT│   │Text │              │ViT│       │ViT│       │ViT│     │ViT│
  └───┘   │TFM  │              └───┘       └───┘       └───┘     └───┘
    │     └─────┘                │           │           │         │
    │        │              (共享权重)             (共享权重)
    ▼        ▼                   ▼           ▼           ▼         ▼
  ProjI    ProjT                Proj        Proj        Head      Head
    │        │                   │           │           │         │
    └──InfoNCE──┘                └─InfoNCE──┘            └─InfoNCE─┘
   跨模态对比                  自监督同域对比         跨域同模态对比
```

### 与 SimCLR 的直接类比

| | SimCLR 的"view" | ReTR Stage 1 的"view" |
|---|---|---|
| 数据 | 同一图的不同 augmentation | 同一材质的 render 和 texture |
| 不变性目标 | 对 crop / color / flip 不变 | 对 lighting / geometry / view angle 不变 |
| 学到的特征 | 视觉对象的本质（不依赖增强） | 材质的本质（不依赖渲染条件） |

**Stage 1 之所以用共享 ViT，不是省事的选择，而是设计上必要的选择** — 因为我们恰好有"天然配对"的两种 view（不需要做数据增强），共享 ViT 既保证了 query 和 gallery 在同一空间可比，也强迫模型学到 render-invariant material representation。

---

## 两阶段训练方案

### Stage 1: 纯视觉对比学习

**目标**：让渲染图和贴图在 embedding 空间中靠近

```
渲染图 ──→ ViT ──→ CLS token ──→ ContrastiveHead ──→ img_emb (B, 256)
贴图   ──→ ViT ──→ CLS token ──→ ContrastiveHead ──→ tex_emb (B, 256)

L_visual = SymmetricInfoNCE(img_emb, tex_emb)
```

| 配置 | 值 |
|------|-----|
| 训练数据 | 17,612 对 (渲染图, 贴图) |
| ViT | **解冻**，全量微调 |
| LLM | **未加载**（省显存） |
| Loss | 对称 InfoNCE + 可学习温度 τ |
| Batch size | 32 |
| 学习率 | 1e-4 |
| Epochs | 5 |
| 显存占用 | ~14GB |

**结果**：R@1=34.5%, R@5=71.6%, RMR-Score=0.524

### Stage 2: 多模态对比学习

**目标**：加入文本理解能力，让 LLM 融合"图+文"后的 query 向量也能检索贴图

```
路径 A (视觉，与 Stage 1 相同):
  渲染图 ──→ ViT ──→ ContrastiveHead ──→ img_emb
  贴图   ──→ ViT ──→ ContrastiveHead ──→ tex_emb
  L_visual = SymInfoNCE(img_emb, tex_emb)

路径 B (多模态，新增):
  渲染图 + 文本描述 ──→ ViT ──→ MLP1 ──→ Qwen2 LLM ──→ RetrievalHead ──→ query_emb
  贴图              ──→ ViT ──→ ContrastiveHead ──→ tex_emb
  L_multimodal = SymInfoNCE(query_emb, tex_emb)

L_total = α · L_visual + β · L_multimodal
```

| 配置 | 值 |
|------|-----|
| 训练数据 | 17,612 三元组 (渲染图, 贴图, 文本描述) |
| ViT | **冻结**（使用 Stage 1 权重） |
| ContrastiveHead | **冻结**（锚定共享空间） |
| MLP1 bridge | **解冻** |
| LLM (Qwen2) | **全参数/LoRA 微调** (r=64, alpha=128) |
| RetrievalHead | **从零训练** |
| Loss 权重 | α=1.0, β=1.0 |
| Effective batch | 32 (bs=8 × grad_accum=4) |
| 学习率 | 5e-5 |
| Epochs | 5 |
| 可训练参数 | 40.2M / 975M (4.1%) |
| 显存占用 | ~12GB |

## 模型组件详解

| 组件 | 参数量 | 输入→输出 | Stage 1 | Stage 2 |
|------|--------|----------|---------|---------|
| InternViT-300M | 300M | (B,3,448,448) → (B,1025,1024) | 解冻微调 | 冻结 |
| ContrastiveHead | ~525K | (B,1024) → (B,256) L2归一化 | 从零训练 | 冻结 |
| MLP1 bridge | ~4M | (B,N,1024) → (B,N,896) | 未加载 | 解冻 |
| Qwen2 LLM | 558M | (B,seq,896) → (B,seq,896) | 未加载 | LoRA (35M 可训练) |
| RetrievalHead | ~230K | (B,896) → (B,256) L2归一化 | 不存在 | 从零训练 |
| contrastive_log_temp | 1 | 标量，τ=exp(φ)，初始 τ=0.07 | 从零训练 | 继续训练 |

### ContrastiveHead vs RetrievalHead

```
ContrastiveHead (接 ViT CLS token):
  Linear(1024, 1024) → ReLU → Linear(1024, 256) → L2 Normalize
  输入: 只有视觉信息
  能力: "这张图看起来像什么材质"

RetrievalHead (接 LLM 最后一层 hidden state):
  Linear(896, 448) → ReLU → Linear(448, 256) → L2 Normalize
  输入: 图像 + 文本的融合理解
  能力: "用户说的'比这个粗糙一点的木头'对应哪种材质"
```

两个 head 输出到**同一个 256 维空间**，因此：
- img_emb 可以和 tex_emb 比较（纯视觉检索）
- query_emb 也可以和 tex_emb 比较（对话式检索）
- img_emb 和 query_emb 也可以互相比较

## Loss 详解

### Stage 1: 对称 InfoNCE

```python
# batch 内每对 (img_i, tex_i) 是正对，其余是负对
logits = img_emb @ tex_emb.T / τ   # (B, B) 相似度矩阵
labels = [0, 1, 2, ..., B-1]       # 对角线是正确答案

loss = ( CrossEntropy(logits, labels)      # 图→贴图方向
       + CrossEntropy(logits.T, labels)    # 贴图→图方向
       ) / 2
```

### Stage 2: 双路联合 Loss

```python
# 路径 A: 视觉对比 (和 Stage 1 一样，但 ViT 冻结)
img_emb = get_cls_embedding(rendered_img)        # ViT → ContrastiveHead
tex_emb = get_cls_embedding(texture_img)         # ViT → ContrastiveHead
L_visual = SymInfoNCE(img_emb, tex_emb)

# 路径 B: 多模态对比 (新增)
query_emb = get_retrieval_embedding(             # ViT → MLP → LLM → RetrievalHead
    rendered_img, text_input_ids, attention_mask
)
L_multimodal = SymInfoNCE(query_emb, tex_emb)    # 复用 tex_emb

# 总 Loss
L_total = α * L_visual + β * L_multimodal
```

**为什么路径 A 仍然保留？**
- 锚定共享 embedding 空间：如果只训路径 B，tex_emb 的空间可能漂移
- 保持纯视觉检索能力不退化

### 温度参数 τ

```
τ = exp(φ)，φ 是可学习标量，初始 φ = log(0.07) ≈ -2.66

τ 小 (0.07): softmax 尖锐 → 模型"自信"，只关注最相似的 → exploitation
τ 大 (1.0):  softmax 平坦 → 模型"犹豫"，每个候选都考虑  → exploration

训练策略: 前 2 epoch 冻结 τ（防止早期崩溃），之后和模型一起学习
每步 clamp 到 [e^{-4.6}, e^{2.3}] 防止极端值
```

## 推理流程

### 模式 1: 纯视觉检索（不需要 LLM）

```python
# 编码：只用 ViT + ContrastiveHead
img_emb = model.get_cls_embedding(rendered_image)   # (1, 256)

# 检索：和预计算的贴图库 embedding 比较
similarity = img_emb @ gallery_embeddings.T          # (1, N)
top_k = similarity.argsort(descending=True)[:5]
```

### 模式 2: 对话式检索（需要完整模型）

```python
# 编码：ViT + MLP + LLM + RetrievalHead
query_emb = model.get_retrieval_embedding(
    pixel_values=rendered_image,
    input_ids=tokenizer("比这个更粗糙的木材"),
    attention_mask=...
)  # (1, 256)

# 检索：同样的贴图库
similarity = query_emb @ gallery_embeddings.T        # (1, N)
top_k = similarity.argsort(descending=True)[:5]
```

### 贴图库索引（离线预计算）

```python
# 所有贴图只走 ViT + ContrastiveHead
for texture in texture_library:
    tex_emb = model.get_cls_embedding(texture)
    faiss_index.add(tex_emb)    # 存入 FAISS IndexFlatIP
```

## 评估指标

### 标准指标

| 指标 | 含义 |
|------|------|
| R@K | top-K 里找到正确答案的比例 |
| MRR | 正确答案排名倒数的均值，越接近 1 越好 |
| MedR | rank 中位数，越小越好，理想值=1 |
| nDCG@10 | 归一化折扣累积增益 |
| MAP@10 | 平均精度 |

### RMR-Score（我们提出的综合指标）

```
RMR = (w1 · R@1 + w2 · R@5 + w3 · MRR) / (w1 + w2 + w3)
```

融合精准度 (R@1)、覆盖度 (R@5)、排序质量 (MRR) 三个维度。
解决跨模态检索评估指标碎片化的问题——现有工作要么只看 R@K，要么用 rSum 简单加总。

### Stage 1 Benchmark 结果（gallery=1,956）

| Method | R@1 | R@5 | R@10 | MRR | MedR | nDCG@10 | RMR |
|--------|-----|-----|------|-----|------|---------|-----|
| Random | 0.1 | 0.3 | 0.5 | .004 | 978 | .004 | .002 |
| CLIP ViT-B/32 | 1.2 | 3.1 | 4.7 | .026 | 500 | .026 | .023 |
| CLIP ViT-L/14 | 1.8 | 4.5 | 7.0 | .037 | 473 | .040 | .033 |
| InternVL2-1B (zero-shot) | 0.3 | 1.1 | 1.5 | .011 | 722 | .008 | .008 |
| Ours (frozen ViT) | 11.0 | 31.0 | 43.3 | .216 | 14 | .253 | .212 |
| **Ours (unfrozen ViT)** | **34.5** | **71.6** | **81.4** | **.511** | **2** | **.579** | **.524** |

## 项目性质与贡献定位（诚实分析）

> 本节用于面试和论文 framing 时如何客观介绍项目，避免过度包装。

### 技术栈剖析：本质是什么？

把项目剥光了看，技术栈完全由**已有方法**组合而成：

| 组件 | 来源 | 年份 |
|---|---|---|
| 预训练 ViT + 投影头 | CLIP / SimCLR | 2020-2021 |
| ViT 微调（end-to-end） | CLIP fine-tune | 2021 起 |
| LoRA on LLM | Hu et al. | 2022 |
| 对称 InfoNCE loss | CLIP | 2021 |
| 联合 loss = α·L1 + β·L2 | 加权求和 | 通用 |
| FAISS IndexFlatIP | Meta | 2017 |

**没有任何一个组件是 2024 年之后才发明的新东西**。如果送 ICLR/CVPR review，第一个问题必然是 "What's the technical novelty?"

### 不是 SFT，但很接近：是"contrastive retrieval fine-tuning"

容易被误认为是"大号 SFT"，但实际上 loss 完全不同：

| 类型 | Loss | 训练目标 |
|---|---|---|
| SFT (Supervised Fine-Tuning) | `CE(logits, target_tokens)` | 让模型生成正确 token |
| **本项目（对比检索微调）** | `SymInfoNCE(emb_a, emb_b)` | 让两个向量可比较 |

更准确的工业切口名称：

> **"Contrastive retrieval fine-tuning of a VLM, with auxiliary projection heads."**
> （用大白话：VLM 上加两个召回头做对比训练）

### 真·小创新（论文里可写）

1. **LLM hidden state 用作 retrieval head 的输入**
   - 主流图文检索（CLIP / BLIP / SigLIP）从来不用 generative LLM 做 query encoder
   - 它们用的是 contrastive text encoder
   - 我们用 Qwen2 (auto-regressive LLM) 的最后 hidden state → 算 1 个微小 novelty

2. **Frozen ContrastiveHead 锚定共享空间**
   - Stage 2 把 Stage 1 的 ContrastiveHead 冻结，强迫 RetrievalHead 对齐到固定空间
   - 这样 gallery 不用重新编码，可直接复用 Stage 1 的索引
   - 半个 novelty（技术不深，但实用且工程上有意义）

3. **应用场景新：render → PBR texture**
   - 文献里几乎没人做"3D 渲染图 → 材质贴图"检索
   - 这是 **application novelty**，不是 method novelty
   - 可写成 "new task / dataset" 角度

### 假·大创新（不要这么吹）

- "We propose a novel contrastive learning framework..." → 你只用了 InfoNCE
- "Our innovative dual-path architecture..." → 加权求和不是创新
- "We introduce a parameter-efficient fine-tuning..." → 那是 LoRA

### 投稿期望（诚实评估）

| 会议级别 | 接收概率 | 怎么写 |
|---|---|---|
| NeurIPS / ICLR / CVPR | <5% | 除非加非常新的元素 |
| ECCV / ICCV / AAAI | ~10% | 应用 novelty 拉满 + 完整 benchmark |
| WACV / BMVC / ACCV | ~30% | 完整故事 + 实验充分 |
| **Application track / workshop** | ~60% | "把 VLM 应用到游戏材质检索" 叙事 ✅ |
| arXiv preprint + 工程 demo | 100% | 价值在工程而不在论文 |

### 推荐的论文 framing：System Paper 而非 Method Paper

> **"This is a system paper, not a method paper."**

把论文 framing 调整为 "a complete material retrieval system" 而非 "a novel method"。Application / industry track 反而更适合 — reviewer 不会拿 method novelty 卡你，只看：
- 系统是否完整
- 指标是否扎实
- 是否解决真问题

按这个 framing，可以分以下几个真实贡献点：

1. **Complete pipeline**: 数据生成 → 双阶段训练 → FAISS 部署 → FastAPI 服务
2. **Real production metrics**: R@1=44.8% 的真实可用指标，已集成到 RAGenLah
3. **Empirical findings**:
   - ViT 微调对 render-texture 域是必需的（+24pp 实证）
   - LoRA r=64 是性能/参数甜点
   - 文本描述用 LLM 自生成可行，不需要人工标注
4. **Internal infrastructure value**: MIAO 实验室的材质检索基础设施

### 如果想让它"更高级"（可选改进方向）

如果要往会议投，需要加 1-2 个"看起来新"的元素：

1. **Hard Negative Mining**
   - 普通 InfoNCE 用 batch 内随机负样本
   - 改成"材质类别 × 颜色 × 粗糙度"三维聚类的 hard negative
   - 多一节 "Material-Aware Hard Negative Sampling"

2. **Material Property Disentanglement Loss**
   - 把 256-d 拆成 64+64+64+64，分别对应 albedo/normal/roughness/metallic
   - 加正则项让每段只编码对应属性
   - 多一节 "interpretable embedding space"

3. **Zero-shot 泛化实验**
   - PBR 数据集训，BlenderKit / Substance 上 zero-shot 测
   - 跨数据集泛化作为强 evidence

4. **User study**
   - 找 10 个 3D artist 对比 "我们 vs CLIP" 找材质的速度
   - 用户偏好 + 时间数据 → application paper 的杀器

### 一句话定位

**"大号 SFT + 两个召回头 + 联合 loss" 是工程上准确的描述**。要硬塞 method novelty 比较吃力，但作为 **system / application paper** 完全够格。论文叙事建议从"提出新方法"切换为"构建完整系统并解决新任务"，reviewer 的预期就对了。

---

## 数据

| 项目 | 路径 |
|------|------|
| 渲染数据 | `/home/ubuntu/alan_repository/texture_material/output/` |
| Stage 1 训练集 | `dataset/train.jsonl` (17,612 pairs) |
| Stage 1 验证集 | `dataset/val.jsonl` (1,956 pairs) |
| Stage 2 训练集 | `dataset/train_with_desc.jsonl` (17,612 triplets) |
| Stage 2 验证集 | `dataset/val_with_desc.jsonl` (1,956 triplets) |
| 基础模型 | `/home/ubuntu/alan_repository/models/InternVL2-1B` |
| Stage 1 权重 | `contrastive_ckpt_unfrozen/best_contrastive_weights.pt` |
| Stage 2 权重 | `multimodal_contrastive_ckpt/` (训练中) |

数据格式：
```json
// Stage 1
{"id": "2230", "image": "2230/fig.png", "texture": "2230/texture.png"}

// Stage 2 (新增 description 字段)
{"id": "2230", "image": "2230/fig.png", "texture": "2230/texture.png",
 "description": "a combination of fabric and leather, with a smooth, slightly glossy surface..."}
```

文本描述由 InternVL2-1B 自身生成（prompt: "Describe the material texture in this image in one paragraph"）。

## 训练命令

### Stage 1: 视觉对比学习

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

### Stage 2: 多模态对比学习

```bash
python internvl/train/internvl_multimodal_contrastive.py \
    --model_name_or_path /home/ubuntu/alan_repository/models/InternVL2-1B \
    --contrastive_ckpt /home/ubuntu/alan_repository/texture_material/contrastive_ckpt_unfrozen/best_contrastive_weights.pt \
    --train_jsonl /home/ubuntu/alan_repository/texture_material/dataset/train_with_desc.jsonl \
    --val_jsonl   /home/ubuntu/alan_repository/texture_material/dataset/val_with_desc.jsonl \
    --image_root  /home/ubuntu/alan_repository/texture_material/output \
    --output_dir  /home/ubuntu/alan_repository/texture_material/multimodal_contrastive_ckpt \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --lora_r 64 --lora_alpha 128 \
    --alpha 1.0 --beta 1.0
```

### Benchmark 评测

```bash
python internvl/train/benchmark_all.py \
    --val_jsonl /home/ubuntu/alan_repository/texture_material/dataset/val.jsonl \
    --image_root /home/ubuntu/alan_repository/texture_material/output \
    --internvl_base /home/ubuntu/alan_repository/models/InternVL2-1B \
    --ckpt_unfrozen /home/ubuntu/alan_repository/texture_material/contrastive_ckpt_unfrozen/best_contrastive_weights.pt
```

## 文件结构

```
InternVL/internvl_chat/internvl/
├── model/internvl_chat/
│   └── modeling_internvl_chat.py         # 模型定义
│       ├── ContrastiveHead               #   ViT CLS → 256维 (视觉检索)
│       ├── RetrievalHead                 #   LLM hidden → 256维 (对话检索)
│       ├── get_cls_embedding()           #   ViT → ContrastiveHead → emb
│       ├── get_retrieval_embedding()     #   ViT+LLM → RetrievalHead → emb
│       ├── contrastive_forward()         #   Stage 1 loss
│       └── multimodal_contrastive_forward()  # Stage 2 双路 loss
├── train/
│   ├── internvl_contrastive_finetune.py      # Stage 1 训练脚本
│   ├── dataset_contrastive.py                # Stage 1 数据集 (image pairs)
│   ├── internvl_multimodal_contrastive.py    # Stage 2 训练脚本
│   ├── dataset_multimodal_contrastive.py     # Stage 2 数据集 (image+text triplets)
│   ├── generate_texture_descriptions.py      # 用 InternVL 生成材质文本描述
│   ├── benchmark_all.py                      # 全 baseline 统一评测
│   ├── compute_rmr_score.py                  # RMR-Score 计算
│   ├── eval_retrieval.py                     # Recall@K 评估
│   └── eval_retrieval_cases.py               # Case 可视化
└── paper/
    ├── main.tex                              # 论文 LaTeX
    └── main.pdf                              # 编译后 PDF
```

## 资源需求

| 方案 | 可训练参数 | 显存需求 | L20 (46GB) |
|------|-----------|---------|------------|
| Stage 1 (ViT-only) | 300M | ~14 GB | ✅ |
| Stage 2 (LoRA, 当前方案) | 40M (4.1%) | ~12 GB | ✅ |
| Stage 2 (全参数微调) | 975M (100%) | ~20-22 GB | ✅ |
| InternVL2-8B 全参数 | ~8B | ~80 GB | ❌ 需要 2× GPU |

## 环境

- GPU: NVIDIA L20 (46GB VRAM)
- Conda: `qwen` 环境
- Python: 3.10
- 依赖: `torch 2.5.1+cu121`, `transformers 4.48.3`, `peft 0.18.1`, `timm`, `einops`, `faiss-cpu`
