# InternVL Training Code Map

This file summarizes where training code is located under `InternVL/` and whether each branch provides an end-to-end open-source training path.

## Directory Overview

```text
InternVL/
├── internvl_chat/              # Main multimodal chat training (InternVL 1.2/1.5/2.0/2.5/3.0)
│   ├── internvl/train/         # Python training entrypoints
│   ├── shell/                  # Stage scripts (stage1/stage1.5/stage2/2nd_finetune/MPO)
│   └── tools/                  # Data conversion and reasoning data pipeline
├── internvl_chat_gpt_oss/      # InternVL3.5 GPT-OSS training code (stage0-3 scripts)
├── internvl_chat_llava/        # LLaVA-style pretrain/finetune code and scripts
├── internvl_g/                 # Stage-2 retrieval finetuning code
├── classification/             # Image classification linear-probing training
└── segmentation/               # Segmentation training based on MMSeg
```

## Completeness Status

### 1) `internvl_chat/`
- Status: **Training code is complete for open-sourced stages**.
- Includes:
  - Pretrain/finetune entrypoints in `internvl/train/`
  - Multi-version stage scripts in `shell/` (`stage1`, `stage1.5`, `stage2`, `2nd_finetune`, `mpo`)
  - Dataset processing pipeline in `tools/reasoning_data_pipeline/`
- Note:
  - Data and pretrained checkpoints are external dependencies.
  - Example shell scripts often use placeholder paths (for example `--meta_path "./path/to/.../mixture.json"`).

### 2) `internvl_chat_gpt_oss/`
- Status: **Contains open-sourced training code for InternVL3.5 GPT-OSS path**.
- Includes stage scripts under `shell/internvl3_5_gpt_oss/` and training code under `internvl/train/`.
- Note:
  - Requires external environment package, data, and model assets.

### 3) `internvl_chat_llava/`
- Status: **Contains pretrain/finetune code and runnable scripts**.
- Includes `llava/train/*.py` and script sets under `scripts/` and `scripts_internvl/`.

### 4) `internvl_g/`
- Status: **Stage-2 retrieval finetuning code is available**.
- Note:
  - The README states that generative pre-training code is not planned for release.

### 5) `classification/`
- Status: **Linear-probing training code is available**.
- Includes `main.py`, configs, and `train_in1k.sh`.
- Note:
  - Requires external datasets and pretrained weights.

### 6) `segmentation/`
- Status: **Partial training support in this repo branch**.
- Note:
  - The README explicitly says this open-source branch does not include DeepSpeed integration in MMSeg and currently supports linear/head tuning, not full-parameter training.

## Practical Conclusion

For **InternVL multimodal chat models** (`internvl_chat/` and `internvl_chat_gpt_oss/`), the repository contains the main training code and stage scripts.  
To actually run end-to-end training, you still need to prepare external datasets, pretrained checkpoints, and appropriate distributed hardware/runtime.
