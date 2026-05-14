# Training Guide — JEE-LLM Fine-Tuning

## Overview

This guide walks through the complete 3-stage training pipeline from scratch.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.11 |
| CUDA | 11.8 | 12.1 |
| VRAM (single GPU) | 16 GB | 24 GB (4090/A10) |
| VRAM (multi-GPU) | 4×16 GB | 4×80 GB (A100) |
| RAM | 32 GB | 64 GB |
| Storage | 100 GB | 500 GB |

---

## Stage 1 — Supervised Fine-Tuning (QLoRA)

### What it does
Trains the base model on ~50K JEE problems with Chain-of-Thought solutions using 4-bit quantization + LoRA adapters.

### Expected duration
- **1× RTX 4090 (24GB)**: ~8 hours (3 epochs, 50K examples)
- **4× A100 (80GB)**: ~2 hours

### Command
```bash
make train-sft
# OR
bash scripts/train_qlora.sh --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Key hyperparameters to tune
| Param | Default | Notes |
|-------|---------|-------|
| `lora_r` | 64 | Higher = more capacity, more VRAM |
| `lora_alpha` | 128 | Usually 2× lora_r |
| `learning_rate` | 2e-4 | Reduce to 1e-4 if loss is unstable |
| `per_device_train_batch_size` | 4 | Reduce to 2 if OOM |
| `gradient_accumulation_steps` | 8 | Effective batch = 4×8=32 |
| `max_seq_length` | 4096 | Reduce to 2048 to save memory |

### Monitoring
```bash
# WandB dashboard (auto-opens if configured)
wandb login
# Then check: https://wandb.ai/YOUR_USERNAME/jee-llm-finetuning
```

**Expected training loss curve:**
- Steps 0–200: Loss ~2.5–3.0 (model learning format)
- Steps 200–1000: Loss ~1.5–2.0 (learning reasoning patterns)
- Steps 1000+: Loss ~0.8–1.2 (refining CoT quality)

---

## Stage 2 — Preference Alignment

### 2a. Reward Model Training (optional for PPO)
```bash
python src/training/train_reward_model.py --config configs/reward_model.yaml
```

### 2b. GRPO Training (recommended)
GRPO (Group Relative Policy Optimization) — no separate reward model needed.
```bash
make train-grpo
```

### 2c. DPO Training (alternative/complement to GRPO)
```bash
make train-dpo
```

**Recommended order**: Run SFT → GRPO → DPO for best results.

---

## Stage 3 — SPIN Self-Play

```bash
python src/training/train_spin.py --config configs/spin.yaml
```

---

## Memory Optimization Tips

### For 16GB VRAM (e.g., RTX 3090/4080)
```yaml
# In configs/sft_qlora.yaml:
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  max_seq_length: 2048
  gradient_checkpointing: true
lora:
  r: 32          # Reduce from 64
  lora_alpha: 64
```

### For CPU offloading (8GB VRAM)
```bash
# Use DeepSpeed with CPU offload
bash scripts/train_distributed.sh --gpus 1
# (ZeRO-3 config enables CPU optimizer offload)
```

---

## After Training

### Merge adapters
```bash
make merge
```

### Evaluate
```bash
make evaluate MODEL_PATH=checkpoints/jee-llm-v1-merged
```

### Start daily-use interface
```bash
make serve-gradio MODEL_PATH=checkpoints/jee-llm-v1-merged
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce batch size or max_seq_length |
| `Flash attention not installed` | `pip install flash-attn --no-build-isolation` |
| `RuntimeError: Expected all tensors to be on the same device` | Set `CUDA_VISIBLE_DEVICES=0` |
| Wandb not logging | Run `wandb login` first |
| Model generation is gibberish | Ensure chat template is applied correctly |

---

## Model Selection Guide

| Base Model | VRAM Required | JEEBench Score | Notes |
|-----------|--------------|----------------|-------|
| LLaMA-3-8B | 16GB | ~58% | Best for single GPU |
| Mistral-7B-v0.3 | 16GB | ~55% | Faster inference |
| Qwen2-7B | 16GB | ~57% | Strong at math |
| LLaMA-3-70B | 4×80GB | ~68% | Best overall |
| Mistral-22B | 2×80GB | ~63% | Good balance |

---

*See also: [DATASET_GUIDE.md](DATASET_GUIDE.md) | [API_REFERENCE.md](API_REFERENCE.md)*
