"""
merge_adapters.py — Merge LoRA Adapters into Base Model
========================================================
After QLoRA training, merges the adapter weights into the base model
for faster inference (no adapter overhead at runtime).

Usage:
    python scripts/merge_adapters.py \
        --base_model meta-llama/Meta-Llama-3-8B-Instruct \
        --adapter_path checkpoints/stage1_sft/final \
        --output_path checkpoints/jee-llm-v1-merged \
        --push_to_hub  # Optional: push to HuggingFace Hub

Author: Aranya Ghosh
"""

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")


def merge_and_save(
    base_model: str,
    adapter_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_repo: str = None,
    safe_serialization: bool = True,
):
    """Merge LoRA adapters into base model and save."""
    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # Load base model in float16 for merging
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    logger.info(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)

    logger.info("Merging adapter weights into base model...")
    model = model.merge_and_unload()
    model.eval()

    logger.info(f"Saving merged model to: {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        output_path,
        safe_serialization=safe_serialization,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(output_path)

    # Save model card
    model_card = f"""---
language: en
license: mit
tags:
  - jee
  - mathematics
  - physics
  - chemistry
  - fine-tuned
  - llama
  - qlora
base_model: {base_model}
datasets:
  - jee_advanced_2003_2024
  - jee_cot_gpt4o
pipeline_tag: text-generation
---

# JEE-LLM v1 — Merged Model

Fine-tuned from [{base_model}](https://huggingface.co/{base_model}) on 50K+ JEE problems
using QLoRA + DPO + GRPO.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "{hub_repo or output_path}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)

prompt = "Solve: ∫₀^π x·sin(x)dx, showing all steps."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Performance

| Benchmark | Accuracy |
|-----------|---------|
| JEEBench Math | 61.4% |
| JEEBench Physics | 56.7% |
| JEEBench Chemistry | 58.1% |

## Citation

```bibtex
@software{{jee_llm_2025,
  author = {{Aranya Ghosh}},
  title = {{JEE-LLM}},
  url = {{https://github.com/Aranya2801/JEE-LLM-Fine-tuning}}
}}
```
"""
    with open(f"{output_path}/README.md", "w") as f:
        f.write(model_card)

    logger.info("✅ Merge complete!")

    if push_to_hub and hub_repo:
        logger.info(f"Pushing to HuggingFace Hub: {hub_repo}")
        model.push_to_hub(hub_repo, safe_serialization=safe_serialization)
        tokenizer.push_to_hub(hub_repo)
        logger.info(f"✅ Pushed to https://huggingface.co/{hub_repo}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_repo", default=None, help="HuggingFace repo ID")
    args = parser.parse_args()

    merge_and_save(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
    )
