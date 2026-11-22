"""Merge a LoRA adapter into its base model and save a plain HF checkpoint.

Needed before GRPO/DPO (they start from the merged SFT model) and before
GGUF export.

    python scripts/merge_adapters.py --base Qwen/Qwen2.5-7B-Instruct \
        --adapter checkpoints/sft/final --out checkpoints/sft-merged

Merge in bf16/fp16, *not* on top of the 4-bit base: merging into quantized
weights and then dequantizing loses accuracy.
"""

import argparse
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("merge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--push-to-hub", default=None, metavar="REPO_ID")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    dtype = getattr(torch, args.dtype)
    log.info("loading base %s", args.base)
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(args.adapter)

    model.save_pretrained(args.out, max_shard_size="5GB")
    tok.save_pretrained(args.out)
    log.info("saved merged model to %s", args.out)

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub, private=True)
        tok.push_to_hub(args.push_to_hub, private=True)
        log.info("pushed to %s (private)", args.push_to_hub)


if __name__ == "__main__":
    main()
