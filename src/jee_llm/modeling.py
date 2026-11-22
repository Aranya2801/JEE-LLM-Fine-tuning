"""Model / tokenizer / LoRA setup shared by all training stages."""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "auto": "auto"}


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_cfg: dict, quant_cfg: dict | None = None):
    """Load a causal LM, optionally 4-bit (QLoRA).

    model_cfg keys: name, dtype, attn_implementation
    quant_cfg keys: load_in_4bit, quant_type, double_quant
    """
    from transformers import AutoModelForCausalLM

    dtype = _DTYPES[model_cfg.get("dtype", "bfloat16")]
    kwargs = {"dtype": dtype}
    if model_cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = model_cfg["attn_implementation"]

    if quant_cfg and quant_cfg.get("load_in_4bit"):
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit loading needs a CUDA GPU (bitsandbytes); set quantization.load_in_4bit=false")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_cfg.get("double_quant", True),
            bnb_4bit_compute_dtype=dtype if dtype != "auto" else torch.bfloat16,
        )
        # one process per GPU under accelerate/torchrun
        import os

        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", 0))}

    log.info("loading %s (%s)", model_cfg["name"], {k: v for k, v in kwargs.items() if k != "quantization_config"})
    model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], **kwargs)

    if quant_cfg and quant_cfg.get("load_in_4bit"):
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def lora_config(cfg: dict | None):
    if not cfg:
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.get("r", 16),
        lora_alpha=cfg.get("alpha", 32),
        lora_dropout=cfg.get("dropout", 0.05),
        target_modules=cfg.get("target_modules", "all-linear"),
        bias="none",
        task_type="CAUSAL_LM",
    )


def describe_trainable(model) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return f"trainable params: {trainable:,} / {total:,} ({100 * trainable / max(total, 1):.2f}%)"
