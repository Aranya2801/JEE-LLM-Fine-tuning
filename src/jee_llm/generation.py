"""Sampling completions, with vLLM if it's installed and plain transformers otherwise.

The HF path is slow but works anywhere (including CPU for smoke tests).
For real JEEBench runs with 8+ samples per question use vLLM.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class Generator:
    def __init__(self, model: str, adapter: str | None = None, backend: str = "auto", max_model_len: int = 8192):
        self.model_name = model
        self.adapter = adapter
        if backend == "auto":
            backend = "vllm" if _vllm_available() else "hf"
        self.backend = backend
        log.info("generation backend: %s", backend)
        if backend == "vllm":
            self._init_vllm(max_model_len)
        else:
            self._init_hf()

    # -- vllm -----------------------------------------------------------------
    def _init_vllm(self, max_model_len: int) -> None:
        from vllm import LLM

        self.llm = LLM(
            model=self.model_name,
            enable_lora=self.adapter is not None,
            max_lora_rank=64,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.9,
        )

    def _gen_vllm(self, conversations, n, temperature, top_p, max_new_tokens, seed):
        from vllm import SamplingParams

        params = SamplingParams(
            n=n, temperature=temperature, top_p=top_p if temperature > 0 else 1.0,
            max_tokens=max_new_tokens, seed=seed,
        )
        lora = None
        if self.adapter:
            from vllm.lora.request import LoRARequest

            lora = LoRARequest("adapter", 1, self.adapter)
        outs = self.llm.chat(conversations, params, lora_request=lora, use_tqdm=True)
        return [[o.text for o in out.outputs] for out in outs]

    # -- transformers -----------------------------------------------------------
    def _init_hf(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(self.adapter or self.model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
        if self.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()

    def _gen_hf(self, conversations, n, temperature, top_p, max_new_tokens, seed, batch_size=8):
        import torch

        torch.manual_seed(seed)
        results = []
        for i in range(0, len(conversations), batch_size):
            batch = conversations[i : i + batch_size]
            texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in batch]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            kwargs = dict(max_new_tokens=max_new_tokens, num_return_sequences=n, pad_token_id=self.tok.pad_token_id)
            if temperature > 0:
                kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                kwargs.update(do_sample=False)
            with torch.no_grad():
                out = self.model.generate(**enc, **kwargs)
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            decoded = self.tok.batch_decode(new_tokens, skip_special_tokens=True)
            for j in range(len(batch)):
                results.append(decoded[j * n : (j + 1) * n])
        return results

    # ---------------------------------------------------------------------------
    def generate(self, conversations, n=1, temperature=0.0, top_p=0.95, max_new_tokens=2048, seed=0):
        if temperature == 0 and n > 1:
            raise ValueError("n > 1 with temperature 0 just returns n copies of the greedy answer")
        fn = self._gen_vllm if self.backend == "vllm" else self._gen_hf
        return fn(conversations, n, temperature, top_p, max_new_tokens, seed)


def _vllm_available() -> bool:
    try:
        import torch
        import vllm  # noqa: F401

        return torch.cuda.is_available()
    except ImportError:
        return False
