"""Build a tiny random Qwen2-style model + tokenizer for smoke tests.

Nothing is downloaded: the tokenizer is a byte-level BPE trained on the sample
problems, the model is randomly initialised (~1M params). Output is garbage,
the point is to run SFT -> GRPO -> eval end to end on a laptop CPU in a
couple of minutes and catch API breakage.

    python scripts/make_tiny_model.py --out /tmp/tiny-qwen
"""

import argparse
import json
from pathlib import Path

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{% if m['role'] == 'assistant' %}{% generation %}{{ m['content'] }}<|im_end|>{% endgeneration %}"
    "{% else %}{{ m['content'] }}<|im_end|>{% endif %}\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/tiny-qwen")
    ap.add_argument("--data", default="dataset/samples/jee_style_samples.jsonl")
    args = ap.parse_args()

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    texts = []
    for line in open(args.data, encoding="utf-8"):
        row = json.loads(line)
        texts += [row["question"], row["solution"], " ".join(row.get("options", {}).values())]

    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts * 4, trainers.BpeTrainer(
        vocab_size=600, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
        bos_token=None, unk_token=None,
    )
    hf_tok.chat_template = CHAT_TEMPLATE

    cfg = Qwen2Config(
        vocab_size=len(hf_tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
        eos_token_id=hf_tok.eos_token_id, pad_token_id=hf_tok.pad_token_id, tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(cfg)
    out = Path(args.out)
    model.save_pretrained(out)
    hf_tok.save_pretrained(out)
    print(f"saved tiny model ({sum(p.numel() for p in model.parameters()):,} params) to {out}")


if __name__ == "__main__":
    main()
