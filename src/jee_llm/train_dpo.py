"""Optional offline alternative to GRPO: DPO on self-generated pairs.

Pairs come from scripts/build_preference_pairs.py (sample k solutions from the
SFT model, keep one verified-correct and one wrong per question). This is the
"rejection sampling + DPO" baseline that GRPO gets compared against.

    python -m jee_llm.train_dpo configs/dpo.yaml
"""

from __future__ import annotations

import logging
import sys

from .config import parse_cli
from .data import read_jsonl

log = logging.getLogger("jee_llm.dpo")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    cfg = parse_cli(sys.argv[1:] if argv is None else argv)

    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    from .modeling import describe_trainable, load_model, load_tokenizer, lora_config

    rows = [
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in read_jsonl(cfg["data"]["pairs_file"])
    ]
    if not rows:
        raise SystemExit("pairs file is empty")
    train_ds = Dataset.from_list(rows)
    log.info("%d preference pairs", len(train_ds))

    tokenizer = load_tokenizer(cfg["model"]["name"])
    model = load_model(cfg["model"], cfg.get("quantization"))
    args = DPOConfig(output_dir=cfg["output_dir"], seed=cfg.get("seed", 42), **cfg.get("training", {}))

    # with a peft_config and no ref_model, TRL uses the frozen base (adapters off) as reference
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=lora_config(cfg.get("lora")),
    )
    log.info(describe_trainable(trainer.model))
    trainer.train()
    trainer.save_model(f"{cfg['output_dir']}/final")
    tokenizer.save_pretrained(f"{cfg['output_dir']}/final")


if __name__ == "__main__":
    main()
