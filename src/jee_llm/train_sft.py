"""Stage 1: supervised fine-tuning on worked solutions (QLoRA by default).

    python -m jee_llm.train_sft configs/sft_qlora.yaml
    accelerate launch -m jee_llm.train_sft configs/sft_qlora.yaml   # multi-GPU
"""

from __future__ import annotations

import logging
import random
import sys

from .config import parse_cli
from .data import load_problems
from .prompts import to_sft_example

log = logging.getLogger("jee_llm.sft")


def build_dataset(files: list[str], max_samples: int | None, seed: int):
    from datasets import Dataset

    rows = []
    for path in files:
        problems = [p for p in load_problems(path) if p.solution]
        log.info("%s: %d problems with solutions", path, len(problems))
        rows.extend(to_sft_example(p) for p in problems)
    random.Random(seed).shuffle(rows)
    if max_samples:
        rows = rows[:max_samples]
    if not rows:
        raise SystemExit("no training rows; check data.train_files")
    return Dataset.from_list(rows)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    cfg = parse_cli(sys.argv[1:] if argv is None else argv)

    from trl import SFTConfig, SFTTrainer

    from .modeling import describe_trainable, load_model, load_tokenizer, lora_config

    seed = cfg.get("seed", 42)
    data_cfg = cfg["data"]
    train_ds = build_dataset(data_cfg["train_files"], data_cfg.get("max_train_samples"), seed)
    eval_ds = build_dataset([data_cfg["eval_file"]], None, seed) if data_cfg.get("eval_file") else None
    log.info("train rows: %d, eval rows: %s", len(train_ds), len(eval_ds) if eval_ds else "-")

    tokenizer = load_tokenizer(cfg["model"]["name"])
    model = load_model(cfg["model"], cfg.get("quantization"))

    args = SFTConfig(output_dir=cfg["output_dir"], seed=seed, **cfg.get("training", {}))
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config(cfg.get("lora")),
    )
    log.info(describe_trainable(trainer.model))
    trainer.train()
    trainer.save_model(f"{cfg['output_dir']}/final")
    tokenizer.save_pretrained(f"{cfg['output_dir']}/final")


if __name__ == "__main__":
    main()
