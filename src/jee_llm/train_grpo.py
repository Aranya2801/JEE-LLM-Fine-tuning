"""Stage 2: GRPO with verifiable rewards.

No reward model: the reward is computed by checking the extracted answer
against the gold answer (see rewards.py). Which correctness reward to use is
set in the config:

    reward:
      funcs: [jee_marks, format]
      weights: [1.0, 0.1]

    python -m jee_llm.train_grpo configs/grpo.yaml
"""

from __future__ import annotations

import logging
import random
import sys

from .config import parse_cli
from .data import load_problems
from .prompts import to_rl_example
from .rewards import REWARD_FUNCS

log = logging.getLogger("jee_llm.grpo")


def build_dataset(files: list[str], allow_skip: bool, max_samples: int | None, seed: int):
    from datasets import Dataset

    rows = []
    for path in files:
        problems = load_problems(path)
        rows.extend(to_rl_example(p, allow_skip=allow_skip) for p in problems)
        log.info("%s: %d prompts", path, len(problems))
    random.Random(seed).shuffle(rows)
    if max_samples:
        rows = rows[:max_samples]
    return Dataset.from_list(rows)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    cfg = parse_cli(sys.argv[1:] if argv is None else argv)

    from trl import GRPOConfig, GRPOTrainer

    from .modeling import describe_trainable, load_model, load_tokenizer, lora_config

    seed = cfg.get("seed", 42)
    data_cfg = cfg["data"]
    allow_skip = bool(data_cfg.get("allow_skip", False))
    train_ds = build_dataset(data_cfg["train_files"], allow_skip, data_cfg.get("max_train_samples"), seed)

    reward_cfg = cfg.get("reward", {"funcs": ["correctness", "format"], "weights": [1.0, 0.1]})
    unknown = set(reward_cfg["funcs"]) - set(REWARD_FUNCS)
    if unknown:
        raise SystemExit(f"unknown reward funcs {unknown}, choose from {sorted(REWARD_FUNCS)}")
    if allow_skip and "jee_marks" not in reward_cfg["funcs"]:
        # with binary correctness, SKIP always scores 0 = same as wrong, so there's nothing to learn
        log.warning("allow_skip=true but jee_marks reward is off; skipping will never be rewarded")
    reward_funcs = [REWARD_FUNCS[name] for name in reward_cfg["funcs"]]

    tokenizer = load_tokenizer(cfg["model"]["name"])
    model = load_model(cfg["model"], cfg.get("quantization"))

    training = dict(cfg.get("training", {}))
    training.setdefault("reward_weights", reward_cfg.get("weights"))
    args = GRPOConfig(output_dir=cfg["output_dir"], seed=seed, **training)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
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
