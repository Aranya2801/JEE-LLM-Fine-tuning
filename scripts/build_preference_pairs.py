"""Sample k solutions per question from a model and keep (correct, wrong) pairs for DPO.

    python scripts/build_preference_pairs.py --model checkpoints/sft-merged \
        --data dataset/train_rl.jsonl --k 8 --out dataset/dpo_pairs.jsonl --limit 5000

Questions where every sample is right (too easy) or every sample is wrong
(no positive to learn from) produce no pair. The yield is logged, since it
says a lot about where the SFT model is.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jee_llm.answers import extract_final_answer, is_correct  # noqa: E402
from jee_llm.data import load_problems, write_jsonl  # noqa: E402
from jee_llm.prompts import build_messages  # noqa: E402

log = logging.getLogger("pairs")


def make_pairs(problems, completions, rng, max_pairs_per_question=1):
    pairs, stats = [], {"all_right": 0, "all_wrong": 0, "used": 0}
    for p, samples in zip(problems, completions, strict=True):
        verdict = [is_correct(extract_final_answer(s), p.answer, p.question_type) for s in samples]
        good = [s for s, v in zip(samples, verdict, strict=True) if v]
        bad = [s for s, v in zip(samples, verdict, strict=True) if not v]
        if not bad:
            stats["all_right"] += 1
            continue
        if not good:
            stats["all_wrong"] += 1
            continue
        stats["used"] += 1
        for _ in range(min(max_pairs_per_question, len(good), len(bad))):
            pairs.append({
                "id": p.id,
                "prompt": build_messages(p),
                "chosen": [{"role": "assistant", "content": rng.choice(good)}],
                "rejected": [{"role": "assistant", "content": rng.choice(bad)}],
            })
    return pairs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pairs-per-question", type=int, default=1)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rng = random.Random(args.seed)
    problems = load_problems(args.data)
    rng.shuffle(problems)
    problems = problems[: args.limit] if args.limit else problems

    from jee_llm.generation import Generator

    gen = Generator(args.model, adapter=args.adapter, backend=args.backend)
    completions = gen.generate([build_messages(p) for p in problems], n=args.k, temperature=args.temperature,
                               max_new_tokens=args.max_new_tokens, seed=args.seed)
    pairs, stats = make_pairs(problems, completions, rng, args.pairs_per_question)
    write_jsonl(args.out, pairs)
    log.info("%d questions -> %d pairs (%s)", len(problems), len(pairs), stats)


if __name__ == "__main__":
    main()
