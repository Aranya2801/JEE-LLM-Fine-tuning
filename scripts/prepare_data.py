"""Download sources, convert to the Problem schema, decontaminate, split.

    python scripts/prepare_data.py --out dataset/
    python scripts/prepare_data.py --out dataset/ --extra my_scraped_jee_mains.jsonl --numina 50000

Writes:
    dataset/jeebench.jsonl     test set, never trained on (515 problems)
    dataset/train_sft.jsonl    problems with worked solutions
    dataset/train_rl.jsonl     problems with checkable answers (MCQ / integer / numerical / short math)
    dataset/val.jsonl          small held-out slice of the training pool
    dataset/contamination.json what got dropped and why

--extra files must already be in the Problem schema (see src/jee_llm/data.py).
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jee_llm.answers import parse_number  # noqa: E402
from jee_llm.contamination import NgramIndex  # noqa: E402
from jee_llm.data import Problem, from_gsm8k, from_jeebench, from_math, load_problems, write_jsonl  # noqa: E402

log = logging.getLogger("prepare")

MATH_CONFIGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]


def load_jeebench() -> list[Problem]:
    from datasets import load_dataset

    ds = load_dataset("daman1209arora/jeebench", split="test")
    return [from_jeebench(r) for r in ds]


def load_math() -> list[Problem]:
    # the original hendrycks/competition_math repo was taken down; this mirror has the same data
    from datasets import load_dataset

    out = []
    for cfg in MATH_CONFIGS:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
        for r in ds:
            p = from_math(r, len(out))
            if p:
                out.append(p)
    return out


def load_gsm8k() -> list[Problem]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    return [from_gsm8k(r, i) for i, r in enumerate(ds)]


def load_numina(limit: int, seed: int) -> list[Problem]:
    """NuminaMath-CoT, olympiad / competition subsets only (closest to JEE difficulty)."""
    from datasets import load_dataset

    from jee_llm.answers import extract_boxed

    keep = {"olympiads", "amc_aime", "aops_forum", "cn_contest"}
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train").shuffle(seed=seed)
    out = []
    for r in ds:
        if r["source"] not in keep:
            continue
        ans = extract_boxed(r["solution"])
        if not ans:
            continue
        out.append(Problem(
            id=f"numina_{len(out):06d}", subject="math", question_type="open_ended", question=r["problem"],
            answer=ans, solution=r["solution"], source=f"NuminaMath/{r['source']}",
        ))
        if len(out) >= limit:
            break
    return out


def rl_friendly(p: Problem) -> bool:
    """Only keep RL prompts whose answers the verifier can actually check."""
    if p.question_type != "open_ended":
        return True
    return parse_number(p.answer) is not None  # numeric open-ended answers only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--extra", nargs="*", default=[], help="additional Problem-schema jsonl files")
    ap.add_argument("--numina", type=int, default=0, help="how many NuminaMath-CoT problems to add (0 = none)")
    ap.add_argument("--no-gsm8k", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--ngram", type=int, default=13)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out)

    test = load_jeebench()
    write_jsonl(out / "jeebench.jsonl", (p.to_dict() for p in test))
    log.info("jeebench: %d", len(test))

    pool: list[Problem] = []
    pool += load_math()
    if not args.no_gsm8k:
        pool += load_gsm8k()
    if args.numina:
        pool += load_numina(args.numina, args.seed)
    for path in args.extra:
        pool += load_problems(path)
    log.info("training pool before decontamination: %d", len(pool))

    # drop anything sharing a 13-gram with a JEEBench question
    index = NgramIndex(args.ngram)
    index.add_all(p.question for p in test)
    dropped, clean = [], []
    for p in pool:
        r = index.check(p.id, p.question)
        (dropped if r.n_overlap else clean).append((p, r))
    log.info("dropped %d training problems overlapping JEEBench", len(dropped))
    report = [{"id": p.id, "source": p.source, "overlap": r.n_overlap, "fraction": round(r.fraction, 3)}
              for p, r in dropped]
    (out / "contamination.json").write_text(json.dumps(report, indent=2))

    clean = [p for p, _ in clean]
    random.Random(args.seed).shuffle(clean)
    n_val = max(1, int(len(clean) * args.val_frac))
    val, train = clean[:n_val], clean[n_val:]

    n_sft = write_jsonl(out / "train_sft.jsonl", (p.to_dict() for p in train if p.solution))
    n_rl = write_jsonl(out / "train_rl.jsonl", (p.to_dict() for p in train if rl_friendly(p)))
    write_jsonl(out / "val.jsonl", (p.to_dict() for p in val if p.solution))
    log.info("train_sft %d | train_rl %d | val %d", n_sft, n_rl, len(val))


if __name__ == "__main__":
    main()
