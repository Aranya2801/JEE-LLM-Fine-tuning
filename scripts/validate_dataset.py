"""Sanity checks for Problem-schema jsonl files before training on them.

    python scripts/validate_dataset.py dataset/train_sft.jsonl dataset/jeebench.jsonl

Fails (exit 1) on schema errors and on reference solutions whose final answer
doesn't match the gold answer. The second one matters a lot: training on
solutions that end at the wrong answer is worse than not training on them.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jee_llm.answers import extract_final_answer, is_correct, normalize_gold  # noqa: E402
from jee_llm.data import Problem, read_jsonl  # noqa: E402


def check(path: str, max_show: int = 10) -> bool:
    errors, mismatched, ids = [], [], Counter()
    subjects, types = Counter(), Counter()
    n = with_solution = 0
    for i, row in enumerate(read_jsonl(path), 1):
        n += 1
        try:
            p = Problem(**row)
            normalize_gold(p.answer, p.question_type)
        except (TypeError, ValueError) as e:
            errors.append(f"line {i}: {e}")
            continue
        ids[p.id] += 1
        subjects[p.subject] += 1
        types[p.question_type] += 1
        if p.solution:
            with_solution += 1
            if not is_correct(extract_final_answer(p.solution), p.answer, p.question_type):
                mismatched.append(p.id)

    dupes = [k for k, v in ids.items() if v > 1]
    print(f"\n{path}: {n} rows, {with_solution} with solutions")
    print(f"  subjects: {dict(subjects)}")
    print(f"  types:    {dict(types)}")
    for label, items in (("schema errors", errors), ("duplicate ids", dupes),
                         ("solution doesn't reach gold answer", mismatched)):
        if items:
            print(f"  {label}: {len(items)}")
            for x in items[:max_show]:
                print(f"    {x}")
    # a few mismatches are expected in big scraped sets (parser misses); flag above 2%
    bad_rate = len(mismatched) / with_solution if with_solution else 0
    return not errors and not dupes and bad_rate <= 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    ok = all([check(f) for f in args.files])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
