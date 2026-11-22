"""Check any training file against the eval set for n-gram overlap.

    python scripts/check_contamination.py --train dataset/train_sft.jsonl my_extra.jsonl --eval dataset/jeebench.jsonl

prepare_data.py already does this for the sources it downloads; use this for
anything you add by hand (scraped papers, coaching material, ...).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jee_llm.contamination import find_contaminated  # noqa: E402
from jee_llm.data import read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--threshold", type=float, default=0.0, help="min fraction of eval n-grams found in train")
    args = ap.parse_args()

    def train_texts():
        for path in args.train:
            for r in read_jsonl(path):
                yield r["question"]
                yield r.get("solution", "")

    evals = [(r["id"], r["question"]) for r in read_jsonl(args.eval)]
    flagged = sorted(find_contaminated(train_texts(), evals, n=args.n, threshold=args.threshold),
                     key=lambda r: -r.fraction)
    print(f"{len(flagged)}/{len(evals)} eval items overlap with training data ({args.n}-grams)")
    for r in flagged[:30]:
        print(f"  {r.item_id:<20} {r.n_overlap:>4}/{r.n_ngrams:<4} ({100 * r.fraction:.0f}%)")
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
