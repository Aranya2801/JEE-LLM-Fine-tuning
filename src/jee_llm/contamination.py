"""Train/test overlap check.

JEE questions are all over the internet (coaching sites, forums, solution
PDFs) so any scraped training set will contain JEEBench questions unless you
look for them. I use the same idea as the GPT-3 paper: flag an eval item if
it shares any 13-gram (after normalisation) with the training data.

13 is fairly strict for short JEE questions, so I also report the max
overlap fraction per item and let you pick a threshold.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


def normalize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"\\[a-z]+", " ", text)  # drop latex commands, keep their args
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.split()


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


@dataclass
class OverlapResult:
    item_id: str
    n_ngrams: int
    n_overlap: int

    @property
    def fraction(self) -> float:
        return self.n_overlap / self.n_ngrams if self.n_ngrams else 0.0


class NgramIndex:
    def __init__(self, n: int = 13):
        self.n = n
        self._grams: set[int] = set()

    def add(self, text: str) -> None:
        # hashing keeps memory sane for a few hundred thousand training docs
        self._grams.update(hash(g) for g in ngrams(normalize(text), self.n))

    def add_all(self, texts: Iterable[str]) -> None:
        for t in texts:
            self.add(t)

    def check(self, item_id: str, text: str) -> OverlapResult:
        grams = ngrams(normalize(text), self.n)
        hits = sum(1 for g in grams if hash(g) in self._grams)
        return OverlapResult(item_id, len(grams), hits)

    def __len__(self) -> int:
        return len(self._grams)


def find_contaminated(
    train_texts: Iterable[str],
    eval_items: Iterable[tuple[str, str]],
    n: int = 13,
    threshold: float = 0.0,
) -> list[OverlapResult]:
    """Eval items whose overlap fraction is > threshold (default: any shared n-gram)."""
    index = NgramIndex(n)
    index.add_all(train_texts)
    flagged = []
    for item_id, text in eval_items:
        r = index.check(item_id, text)
        if r.n_overlap and r.fraction > threshold:
            flagged.append(r)
    return flagged
