"""Aggregating per-question results into the numbers that go in tables."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import comb

from .answers import canonical_key, extract_final_answer, is_correct, majority_vote
from .scoring import score_answer


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from Chen et al. 2021 (Codex paper), n samples with c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


@dataclass
class ItemResult:
    id: str
    subject: str
    question_type: str
    gold: str
    completions: list[str]
    preds: list[str | None] = field(default_factory=list)
    correct: list[bool] = field(default_factory=list)
    vote: str | None = None
    vote_share: float = 0.0

    @classmethod
    def build(cls, id, subject, question_type, gold, completions) -> ItemResult:
        r = cls(id, subject, question_type, gold, list(completions))
        r.preds = [extract_final_answer(c) for c in r.completions]
        r.correct = [is_correct(p, gold, question_type) for p in r.preds]
        keys = [canonical_key(p, question_type) for p in r.preds]
        r.vote, r.vote_share = majority_vote(keys)
        return r

    def vote_pred(self) -> str | None:
        """A representative raw prediction for the majority key."""
        for p in self.preds:
            if canonical_key(p, self.question_type) == self.vote:
                return p
        return None


def summarize(results: list[ItemResult], ks: tuple[int, ...] = (1, 4, 8)) -> dict:
    """Overall + per subject + per question type metrics."""

    def block(rs: list[ItemResult]) -> dict:
        if not rs:
            return {}
        n_samples = min(len(r.completions) for r in rs)
        out = {"n": len(rs), "samples_per_item": n_samples}
        # accuracy@1 uses only the first completion (the greedy one when temperature=0);
        # pass@1 averages over all samples, which is less noisy when sampling
        out["accuracy@1"] = sum(r.correct[0] for r in rs) / len(rs)
        out["no_answer_rate"] = sum(r.preds[0] is None for r in rs) / len(rs)
        for k in ks:
            if k <= n_samples:
                out[f"pass@{k}"] = sum(pass_at_k(n_samples, sum(r.correct[:n_samples]), k) for r in rs) / len(rs)
        if n_samples > 1:
            out[f"maj@{n_samples}"] = sum(is_correct(r.vote_pred(), r.gold, r.question_type) for r in rs) / len(rs)
        # exam score with the first sample, as a fraction of max marks
        scores = [score_answer(r.preds[0], r.gold, r.question_type) for r in rs]
        out["jee_score"] = sum(s.marks for s in scores)
        out["jee_max"] = sum(s.max_marks for s in scores)
        out["jee_score_pct"] = out["jee_score"] / out["jee_max"] if out["jee_max"] else 0.0
        return out

    by_subject: dict[str, list] = defaultdict(list)
    by_type: dict[str, list] = defaultdict(list)
    for r in results:
        by_subject[r.subject].append(r)
        by_type[r.question_type].append(r)
    return {
        "overall": block(results),
        "by_subject": {k: block(v) for k, v in sorted(by_subject.items())},
        "by_type": {k: block(v) for k, v in sorted(by_type.items())},
    }


def abstention_curve(results: list[ItemResult], thresholds=(0.0, 0.25, 0.5, 0.75, 1.0)) -> list[dict]:
    """Exam score if we only answer when the majority vote share is >= tau.

    This is the self-consistency thresholding trick from the JEEBench paper:
    with negative marking, skipping low-agreement questions can raise the
    score even though accuracy on attempted questions is what changes.
    Integer/numerical questions have no negative marks, so we always answer those.
    """
    rows = []
    for tau in thresholds:
        total = maxm = attempted = 0.0
        for r in results:
            s = score_answer(r.vote_pred(), r.gold, r.question_type)
            maxm += s.max_marks
            negative_marking = r.question_type in ("single_correct", "multiple_correct")
            if negative_marking and r.vote_share < tau:
                continue  # skip -> 0 marks
            attempted += 1
            total += s.marks
        rows.append({
            "threshold": tau,
            "score": total,
            "max": maxm,
            "score_pct": total / maxm if maxm else 0.0,
            "attempted": int(attempted),
        })
    return rows
