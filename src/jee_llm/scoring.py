"""JEE Advanced marking scheme.

Following the JEEBench paper (Arora et al., EMNLP 2023), which in turn follows
the 2016-2023 papers:

  single_correct    +3 correct, -1 wrong, 0 skipped
  multiple_correct  +4 all correct options and nothing else
                    +1 per correct option if no wrong option is picked (partial)
                    -2 if any wrong option is picked
                    0 skipped
  integer/numerical +3 correct, 0 otherwise (no negative marking)

The partial-marking rule for multi-correct has changed a few times across
years (2022+ gives +3 when 3 of 4 correct are picked, etc). I use the simpler
JEEBench version so numbers are comparable to that paper. If you want a
specific year's scheme, override MARKS.
"""

from __future__ import annotations

from dataclasses import dataclass

from .answers import is_correct, is_skip, parse_options

MARKS = {
    "single_correct": {"correct": 3, "wrong": -1},
    "multiple_correct": {"correct": 4, "wrong": -2, "partial_per_option": 1},
    "integer": {"correct": 3, "wrong": 0},
    "numerical": {"correct": 3, "wrong": 0},
}


@dataclass(frozen=True)
class Score:
    marks: float
    max_marks: float
    status: str  # "correct" | "partial" | "wrong" | "skipped" | "no_answer"


def max_marks(question_type: str) -> float:
    return float(MARKS.get(question_type, {"correct": 1})["correct"])


def score_answer(pred: str | None, gold: str, question_type: str) -> Score:
    """Marks awarded for one extracted answer under the JEE scheme.

    `pred=None` means we couldn't extract anything. I treat that as *wrong*
    rather than skipped: a model that rambles without committing shouldn't get
    the benefit of abstention. Explicit skips (\\boxed{SKIP}) score 0.
    """
    top = max_marks(question_type)
    scheme = MARKS.get(question_type)

    if pred is not None and is_skip(pred):
        return Score(0.0, top, "skipped")

    if scheme is None:  # open-ended, not a JEE format: plain 0/1
        ok = is_correct(pred, gold, question_type)
        return Score(1.0 if ok else 0.0, 1.0, "correct" if ok else ("no_answer" if pred is None else "wrong"))

    if pred is None:
        return Score(float(scheme["wrong"]), top, "no_answer")

    if question_type == "multiple_correct":
        gold_set = parse_options(gold)
        pred_set = parse_options(pred)
        if pred_set is None:
            return Score(float(scheme["wrong"]), top, "wrong")
        if pred_set == gold_set:
            return Score(top, top, "correct")
        if pred_set - gold_set:  # picked something wrong
            return Score(float(scheme["wrong"]), top, "wrong")
        # strict subset of the right options
        return Score(float(scheme["partial_per_option"] * len(pred_set)), top, "partial")

    if is_correct(pred, gold, question_type):
        return Score(top, top, "correct")
    return Score(float(scheme["wrong"]), top, "wrong")
