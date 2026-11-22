"""Reward functions for GRPO.

TRL calls each function as f(prompts, completions, **dataset_columns) and
expects a list of floats back. With conversational datasets a completion is
[{"role": "assistant", "content": "..."}].

Two correctness rewards, which is the main thing the RL experiments compare:

* `correctness_reward`  - 1 if fully correct, else 0 (the usual RLVR setup)
* `jee_marks_reward`    - actual exam marks scaled to [-0.5, 1]. Wrong answers
                          are penalised, skipping is 0, partial credit exists
                          for multi-correct. The question is whether this
                          teaches the model *when not to answer*.
"""

from __future__ import annotations

import re

from .answers import extract_boxed, extract_final_answer, is_correct
from .scoring import score_answer


def _text(completion) -> str:
    if isinstance(completion, str):
        return completion
    # conversational: list of messages, take the last assistant turn
    return completion[-1]["content"] if completion else ""


def correctness_reward(prompts, completions, answer, question_type, **kwargs) -> list[float]:
    out = []
    for c, gold, qt in zip(completions, answer, question_type, strict=True):
        pred = extract_final_answer(_text(c))
        out.append(1.0 if is_correct(pred, gold, qt) else 0.0)
    return out


def jee_marks_reward(prompts, completions, answer, question_type, **kwargs) -> list[float]:
    out = []
    for c, gold, qt in zip(completions, answer, question_type, strict=True):
        pred = extract_final_answer(_text(c))
        s = score_answer(pred, gold, qt)
        out.append(s.marks / s.max_marks)
    return out


def format_reward(prompts, completions, **kwargs) -> list[float]:
    """Small bonus for exactly one \\boxed{} at the end.

    Kept small (0.1 by default via reward_weights in the config) so it can't
    dominate. Mostly matters for the first few dozen steps when the base model
    hasn't picked up the format yet.
    """
    out = []
    for c in completions:
        t = _text(c).strip()
        n_boxed = len(re.findall(r"\\boxed\s*\{", t))
        # the box should close the solution, not sit in the middle of more working
        ok = n_boxed == 1 and extract_boxed(t) is not None and len(t) - t.rfind("\\boxed") < 300
        out.append(1.0 if ok else 0.0)
    return out


REWARD_FUNCS = {
    "correctness": correctness_reward,
    "jee_marks": jee_marks_reward,
    "format": format_reward,
}
