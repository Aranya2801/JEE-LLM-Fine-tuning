"""Prompt construction.

One system prompt for every stage (SFT, RL, eval). Changing the prompt between
training and eval is an easy way to fool yourself, so everything goes through
`build_messages`.
"""

from __future__ import annotations

from .data import Problem

SYSTEM_PROMPT = (
    "You are solving a JEE Advanced problem. Think step by step and show your "
    "working. End with the final answer in \\boxed{}."
)

_FORMAT_HINT = {
    "single_correct": "Exactly one option is correct. Put the letter in \\boxed{}, e.g. \\boxed{B}.",
    "multiple_correct": (
        "One or more options are correct. Put all correct letters in \\boxed{}, "
        "e.g. \\boxed{AC}. A wrong option costs marks, so only include options you are sure of."
    ),
    "integer": "The answer is a non-negative integer. Put it in \\boxed{}.",
    "numerical": "The answer is a number, correct to two decimal places. Put it in \\boxed{}.",
    "open_ended": "Put the final answer in \\boxed{}.",
}

SKIP_HINT = (
    " There is negative marking. If you are not confident, you may answer "
    "\\boxed{SKIP} to score 0 instead."
)


def format_question(p: Problem, allow_skip: bool = False) -> str:
    parts = [p.question.strip()]
    if p.options:
        parts.append("\n".join(f"({k}) {v}" for k, v in sorted(p.options.items())))
    hint = _FORMAT_HINT[p.question_type]
    if allow_skip and p.question_type in ("single_correct", "multiple_correct"):
        hint += SKIP_HINT
    parts.append(hint)
    return "\n\n".join(parts)


def build_messages(p: Problem, allow_skip: bool = False) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_question(p, allow_skip=allow_skip)},
    ]


def to_sft_example(p: Problem) -> dict:
    """prompt/completion conversational format; TRL masks the prompt tokens by default."""
    if not p.solution:
        raise ValueError(f"{p.id} has no reference solution, can't use it for SFT")
    solution = p.solution.strip()
    if "\\boxed" not in solution:
        solution += f"\n\nThe final answer is \\boxed{{{p.answer}}}."
    return {
        "prompt": build_messages(p),
        "completion": [{"role": "assistant", "content": solution}],
    }


def to_rl_example(p: Problem, allow_skip: bool = False) -> dict:
    """GRPO only needs the prompt plus whatever the reward functions read."""
    return {
        "prompt": build_messages(p, allow_skip=allow_skip),
        "answer": p.answer,
        "question_type": p.question_type,
        "subject": p.subject,
    }
