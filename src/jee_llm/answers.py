"""Pulling the final answer out of a model's solution and checking it.

This is the piece everything else leans on (eval, GRPO rewards, building DPO
pairs), so it's deliberately conservative: if we can't find an answer we
return None instead of guessing. A wrong "extracted" answer is worse than
no answer because it silently corrupts the reward signal.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

SKIP_TOKENS = {"skip", "skipped", "none", "noanswer", "pass"}

_LETTERS = "ABCD"


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    """Return the contents of the *last* \\boxed{...} in `text`.

    Handles nested braces (\\boxed{\\frac{1}{2}}), which a plain regex can't.
    Also accepts \\fbox{...} since some models use it.
    """
    if not text:
        return None
    starts = [m.end() for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    depth = 1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None  # unbalanced, probably truncated generation


_FINAL_PATTERNS = [
    r"final answer\s*(?:is|:)\s*(.+)",
    r"the answer is\s*:?\s*(.+)",
    r"^answer\s*:\s*(.+)",
]


def extract_final_answer(text: str) -> str | None:
    """\\boxed{} first, then a few 'final answer is ...' fallbacks."""
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    if not text:
        return None
    # look at the tail only, the model often restates the question earlier
    tail = text[-600:]
    for pat in _FINAL_PATTERNS:
        matches = list(re.finditer(pat, tail, flags=re.IGNORECASE | re.MULTILINE))
        if matches:
            ans = matches[-1].group(1).strip()
            ans = ans.splitlines()[0].strip().rstrip(".").strip()
            ans = ans.strip("$").strip()
            return ans or None
    return None


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

def _strip_latex_wrappers(s: str) -> str:
    # \text{A}, \mathrm{B}, \textbf{C} -> A, B, C
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\\(?:text|mathrm|textbf|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    return s.replace("$", "").strip()


def is_skip(answer: str | None) -> bool:
    if answer is None:
        return False
    s = re.sub(r"[^a-z]", "", _strip_latex_wrappers(answer).lower())
    return s in SKIP_TOKENS


def parse_options(answer: str | None) -> frozenset[str] | None:
    """Turn an MCQ answer into a set of option letters.

    "B", "(B)", "A, C", "AC", "A and D", "\\text{(A), (C)}", "option B" all work.
    Returns None if the string doesn't look like an option answer at all, e.g.
    "x = 2" or "the coefficient is 4".
    """
    if answer is None:
        return None
    s = _strip_latex_wrappers(answer).upper()
    s = re.sub(r"\b(?:OPTIONS?|AND|OR)\b", " ", s)
    s = re.sub(r"[()\[\],;&/+.:\s]", " ", s)
    tokens = s.split()
    if not tokens:
        return None
    letters: set[str] = set()
    for tok in tokens:
        # "ACD" style or a single letter; anything else means it's not an option answer
        if re.fullmatch(r"[A-D]{1,4}", tok) and len(set(tok)) == len(tok):
            letters.update(tok)
        else:
            return None
    return frozenset(letters) if letters else None


_UNIT_WORDS = (
    r"m/s\^?2?|m/s|km/h|kg|g|mol|m|cm|mm|nm|s|ms|j|kj|ev|w|n|pa|atm|k|v|a|"
    r"ohm|ω|hz|l|ml|c|°c|°|degrees?|units?|rad"
)


def _clean_numeric(s: str) -> str:
    s = _strip_latex_wrappers(s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace("~", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("−", "-").replace("×", "*").replace("·", "*")
    # "x = 2.5" -> "2.5"
    if "=" in s:
        s = s.split("=")[-1]
    # drop trailing units: "5 m/s", "9 J", "300 K"
    s = re.sub(rf"\s*(?:\\?\s*(?:{_UNIT_WORDS}))\s*$", "", s.strip(), flags=re.IGNORECASE)
    # 1,000 -> 1000 (but only when it really looks like a thousands separator)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    return s.strip()


def _latex_to_python(s: str) -> str:
    # a tiny LaTeX -> python expression translator. sympy's own latex parser
    # needs a pinned antlr runtime and breaks too often to rely on in CI.
    for _ in range(5):
        s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\s*\[([^{}\]]*)\]\s*\{([^{}]*)\}", r"((\2)**(1/(\1)))", s)
        s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*(\d+)", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("π", "pi")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\ln", "log").replace("\\log", "log")
    s = re.sub(r"\\(sin|cos|tan|exp)", r"\1", s)
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("^", "**")
    return s


def parse_number(answer: str | None) -> float | None:
    """Best effort numeric value of an answer string. None if it isn't a number."""
    if answer is None:
        return None
    s = _clean_numeric(answer)
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    # scientific notation written the physics way: 6.6 \times 10^{-34}
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*(?:\\times|\*|x)\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?", s)
    if m:
        return float(m.group(1)) * 10 ** int(m.group(2))
    expr = _latex_to_python(s)
    if re.search(r"[^0-9a-z+\-*/().\s]", expr) or "\\" in expr:
        return None
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        val = parse_expr(
            expr,
            transformations=standard_transformations + (implicit_multiplication_application,),
            local_dict={"e": sympy.E, "pi": sympy.pi},
            evaluate=True,
        )
        if val.free_symbols:
            return None
        out = complex(val.evalf())
        if abs(out.imag) > 1e-12:
            return None
        return float(out.real)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def numbers_match(pred: float, gold: float, abs_tol: float = 0.01, rel_tol: float = 1e-3) -> bool:
    """JEE numerical answers are 'correct up to 2 decimal places'.

    I accept either |pred - gold| <= abs_tol or a small relative error, the
    latter so that answers like 6.63e-34 aren't trivially 'equal' to 0 and
    big answers like 1250 don't need to be exact to the hundredth.
    """
    if math.isnan(pred) or math.isinf(pred):
        return False
    # absolute tolerance only makes sense when gold isn't itself tiny
    if gold == 0 or abs(gold) >= 1e-2:
        if abs(pred - gold) <= abs_tol:
            return True
    return math.isclose(pred, gold, rel_tol=rel_tol, abs_tol=0.0)


MCQ_TYPES = {"single_correct", "multiple_correct"}
NUMERIC_TYPES = {"integer", "numerical"}


def normalize_gold(gold: str, question_type: str) -> frozenset[str] | float | str:
    if question_type in MCQ_TYPES:
        opts = parse_options(gold)
        if opts is None:
            raise ValueError(f"gold answer {gold!r} is not a valid option string")
        return opts
    if question_type in NUMERIC_TYPES:
        val = parse_number(gold)
        if val is None:
            raise ValueError(f"gold answer {gold!r} is not numeric")
        return val
    return gold.strip()


def is_correct(pred: str | None, gold: str, question_type: str) -> bool:
    """Strict correctness (full marks only). For partial credit see scoring.py."""
    if pred is None or is_skip(pred):
        return False
    if question_type in MCQ_TYPES:
        p = parse_options(pred)
        return p is not None and p == normalize_gold(gold, question_type)
    if question_type in NUMERIC_TYPES:
        p = parse_number(pred)
        if p is None:
            return False
        g = normalize_gold(gold, question_type)
        if question_type == "integer":
            return abs(p - round(p)) < 1e-6 and round(p) == round(g)
        return numbers_match(p, g)
    # open ended: numeric if both sides parse, else normalised string compare
    p_num, g_num = parse_number(pred), parse_number(gold)
    if p_num is not None and g_num is not None:
        return numbers_match(p_num, g_num, abs_tol=1e-6, rel_tol=1e-6)
    norm = lambda x: re.sub(r"\s+", "", _strip_latex_wrappers(x)).lower()  # noqa: E731
    return norm(pred) == norm(gold)


def canonical_key(pred: str | None, question_type: str) -> str | None:
    """A hashable form of an answer, used for majority voting across samples.

    Two answers with the same key count as the same vote, e.g. "AC" and "(A), (C)",
    or "0.5" and "\\frac{1}{2}".
    """
    if pred is None:
        return None
    if is_skip(pred):
        return "SKIP"
    if question_type in MCQ_TYPES:
        p = parse_options(pred)
        return "".join(sorted(p)) if p else None
    val = parse_number(pred)
    if val is not None:
        return f"{val:.2f}" if question_type != "integer" else str(round(val))
    return re.sub(r"\s+", "", _strip_latex_wrappers(pred)).lower() or None


def majority_vote(keys: Iterable[str | None]) -> tuple[str | None, float]:
    """Most common non-null key and its share of *all* samples (nulls included)."""
    keys = list(keys)
    if not keys:
        return None, 0.0
    counts: dict[str, int] = {}
    for k in keys:
        if k is not None:
            counts[k] = counts.get(k, 0) + 1
    if not counts:
        return None, 0.0
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0], best[1] / len(keys)
