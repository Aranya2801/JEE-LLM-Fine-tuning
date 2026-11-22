"""Problem schema and loaders.

Everything gets converted into `Problem` early so the training and eval code
only ever sees one format. Raw sources (JEEBench, MATH, GSM8K, your own
scraped papers) each get a small adapter function here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

SUBJECTS = ("math", "physics", "chemistry")
QUESTION_TYPES = ("single_correct", "multiple_correct", "integer", "numerical", "open_ended")


@dataclass
class Problem:
    id: str
    subject: str
    question_type: str
    question: str
    answer: str
    solution: str = ""  # reference chain of thought, optional
    options: dict[str, str] = field(default_factory=dict)
    source: str = ""
    year: int | None = None
    topic: str = ""

    def __post_init__(self) -> None:
        if self.subject not in SUBJECTS:
            raise ValueError(f"{self.id}: unknown subject {self.subject!r}")
        if self.question_type not in QUESTION_TYPES:
            raise ValueError(f"{self.id}: unknown question_type {self.question_type!r}")
        if self.question_type in ("single_correct", "multiple_correct") and not self.options:
            # JEEBench embeds options inside the question text, that's fine
            if not re.search(r"\(A\)", self.question):
                raise ValueError(f"{self.id}: MCQ without options")

    def to_dict(self) -> dict:
        return asdict(self)


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: bad json ({e})") from e


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_problems(path: str | Path) -> list[Problem]:
    return [Problem(**row) for row in read_jsonl(path)]


# ---------------------------------------------------------------------------
# source adapters
# ---------------------------------------------------------------------------

_JEEBENCH_SUBJECT = {"phy": "physics", "chem": "chemistry", "math": "math"}
_JEEBENCH_TYPE = {
    "MCQ": "single_correct",
    "MCQ(multiple)": "multiple_correct",
    "Integer": "integer",
    "Numeric": "numerical",
}


def from_jeebench(row: dict) -> Problem:
    """daman1209arora/jeebench row -> Problem.

    Fields: subject (phy/chem/math), description ("JEE Adv 2016 Paper 1"),
    gold, index, type, question. Options are embedded in the question text.
    """
    year_match = re.search(r"(20\d\d)", row.get("description", ""))
    return Problem(
        id=f"jeebench_{row['index']}",
        subject=_JEEBENCH_SUBJECT[row["subject"]],
        question_type=_JEEBENCH_TYPE[row["type"]],
        question=row["question"].strip(),
        answer=str(row["gold"]).strip(),
        source=row.get("description", "JEEBench"),
        year=int(year_match.group(1)) if year_match else None,
    )


def from_math(row: dict, idx: int) -> Problem | None:
    """Hendrycks MATH row -> Problem. Answer is the last \\boxed{} of the solution."""
    from .answers import extract_boxed

    ans = extract_boxed(row["solution"])
    if not ans:
        return None
    return Problem(
        id=f"math_{idx:05d}",
        subject="math",
        question_type="open_ended",
        question=row["problem"].strip(),
        answer=ans,
        solution=row["solution"].strip(),
        source="MATH",
        topic=row.get("type", ""),
    )


def from_gsm8k(row: dict, idx: int) -> Problem:
    reasoning, _, ans = row["answer"].partition("####")
    # gsm8k uses <<48/2=24>> calculator annotations, they just add noise
    reasoning = re.sub(r"<<[^>]*>>", "", reasoning).strip()
    ans = ans.strip().replace(",", "")
    return Problem(
        id=f"gsm8k_{idx:05d}",
        subject="math",
        question_type="numerical",
        question=row["question"].strip(),
        answer=ans,
        solution=f"{reasoning}\n\nThe final answer is \\boxed{{{ans}}}.",
        source="GSM8K",
    )
