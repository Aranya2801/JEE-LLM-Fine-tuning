from pathlib import Path

import pytest

from jee_llm.answers import is_correct
from jee_llm.contamination import find_contaminated
from jee_llm.data import Problem, from_gsm8k, from_jeebench, from_math, load_problems
from jee_llm.prompts import build_messages, to_rl_example, to_sft_example
from jee_llm.rewards import correctness_reward, format_reward, jee_marks_reward

SAMPLES = Path(__file__).resolve().parents[1] / "dataset" / "samples" / "jee_style_samples.jsonl"


def test_samples_load_and_answers_are_consistent():
    problems = load_problems(SAMPLES)
    assert len(problems) >= 10
    assert {p.subject for p in problems} == {"math", "physics", "chemistry"}
    for p in problems:
        # every reference solution must reach the gold answer, otherwise SFT
        # on the samples would teach wrong answers
        from jee_llm.answers import extract_final_answer

        assert is_correct(extract_final_answer(p.solution), p.answer, p.question_type), p.id


def test_jeebench_adapter():
    row = {
        "subject": "phy",
        "description": "JEE Adv 2016 Paper 1",
        "gold": "ABD",
        "index": 7,
        "type": "MCQ(multiple)",
        "question": "Which are true?\n\n(A) a\n\n(B) b\n\n(C) c\n\n(D) d",
    }
    p = from_jeebench(row)
    assert p.subject == "physics"
    assert p.question_type == "multiple_correct"
    assert p.year == 2016
    assert p.answer == "ABD"


def test_math_and_gsm8k_adapters():
    m = from_math({"problem": "1+1?", "solution": r"It is \boxed{2}.", "type": "Algebra"}, 0)
    assert m.answer == "2"
    assert from_math({"problem": "?", "solution": "no box"}, 1) is None

    g = from_gsm8k({"question": "q", "answer": "3 + 4 = <<3+4=7>>7\n#### 7"}, 0)
    assert g.answer == "7"
    assert "<<" not in g.solution
    assert r"\boxed{7}" in g.solution


def test_problem_validation():
    with pytest.raises(ValueError):
        Problem(id="x", subject="biology", question_type="integer", question="q", answer="1")
    with pytest.raises(ValueError):
        Problem(id="x", subject="math", question_type="single_correct", question="q", answer="A")


def test_prompt_formats():
    p = Problem(
        id="t", subject="math", question_type="single_correct", question="2+2?",
        answer="B", options={"A": "3", "B": "4", "C": "5", "D": "6"}, solution="2+2=4, so (B).",
    )
    msgs = build_messages(p)
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert "(B) 4" in msgs[1]["content"]
    assert "SKIP" not in msgs[1]["content"]
    assert "SKIP" in build_messages(p, allow_skip=True)[1]["content"]

    sft = to_sft_example(p)
    assert r"\boxed{B}" in sft["completion"][0]["content"]
    rl = to_rl_example(p)
    assert rl["answer"] == "B" and rl["question_type"] == "single_correct"


def test_rewards():
    comps = [
        [{"role": "assistant", "content": r"work... \boxed{AC}"}],
        [{"role": "assistant", "content": r"work... \boxed{A}"}],
        [{"role": "assistant", "content": r"work... \boxed{AB}"}],
        [{"role": "assistant", "content": r"\boxed{SKIP}"}],
        [{"role": "assistant", "content": "I give up"}],
    ]
    kw = dict(prompts=[None] * 5, completions=comps, answer=["AC"] * 5, question_type=["multiple_correct"] * 5)
    assert correctness_reward(**kw) == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert jee_marks_reward(**kw) == [1.0, 0.25, -0.5, 0.0, -0.5]
    assert format_reward(prompts=None, completions=comps) == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert format_reward(prompts=None, completions=[r"\boxed{1} and \boxed{2}"]) == [0.0]


def test_contamination():
    q = "A particle of mass m is projected with speed u at an angle theta to the horizontal find the maximum height"
    train = ["unrelated text about chemistry", "From the archive: " + q + " (2019)"]
    clean = "Find the number of unpaired electrons in an iron three plus ion in its ground state configuration"
    evals = [("leaked", q), ("clean", clean)]
    flagged = find_contaminated(train, evals, n=13)
    assert [r.item_id for r in flagged] == ["leaked"]
    assert flagged[0].fraction == 1.0


def test_preference_pairs_skip_uninformative_questions():
    import random
    import sys

    sys.path.insert(0, str(SAMPLES.parents[2] / "scripts"))
    from build_preference_pairs import make_pairs

    problems = load_problems(SAMPLES)[:3]
    gold = [p.answer for p in problems]
    completions = [
        [rf"\boxed{{{gold[0]}}}", r"\boxed{D}" if gold[0] != "D" else r"\boxed{A}"],  # mixed -> pair
        [rf"\boxed{{{gold[1]}}}"] * 2,                                               # all right
        ["no idea", "still no idea"],                                                # all wrong
    ]
    pairs, stats = make_pairs(problems, completions, random.Random(0))
    assert stats == {"all_right": 1, "all_wrong": 1, "used": 1}
    assert len(pairs) == 1 and pairs[0]["id"] == problems[0].id
    assert gold[0] in pairs[0]["chosen"][0]["content"]


def test_config_overrides():
    from jee_llm.config import apply_overrides

    cfg = {"training": {"lr": 1.0, "eval_strategy": "steps"}}
    out = apply_overrides(cfg, ["training.lr=1e-4", "training.eval_strategy=no", "data.files=[a, b]", "x=null"])
    assert out["training"]["lr"] == 1e-4
    assert out["training"]["eval_strategy"] == "no"  # not False
    assert out["data"]["files"] == ["a", "b"]
    assert out["x"] is None
    assert cfg["training"]["lr"] == 1.0  # original untouched
