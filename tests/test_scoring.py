import pytest

from jee_llm.metrics import ItemResult, abstention_curve, pass_at_k, summarize
from jee_llm.scoring import score_answer


@pytest.mark.parametrize(
    "pred, gold, qt, marks, status",
    [
        ("B", "B", "single_correct", 3, "correct"),
        ("C", "B", "single_correct", -1, "wrong"),
        ("SKIP", "B", "single_correct", 0, "skipped"),
        (None, "B", "single_correct", -1, "no_answer"),
        ("ACD", "ACD", "multiple_correct", 4, "correct"),
        ("AC", "ACD", "multiple_correct", 2, "partial"),
        ("A", "ACD", "multiple_correct", 1, "partial"),
        ("AB", "ACD", "multiple_correct", -2, "wrong"),
        ("7", "7", "integer", 3, "correct"),
        ("8", "7", "integer", 0, "wrong"),
        ("1.41", "1.414", "numerical", 3, "correct"),
        ("2", "1.414", "numerical", 0, "wrong"),
    ],
)
def test_marking_scheme(pred, gold, qt, marks, status):
    s = score_answer(pred, gold, qt)
    assert s.marks == marks
    assert s.status == status


def test_pass_at_k():
    assert pass_at_k(8, 0, 1) == 0.0
    assert pass_at_k(8, 8, 1) == 1.0
    assert pass_at_k(8, 2, 1) == pytest.approx(0.25)
    # with 2 of 8 correct, chance at least one of 4 draws is correct
    assert pass_at_k(8, 2, 4) == pytest.approx(1 - (6 * 5 * 4 * 3) / (8 * 7 * 6 * 5))
    assert pass_at_k(4, 1, 4) == 1.0


def _item(id, qt, gold, completions, subject="math"):
    return ItemResult.build(id, subject, qt, gold, completions)


def test_summarize_and_abstention():
    items = [
        _item("1", "single_correct", "A", [r"\boxed{A}"] * 4),
        _item("2", "single_correct", "B", [r"\boxed{C}", r"\boxed{D}", r"\boxed{A}", r"\boxed{C}"]),
        _item("3", "integer", "4", [r"\boxed{4}", r"\boxed{5}", r"\boxed{4}", r"\boxed{4}"], "physics"),
    ]
    s = summarize(items, ks=(1, 4))
    o = s["overall"]
    assert o["n"] == 3
    assert o["accuracy@1"] == pytest.approx(2 / 3)
    assert o["maj@4"] == pytest.approx(2 / 3)
    assert o["jee_score"] == 3 - 1 + 3
    assert o["jee_max"] == 9
    assert set(s["by_subject"]) == {"math", "physics"}

    curve = {row["threshold"]: row for row in abstention_curve(items, thresholds=(0.0, 0.75))}
    assert curve[0.0]["score"] == 5     # majority answers: A (+3), C (-1), 4 (+3)
    assert curve[0.75]["score"] == 6    # item 2 only has 50% agreement -> skipped
    assert curve[0.75]["attempted"] == 2
