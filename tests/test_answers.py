import pytest

from jee_llm.answers import (
    canonical_key,
    extract_boxed,
    extract_final_answer,
    is_correct,
    is_skip,
    majority_vote,
    parse_number,
    parse_options,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        (r"so \boxed{42}", "42"),
        (r"\boxed{3} then later \boxed{\pi}", r"\pi"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"\boxed{\sqrt{\frac{3}{4}}} done", r"\sqrt{\frac{3}{4}}"),
        (r"\fbox{B}", "B"),
        ("no answer here", None),
        (r"truncated \boxed{\frac{1}{2", None),
    ],
)
def test_extract_boxed(text, expected):
    assert extract_boxed(text) == expected


def test_final_answer_fallbacks():
    assert extract_final_answer("blah\nFinal answer: 12.5 m/s") == "12.5 m/s"
    assert extract_final_answer("so the answer is (C).") == "(C)"
    assert extract_final_answer(r"The final answer is $\frac{3}{2}$.") == r"\frac{3}{2}"
    assert extract_final_answer("I don't know") is None


@pytest.mark.parametrize(
    "s, expected",
    [
        ("B", {"B"}),
        ("(B)", {"B"}),
        ("A, C", {"A", "C"}),
        ("ACD", {"A", "C", "D"}),
        ("A and D", {"A", "D"}),
        (r"\text{(A), (C)}", {"A", "C"}),
        ("option b", {"B"}),
        ("x = 2", None),
        ("AAB", None),
        ("E", None),
        ("", None),
    ],
)
def test_parse_options(s, expected):
    got = parse_options(s)
    assert (got is None and expected is None) or got == frozenset(expected)


@pytest.mark.parametrize(
    "s, expected",
    [
        ("42", 42.0),
        ("-7.5", -7.5),
        (r"\frac{1}{2}", 0.5),
        (r"\dfrac{3}{4}", 0.75),
        (r"2\sqrt{2}", 2 * 2**0.5),
        (r"\pi", 3.141592653589793),
        ("x = 2.50", 2.5),
        ("5 m/s", 5.0),
        (r"9 \text{ J}", 9.0),
        ("1,000", 1000.0),
        (r"6.6 \times 10^{-34}", 6.6e-34),
        ("50%", 50.0),
        ("abc", None),
        ("x + 1", None),
    ],
)
def test_parse_number(s, expected):
    got = parse_number(s)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected, rel=1e-9)


class TestIsCorrect:
    def test_single(self):
        assert is_correct("B", "B", "single_correct")
        assert is_correct("(b)", "B", "single_correct")
        assert not is_correct("C", "B", "single_correct")
        assert not is_correct(None, "B", "single_correct")

    def test_multiple_needs_exact_set(self):
        assert is_correct("CA", "AC", "multiple_correct")
        assert not is_correct("A", "AC", "multiple_correct")
        assert not is_correct("ABC", "AC", "multiple_correct")

    def test_numerical_two_decimals(self):
        assert is_correct("0.5", "0.50", "numerical")
        assert is_correct(r"\frac{1}{2}", "0.5", "numerical")
        assert is_correct("2.346", "2.35", "numerical")
        assert not is_correct("2.37", "2.35", "numerical")

    def test_numerical_small_values_use_relative(self):
        # 6.6e-34 vs 0 must not count as equal just because |diff| < 0.01
        assert not is_correct("0", "6.6e-34", "numerical")
        assert is_correct(r"6.6 \times 10^{-34}", "6.6e-34", "numerical")

    def test_integer(self):
        assert is_correct("5", "5", "integer")
        assert is_correct("5.0", "5", "integer")
        assert not is_correct("5.4", "5", "integer")

    def test_skip_is_never_correct(self):
        assert not is_correct("SKIP", "B", "single_correct")


def test_skip_detection():
    assert is_skip("SKIP")
    assert is_skip(r"\text{skip}")
    assert not is_skip("B")
    assert not is_skip(None)


def test_majority_vote_merges_equivalent_answers():
    preds = ["AC", "(A), (C)", "A", None]
    keys = [canonical_key(p, "multiple_correct") for p in preds]
    vote, share = majority_vote(keys)
    assert vote == "AC"
    assert share == 0.5

    keys = [canonical_key(p, "numerical") for p in ["0.5", r"\frac{1}{2}", "0.7"]]
    assert majority_vote(keys)[0] == "0.50"


def test_majority_vote_empty():
    assert majority_vote([]) == (None, 0.0)
    assert majority_vote([None, None]) == (None, 0.0)
