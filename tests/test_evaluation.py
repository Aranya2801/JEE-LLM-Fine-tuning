"""
test_evaluation.py — Unit tests for the evaluation module
"""

import pytest
import sys
sys.path.insert(0, ".")

from src.evaluation.evaluate import (
    extract_boxed,
    extract_mcq_choice,
    extract_numerical,
    check_answer,
)


class TestExtractBoxed:
    def test_basic(self):
        assert extract_boxed("The answer is \\boxed{42}") == "42"

    def test_last_boxed(self):
        assert extract_boxed("\\boxed{3} ... \\boxed{\\pi}") == "\\pi"

    def test_no_boxed(self):
        assert extract_boxed("No answer here") is None

    def test_fraction(self):
        assert extract_boxed("Result: \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


class TestExtractMCQ:
    def test_boxed_letter(self):
        assert extract_mcq_choice("The answer is \\boxed{B}") == "B"

    def test_answer_pattern(self):
        assert extract_mcq_choice("Answer: C is correct.") == "C"

    def test_none(self):
        assert extract_mcq_choice("The result is 42.") is None


class TestExtractNumerical:
    def test_integer(self):
        assert extract_numerical("\\boxed{42}") == 42.0

    def test_float(self):
        assert extract_numerical("The answer is \\boxed{3.14}") == 3.14

    def test_negative(self):
        assert extract_numerical("\\boxed{-7.5}") == -7.5

    def test_none(self):
        assert extract_numerical("No numbers here.") is None


class TestCheckAnswer:
    def test_single_correct_match(self):
        correct, score = check_answer("\\boxed{A}", "A", "single_correct")
        assert correct is True
        assert score == 1.0

    def test_single_correct_wrong(self):
        correct, score = check_answer("\\boxed{B}", "A", "single_correct")
        assert correct is False
        assert score == 0.0

    def test_numerical_exact(self):
        correct, score = check_answer("\\boxed{1.10}", "1.10", "numerical")
        assert correct is True

    def test_numerical_tolerance(self):
        correct, score = check_answer("\\boxed{1.1001}", "1.10", "numerical")
        assert correct is True

    def test_numerical_wrong(self):
        correct, score = check_answer("\\boxed{2.5}", "1.10", "numerical")
        assert correct is False

    def test_multiple_correct_exact(self):
        correct, score = check_answer("\\boxed{AB}", "AB", "multiple_correct")
        assert correct is True

    def test_multiple_correct_partial(self):
        correct, score = check_answer("\\boxed{A}", "AB", "multiple_correct")
        assert correct is False
        assert 0 < score < 1


class TestDataSchema:
    """Test the sample dataset schema."""

    def test_sample_dataset_valid(self):
        import json
        required = ["id", "year", "subject", "question", "correct_answer"]
        with open("dataset/samples/sample_jee_cot.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                for field in required:
                    assert field in ex, f"Missing field '{field}' in {ex.get('id', '?')}"
                assert len(ex["question"]) >= 20, "Question too short"
                assert ex["subject"] in ["Mathematics", "Physics", "Chemistry"]
