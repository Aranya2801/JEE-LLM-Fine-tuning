"""
validate_dataset.py — Dataset Quality Validation
=================================================
Validates all JSONL dataset files for schema compliance,
content quality, and distribution balance.

Usage:
    python scripts/validate_dataset.py --data_dir dataset/ --report

Author: Aranya Ghosh
"""

import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")

REQUIRED_FIELDS = ["id", "subject", "question", "correct_answer"]
VALID_SUBJECTS = {"Mathematics", "Physics", "Chemistry"}
VALID_Q_TYPES = {
    "single_correct", "multiple_correct",
    "numerical", "integer", "matrix_match",
    "paragraph", "open_ended",
}
MIN_QUESTION_LEN = 20
MIN_COT_LEN = 80
MAX_QUESTION_LEN = 3000


def validate_file(filepath: str) -> dict:
    """Validate a single JSONL file and return quality report."""
    path = Path(filepath)
    stats = {
        "file": path.name,
        "total": 0,
        "valid": 0,
        "errors": [],
        "warnings": [],
        "subject_dist": Counter(),
        "type_dist": Counter(),
        "has_cot": 0,
        "has_options": 0,
        "avg_question_len": 0,
        "avg_cot_len": 0,
        "difficulty_dist": Counter(),
    }

    question_lengths = []
    cot_lengths = []
    seen_ids = set()

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                stats["errors"].append(
                    {"line": line_num, "error": f"JSON parse error: {e}"}
                )
                continue

            row_errors = []
            row_warnings = []

            # ── Required fields ──────────────────────────────
            for field in REQUIRED_FIELDS:
                if field not in ex or ex[field] is None or ex[field] == "":
                    row_errors.append(f"Missing required field: '{field}'")

            # ── Duplicate ID ─────────────────────────────────
            ex_id = ex.get("id", f"line_{line_num}")
            if ex_id in seen_ids:
                row_warnings.append(f"Duplicate ID: {ex_id}")
            seen_ids.add(ex_id)

            # ── Subject validation ───────────────────────────
            subject = ex.get("subject", "")
            if subject not in VALID_SUBJECTS:
                row_errors.append(
                    f"Invalid subject '{subject}'. "
                    f"Expected one of: {VALID_SUBJECTS}"
                )

            # ── Question type validation ─────────────────────
            q_type = ex.get("question_type", "")
            if q_type and q_type not in VALID_Q_TYPES:
                row_warnings.append(f"Unknown question type: '{q_type}'")

            # ── Question length ──────────────────────────────
            q = ex.get("question", "")
            q_len = len(q.split())
            if q_len < MIN_QUESTION_LEN:
                row_errors.append(
                    f"Question too short ({q_len} words, min {MIN_QUESTION_LEN})"
                )
            elif q_len > MAX_QUESTION_LEN:
                row_warnings.append(f"Question very long ({q_len} words)")
            question_lengths.append(q_len)

            # ── Chain-of-thought ─────────────────────────────
            cot = ex.get("chain_of_thought", "")
            if cot:
                stats["has_cot"] += 1
                cot_len = len(cot.split())
                cot_lengths.append(cot_len)
                if cot_len < MIN_COT_LEN:
                    row_warnings.append(
                        f"CoT very short ({cot_len} words, recommended >{MIN_COT_LEN})"
                    )

            # ── Options for MCQ ──────────────────────────────
            if ex.get("options"):
                stats["has_options"] += 1
                opts = ex["options"]
                if len(opts) < 4:
                    row_warnings.append(
                        f"MCQ has only {len(opts)} options (expected 4)"
                    )

            # ── Answer correctness heuristic ─────────────────
            answer = str(ex.get("correct_answer", ""))
            if not answer:
                row_errors.append("Empty correct_answer")
            elif q_type in ("single_correct", "multiple_correct"):
                valid_answers = set("ABCD")
                for char in answer.upper():
                    if char.isalpha() and char not in valid_answers:
                        row_warnings.append(
                            f"Answer contains invalid option letter: '{char}'"
                        )

            # ── Accumulate ───────────────────────────────────
            if not row_errors:
                stats["valid"] += 1
                stats["subject_dist"][subject] += 1
                stats["type_dist"][q_type or "unknown"] += 1
                diff = ex.get("difficulty", 0)
                if diff:
                    stats["difficulty_dist"][diff] += 1

            if row_errors:
                stats["errors"].append(
                    {"id": ex_id, "line": line_num, "errors": row_errors}
                )
            if row_warnings:
                stats["warnings"].append(
                    {"id": ex_id, "line": line_num, "warnings": row_warnings}
                )

    # Averages
    stats["avg_question_len"] = (
        sum(question_lengths) / len(question_lengths) if question_lengths else 0
    )
    stats["avg_cot_len"] = (
        sum(cot_lengths) / len(cot_lengths) if cot_lengths else 0
    )
    stats["validity_rate"] = (
        stats["valid"] / stats["total"] if stats["total"] else 0
    )
    stats["cot_coverage"] = (
        stats["has_cot"] / stats["total"] if stats["total"] else 0
    )

    return stats


def print_report(stats_list: list[dict]):
    """Print a formatted validation report."""
    sep = "=" * 70

    print(f"\n{sep}")
    print("  JEE-LLM DATASET VALIDATION REPORT")
    print(sep)

    grand_total = 0
    grand_valid = 0

    for stats in stats_list:
        print(f"\n📄 File: {stats['file']}")
        print(f"   Total rows    : {stats['total']:,}")
        print(f"   Valid rows    : {stats['valid']:,}  "
              f"({stats['validity_rate']:.1%})")
        print(f"   CoT coverage  : {stats['has_cot']:,}  "
              f"({stats['cot_coverage']:.1%})")
        print(f"   Avg Q length  : {stats['avg_question_len']:.0f} words")
        print(f"   Avg CoT len   : {stats['avg_cot_len']:.0f} words")

        if stats["subject_dist"]:
            print("   Subject dist  :", dict(stats["subject_dist"]))
        if stats["type_dist"]:
            print("   Type dist     :", dict(stats["type_dist"]))

        if stats["errors"]:
            print(f"   ❌ Errors ({min(5, len(stats['errors']))} of "
                  f"{len(stats['errors'])} shown):")
            for err in stats["errors"][:5]:
                print(f"      Line {err.get('line','?')}: {err.get('errors', err)}")

        if stats["warnings"]:
            print(f"   ⚠️  Warnings: {len(stats['warnings'])} rows")

        grand_total += stats["total"]
        grand_valid += stats["valid"]

    print(f"\n{sep}")
    print(f"  GRAND TOTAL: {grand_valid:,} / {grand_total:,} valid "
          f"({grand_valid/grand_total:.1%} if grand_total > 0 else 0)")
    print(sep)

    # Quality thresholds
    overall_validity = grand_valid / grand_total if grand_total > 0 else 0
    print("\n📊 Quality Thresholds:")
    threshold_checks = [
        ("Validity rate ≥ 95%", overall_validity >= 0.95),
    ]
    for label, passed in threshold_checks:
        icon = "✅" if passed else "❌"
        print(f"   {icon} {label}")

    print()


def main():
    parser = argparse.ArgumentParser(description="JEE Dataset Validator")
    parser.add_argument(
        "--data_dir", type=str, default="dataset/",
        help="Directory containing JSONL dataset files",
    )
    parser.add_argument(
        "--files", nargs="+", default=None,
        help="Specific files to validate (default: all *.jsonl in data_dir)",
    )
    parser.add_argument(
        "--report", action="store_true", default=True,
        help="Print detailed validation report",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with code 1 if any errors found",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.files:
        files = [data_dir / f for f in args.files]
    else:
        files = sorted(data_dir.glob("*.jsonl"))

    if not files:
        logger.error(f"No JSONL files found in {data_dir}")
        return

    all_stats = []
    for filepath in files:
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            continue
        logger.info(f"Validating: {filepath.name}")
        stats = validate_file(str(filepath))
        all_stats.append(stats)

    if args.report:
        print_report(all_stats)

    if args.strict:
        total_errors = sum(len(s["errors"]) for s in all_stats)
        if total_errors > 0:
            logger.error(f"Strict mode: {total_errors} errors found. Exiting.")
            exit(1)
        logger.info("✅ Strict validation passed!")


if __name__ == "__main__":
    main()
