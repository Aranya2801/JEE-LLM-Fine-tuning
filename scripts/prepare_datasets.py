"""
prepare_datasets.py — Full Dataset Preparation Pipeline
========================================================
Downloads, processes and prepares all datasets for JEE-LLM training.
Run this once before training.

Usage:
    python scripts/prepare_datasets.py \
        --output_dir dataset/ \
        --datasets jee_advanced jee_mains jeebench math gsm8k \
        --generate_cot \
        --num_workers 8

Author: Aranya Ghosh
"""

import os
import json
import logging
import argparse
import hashlib
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Subject / topic taxonomy
# ─────────────────────────────────────────────────────────────

MATH_TOPICS = [
    "Algebra", "Trigonometry", "Coordinate Geometry", "Calculus",
    "Integral Calculus", "Differential Equations", "Vectors and 3D",
    "Probability and Statistics", "Complex Numbers", "Matrices and Determinants",
    "Sequences and Series", "Permutations and Combinations", "Binomial Theorem",
    "Conic Sections", "Limits and Continuity",
]

PHYSICS_TOPICS = [
    "Kinematics", "Newton's Laws", "Work Energy Power", "Rotational Motion",
    "Gravitation", "Simple Harmonic Motion", "Waves", "Thermodynamics",
    "Electrostatics", "Current Electricity", "Magnetism", "Electromagnetic Induction",
    "Alternating Current", "Optics", "Modern Physics", "Semiconductors",
]

CHEMISTRY_TOPICS = [
    "Atomic Structure", "Chemical Bonding", "Thermodynamics", "Equilibrium",
    "Electrochemistry", "Chemical Kinetics", "Solid State", "Solutions",
    "Surface Chemistry", "Coordination Chemistry", "Organic Chemistry - Basics",
    "Hydrocarbons", "Halogenated Compounds", "Alcohols Phenols Ethers",
    "Aldehydes Ketones Carboxylic Acids", "Amines", "Polymers", "Biomolecules",
    "Periodic Table", "p-Block Elements", "d-Block Elements",
]

SUBJECT_TOPICS = {
    "Mathematics": MATH_TOPICS,
    "Physics": PHYSICS_TOPICS,
    "Chemistry": CHEMISTRY_TOPICS,
}


# ─────────────────────────────────────────────────────────────
# HuggingFace dataset loaders
# ─────────────────────────────────────────────────────────────

def download_math_dataset(output_dir: Path) -> int:
    """Download MATH competition dataset and convert to JEE format."""
    logger.info("Downloading MATH competition dataset...")
    try:
        ds = load_dataset("hendrycks/competition_mathematics", split="train+test", trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load MATH dataset: {e}. Skipping.")
        return 0

    out_path = output_dir / "math_competition.jsonl"
    count = 0
    with open(out_path, "w") as f:
        for i, ex in enumerate(tqdm(ds, desc="MATH")):
            record = {
                "id": f"math_{i:05d}",
                "year": None,
                "paper": None,
                "subject": "Mathematics",
                "topic": ex.get("type", "Algebra"),
                "question_type": "open_ended",
                "question": ex["problem"],
                "options": {},
                "correct_answer": ex["solution"].split("=")[-1].strip()
                    if "=" in ex.get("solution", "") else "",
                "chain_of_thought": ex.get("solution", ""),
                "marks": 4,
                "negative_marks": 0,
                "difficulty": min(5, int(ex.get("level", "Level 3").split()[-1])),
                "source": "MATH Competition Dataset",
                "augmented": False,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"  Saved {count} MATH problems to {out_path}")
    return count


def download_gsm8k(output_dir: Path) -> int:
    """Download GSM8K dataset."""
    logger.info("Downloading GSM8K dataset...")
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
    except Exception as e:
        logger.warning(f"Could not load GSM8K: {e}. Skipping.")
        return 0

    out_path = output_dir / "gsm8k_train.jsonl"
    count = 0
    with open(out_path, "w") as f:
        for i, ex in enumerate(tqdm(ds, desc="GSM8K")):
            answer_raw = ex["answer"]
            # GSM8K uses #### to separate reasoning from answer
            parts = answer_raw.split("####")
            cot = parts[0].strip()
            answer = parts[1].strip() if len(parts) > 1 else ""

            record = {
                "id": f"gsm8k_{i:05d}",
                "year": None,
                "paper": None,
                "subject": "Mathematics",
                "topic": "Arithmetic and Word Problems",
                "question_type": "numerical",
                "question": ex["question"],
                "options": {},
                "correct_answer": answer,
                "chain_of_thought": cot,
                "marks": 3,
                "negative_marks": 0,
                "difficulty": 2,
                "source": "GSM8K",
                "augmented": False,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"  Saved {count} GSM8K problems to {out_path}")
    return count


def download_jeebench(output_dir: Path) -> int:
    """Download JEEBench evaluation dataset."""
    logger.info("Downloading JEEBench evaluation set...")
    try:
        ds = load_dataset("dair-iitd/jeebench", split="test", trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load JEEBench: {e}. Skipping.")
        return 0

    out_path = output_dir / "jeebench_eval.jsonl"
    count = 0
    with open(out_path, "w") as f:
        for i, ex in enumerate(tqdm(ds, desc="JEEBench")):
            # Map JEEBench fields to our schema
            q_type_map = {
                "MCQ": "single_correct",
                "MCQ(multiple)": "multiple_correct",
                "Integer": "integer",
                "Numeric": "numerical",
            }
            raw_type = ex.get("type", "MCQ")
            q_type = q_type_map.get(raw_type, "single_correct")

            record = {
                "id": ex.get("index", f"jeebench_{i:04d}"),
                "year": ex.get("year", 2016),
                "paper": ex.get("paper", 1),
                "subject": ex.get("subject", "Mathematics").title(),
                "topic": ex.get("chapter", "General"),
                "question_type": q_type,
                "question": ex.get("question", ""),
                "options": {
                    "A": ex.get("option_a", ""),
                    "B": ex.get("option_b", ""),
                    "C": ex.get("option_c", ""),
                    "D": ex.get("option_d", ""),
                } if q_type in ("single_correct", "multiple_correct") else {},
                "correct_answer": str(ex.get("gold", "")),
                "chain_of_thought": "",  # No CoT in JEEBench — will be generated
                "marks": ex.get("marks", 3),
                "negative_marks": ex.get("negative_marks", -1),
                "difficulty": 3,
                "source": f"JEEBench - JEE Advanced {ex.get('year', '')}",
                "augmented": False,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"  Saved {count} JEEBench problems to {out_path}")
    return count


def download_mmlu_stem(output_dir: Path) -> int:
    """Download MMLU STEM subsets."""
    logger.info("Downloading MMLU STEM subsets...")
    mmlu_subjects = {
        "high_school_mathematics": "Mathematics",
        "high_school_physics": "Physics",
        "high_school_chemistry": "Chemistry",
        "college_mathematics": "Mathematics",
        "college_physics": "Physics",
        "college_chemistry": "Chemistry",
    }

    out_path = output_dir / "mmlu_stem.jsonl"
    total = 0
    with open(out_path, "w") as f:
        for mmlu_subj, jee_subj in mmlu_subjects.items():
            try:
                ds = load_dataset("cais/mmlu", mmlu_subj, split="test")
            except Exception as e:
                logger.warning(f"  Could not load MMLU/{mmlu_subj}: {e}")
                continue

            for i, ex in enumerate(tqdm(ds, desc=f"MMLU/{mmlu_subj}")):
                options = {
                    "A": ex["choices"][0] if len(ex["choices"]) > 0 else "",
                    "B": ex["choices"][1] if len(ex["choices"]) > 1 else "",
                    "C": ex["choices"][2] if len(ex["choices"]) > 2 else "",
                    "D": ex["choices"][3] if len(ex["choices"]) > 3 else "",
                }
                answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
                record = {
                    "id": f"mmlu_{mmlu_subj}_{i:04d}",
                    "year": None,
                    "paper": None,
                    "subject": jee_subj,
                    "topic": mmlu_subj.replace("_", " ").title(),
                    "question_type": "single_correct",
                    "question": ex["question"],
                    "options": options,
                    "correct_answer": answer_map.get(ex["answer"], "A"),
                    "chain_of_thought": "",
                    "marks": 3,
                    "negative_marks": -1,
                    "difficulty": 3,
                    "source": f"MMLU {mmlu_subj}",
                    "augmented": False,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1

    logger.info(f"  Saved {total} MMLU STEM problems to {out_path}")
    return total


# ─────────────────────────────────────────────────────────────
# Sample JEE data creator (for users without full dataset)
# ─────────────────────────────────────────────────────────────

def create_sample_training_data(output_dir: Path) -> int:
    """
    Copy sample data to training format for quick testing.
    Users should replace this with the full dataset.
    """
    sample_path = Path("dataset/samples/sample_jee_cot.jsonl")
    if not sample_path.exists():
        logger.warning("Sample data not found. Skipping.")
        return 0

    out_path = output_dir / "jee_sample_training.jsonl"
    count = 0
    with open(sample_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            if line.strip():
                f_out.write(line)
                count += 1

    logger.info(f"  Copied {count} sample problems to {out_path}")
    logger.info(
        "  ⚠️  NOTE: Replace with full JEE dataset (2003-2024) for production training."
    )
    return count


# ─────────────────────────────────────────────────────────────
# Dataset statistics reporter
# ─────────────────────────────────────────────────────────────

def report_dataset_stats(output_dir: Path):
    """Print a summary of all prepared datasets."""
    import glob

    logger.info("\n" + "=" * 55)
    logger.info("  DATASET PREPARATION SUMMARY")
    logger.info("=" * 55)
    logger.info(f"{'File':<40} {'Count':>8}")
    logger.info("-" * 55)

    total = 0
    for path in sorted(glob.glob(str(output_dir / "*.jsonl"))):
        with open(path) as f:
            count = sum(1 for line in f if line.strip())
        total += count
        name = Path(path).name
        logger.info(f"{name:<40} {count:>8,}")

    logger.info("-" * 55)
    logger.info(f"{'TOTAL':<40} {total:>8,}")
    logger.info("=" * 55 + "\n")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

DATASET_HANDLERS = {
    "math": download_math_dataset,
    "gsm8k": download_gsm8k,
    "jeebench": download_jeebench,
    "mmlu": download_mmlu_stem,
    "sample": create_sample_training_data,
}


def main():
    parser = argparse.ArgumentParser(
        description="JEE-LLM Dataset Preparation Pipeline"
    )
    parser.add_argument(
        "--output_dir", type=str, default="dataset/",
        help="Directory to save prepared datasets",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["sample", "jeebench", "math", "gsm8k"],
        choices=list(DATASET_HANDLERS.keys()) + ["all"],
        help="Which datasets to download and prepare",
    )
    parser.add_argument(
        "--generate_cot", action="store_true",
        help="Generate CoT solutions via GPT-4o (requires OPENAI_API_KEY)",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_run = (
        list(DATASET_HANDLERS.keys())
        if "all" in args.datasets
        else args.datasets
    )

    logger.info(f"Preparing datasets: {datasets_to_run}")
    logger.info(f"Output directory: {output_dir}")

    total_count = 0
    for ds_name in datasets_to_run:
        handler = DATASET_HANDLERS.get(ds_name)
        if handler is None:
            logger.warning(f"Unknown dataset: {ds_name}")
            continue
        count = handler(output_dir)
        total_count += count

    logger.info(f"\n✅ Total examples prepared: {total_count:,}")

    # Optional CoT generation
    if args.generate_cot:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error(
                "OPENAI_API_KEY not set. "
                "Cannot generate CoT. Set it in .env file."
            )
        else:
            logger.info("Starting CoT generation via GPT-4o...")
            from src.data.dataset_builder import CoTGenerator

            gen = CoTGenerator(api_key=api_key)
            # Generate CoT for JEEBench problems that lack it
            jeebench_path = output_dir / "jeebench_eval.jsonl"
            if jeebench_path.exists():
                with open(jeebench_path) as f:
                    examples = [json.loads(l) for l in f if l.strip()]
                no_cot = [ex for ex in examples if not ex.get("chain_of_thought")]
                if no_cot:
                    logger.info(
                        f"Generating CoT for {len(no_cot)} JEEBench problems..."
                    )
                    gen.generate_batch(
                        no_cot,
                        num_workers=args.num_workers,
                        output_path=str(output_dir / "jeebench_with_cot.jsonl"),
                    )

    report_dataset_stats(output_dir)
    logger.info("🎉 Dataset preparation complete!")
    logger.info(
        "\nNext steps:\n"
        "  1. Run: make train-sft          # Stage 1: QLoRA SFT\n"
        "  2. Run: make train-dpo          # Stage 2: DPO alignment\n"
        "  3. Run: make train-grpo         # Stage 2: GRPO training\n"
        "  4. Run: make serve-gradio       # Start daily-use interface\n"
    )


if __name__ == "__main__":
    main()
