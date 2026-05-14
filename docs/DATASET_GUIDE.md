# Dataset Guide — JEE-LLM

## Dataset Schema

Every JEE problem in the dataset follows this JSON schema:

```json
{
  "id":            "jee_adv_2023_p1_q07",     // Unique identifier
  "year":          2023,                        // Exam year
  "paper":         1,                           // Paper number (1 or 2)
  "subject":       "Mathematics",               // Mathematics | Physics | Chemistry
  "topic":         "Integral Calculus",         // Specific topic
  "subtopic":      "Integration by Parts",      // (optional) subtopic
  "question_type": "single_correct",           // See types below
  "question":      "Let f(x) = ...",           // Problem text (may contain LaTeX)
  "options": {                                  // Only for MCQ
    "A": "1/13",
    "B": "1/14",
    "C": "1/12",
    "D": "1/11"
  },
  "correct_answer": "A",                        // Gold answer
  "chain_of_thought": "Step 1: ...",           // CoT solution
  "marks":         3,                           // Marks for correct answer
  "negative_marks": -1,                        // Negative marking
  "difficulty":    4,                           // 1 (easy) to 5 (very hard)
  "source":        "JEE Advanced 2023 Paper 1",
  "augmented":     false                        // True if data-augmented
}
```

## Question Types

| Type | Code | Description |
|------|------|-------------|
| Single Correct MCQ | `single_correct` | Exactly one correct option |
| Multiple Correct MCQ | `multiple_correct` | One or more correct options |
| Numerical Answer | `numerical` | A decimal number (±0.01 tolerance) |
| Integer Type | `integer` | Integer from 0 to 9 |
| Matrix Match | `matrix_match` | Match columns |
| Paragraph-based | `paragraph` | Multiple Qs on a passage |

## External Datasets to Download

Run these commands to prepare supplementary datasets:

```bash
# MATH dataset (competition mathematics)
python -c "
from datasets import load_dataset
ds = load_dataset('hendrycks/competition_mathematics', split='train')
ds.to_json('dataset/math_competition.jsonl')
"

# GSM8K (grade school math)
python -c "
from datasets import load_dataset
ds = load_dataset('openai/gsm8k', 'main', split='train')
ds.to_json('dataset/gsm8k_train.jsonl')
"

# JEEBench evaluation set
python -c "
from datasets import load_dataset
ds = load_dataset('dair-iitd/jeebench', split='test')
ds.to_json('dataset/jeebench_eval.jsonl')
"

# MMLU STEM subset
python -c "
from datasets import load_dataset
for subj in ['high_school_mathematics', 'high_school_physics', 'high_school_chemistry']:
    ds = load_dataset('cais/mmlu', subj, split='test')
    ds.to_json(f'dataset/mmlu_{subj}.jsonl')
"
```

## Generating CoT with GPT-4o

```bash
export OPENAI_API_KEY=sk-...
python src/data/dataset_builder.py \
    --raw_data dataset/jee_raw.jsonl \
    --output_dir dataset/ \
    --generate_cot \
    --num_workers 16
```

**Cost estimate**: ~$0.015 per problem with GPT-4o (avg 500 input + 500 output tokens).
50K problems ≈ $750. Use `--num_workers 16` for fastest throughput.

## Data Quality Checklist

Before training, verify your dataset:
```bash
python scripts/validate_dataset.py --data_dir dataset/ --report
```

Quality thresholds:
- Validity rate ≥ 95%
- CoT length ≥ 100 words on average
- Subject distribution: Math ~40%, Physics ~35%, Chemistry ~25%
- At least 20% numerical/integer type questions

## Additional Public Resources

| Resource | URL | Content |
|----------|-----|---------|
| JEEBench | [GitHub](https://github.com/dair-iitd/jeebench) | 515 eval problems |
| JEE Papers (Official) | [NTA Website](https://jeemain.nta.ac.in) | Official papers |
| HC Verma Solutions | Various | Physics reference |
| NCERT Solutions | [NCERT](https://ncert.nic.in) | Textbook solutions |
| AoPS Math Problems | [AoPS](https://artofproblemsolving.com) | Competition math |
| PhysicsWallah | YouTube | Video explanations |
