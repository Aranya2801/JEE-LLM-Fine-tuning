# JEE-LLM: fine-tuning open models for JEE Advanced

SFT and reinforcement learning with verifiable rewards (GRPO) for small open
LLMs on JEE Advanced style maths, physics and chemistry problems, evaluated on
[JEEBench](https://github.com/dair-iitd/jeebench) with the real exam marking
scheme, including negative marking.

The part I'm most interested in is JEE's negative marking. Most math-RL work
uses a 0/1 correctness reward, so a confident wrong answer costs the same as
saying nothing. In the actual exam a wrong MCQ answer costs marks, which means
a good test-taker also has to know when *not* to answer. This repo makes that
trade-off something you can train on and measure.

**Status:** the pipeline (data prep, decontamination, SFT, GRPO, DPO baseline,
evaluation) is implemented and tested end to end on a tiny model in CI. The
full-scale runs on a 7B model haven't been done yet, so there are **no results
reported here yet**. The tables below are filled in from `results/*/metrics.json`
as runs finish.

## Research questions

1. **Does SFT on competition maths help on JEE at all?** JEE Advanced maths is
   close to olympiad/MATH style problems, but physics and chemistry are not, and
   open SFT data for them with checkable answers is thin.
2. **Does RL with verifiable rewards transfer across subjects?** GRPO is trained
   on maths only (that's where checkable answers are plentiful). Does physics and
   chemistry accuracy move too?
3. **Can an exam-marks reward teach abstention?** Compare GRPO with a 0/1
   reward vs. a reward equal to the JEE marks (wrong MCQ = -1, wrong
   multi-correct = -2, partial credit, `SKIP` = 0). Measured by exam score and by
   how much the model's skips line up with the questions it would have got wrong.
4. **GRPO vs. rejection-sampled DPO** on the same prompts and compute budget.

## How it works

```
 MATH, GSM8K, NuminaMath (optional), your own JEE data
        |
        |  prepare_data.py: convert to one schema, drop anything that shares
        |  a 13-gram with a JEEBench question, split off a validation set
        v
 train_sft.jsonl / train_rl.jsonl                     jeebench.jsonl (test only)
        |                                                     ^
   SFT (QLoRA) ---> merge ---> GRPO, reward = answer checker  |
                          \--> sample k, keep (right, wrong) pairs ---> DPO
                                                              |
                         evaluate.py: accuracy, pass@k, maj@k, JEE marks,
                         abstention curve, per subject / question type
```

There's no learned reward model. The reward comes from
`src/jee_llm/answers.py`, which pulls the last `\boxed{}` out of a completion
and checks it against the gold answer: option sets for MCQs (`AC` ==
`(A), (C)`), integers exactly, numerical answers to two decimal places (with a
relative tolerance for very small or large values), and simple LaTeX like
`\frac{1}{2}` or `2\sqrt{2}`. It's the most tested module in the repo, because
a parser bug here quietly turns into a reward bug.

### Marking scheme

Same as the JEEBench paper, which follows the 2016-2023 papers:

| type | correct | wrong | other |
|---|---|---|---|
| single correct MCQ | +3 | -1 | skip 0 |
| multiple correct MCQ | +4 | -2 if any wrong option is picked | +1 per correct option if no wrong one is picked |
| integer / numerical | +3 | 0 | |

A completion with no extractable answer counts as wrong, not as a skip.
Otherwise a model could "abstain" by rambling.

## Setup

```bash
git clone https://github.com/Aranya2801/JEE-LLM-Fine-tuning.git
cd JEE-LLM-Fine-tuning
pip install -e ".[train,dev]"      # add ".[vllm]" on a CUDA box for fast sampling
cp .env.example .env               # HF / W&B tokens

make test                          # unit tests + tiny-model smoke test, CPU is fine
```

Hardware, roughly: QLoRA SFT of a 7B model fits on a single 24 GB GPU at 2k
context. GRPO with 8 rollouts per prompt and vLLM colocated wants an 80 GB card
for 7B. For a single 24 GB GPU, use `Qwen/Qwen2.5-1.5B-Instruct`
(`BASE=Qwen/Qwen2.5-1.5B-Instruct`, and set `model.name` to match in the GRPO/DPO
configs).

## Running it

```bash
make data          # downloads JEEBench + MATH + GSM8K, decontaminates, validates
make eval-base     # baseline numbers first

make sft           # stage 1
make merge-sft
make eval-sft

make grpo          # stage 2, 0/1 correctness reward
make grpo-marks    # stage 2, exam-marks reward + SKIP allowed (RQ3)
make eval-grpo

make pairs && make dpo   # DPO baseline (RQ4)
```

Every stage is `python -m jee_llm.<stage> <config.yaml> [key=value ...]`, with
dotted overrides, e.g.

```bash
python -m jee_llm.train_sft configs/sft_qlora.yaml model.name=Qwen/Qwen2.5-1.5B-Instruct training.learning_rate=2e-4
```

Everything under `training:` in a config goes straight into the matching TRL
config class (`SFTConfig`, `GRPOConfig`, `DPOConfig`), so any TRL option works
without code changes.

### Adding your own JEE data

Write it as jsonl in the schema in `src/jee_llm/data.py` (see
`dataset/samples/jee_style_samples.jsonl` for examples), then

```bash
python scripts/validate_dataset.py my_data.jsonl      # schema + "does the solution reach the gold answer"
python scripts/check_contamination.py --train my_data.jsonl --eval dataset/jeebench.jsonl
python scripts/prepare_data.py --out dataset/ --extra my_data.jsonl
```

The contamination check matters. JEE Advanced papers are copied all over
coaching sites, so anything scraped will contain JEEBench questions unless you
filter them out.

## Evaluation

```bash
python -m jee_llm.evaluate --model checkpoints/sft-merged --adapter checkpoints/grpo/final \
    --data dataset/jeebench.jsonl --num-samples 8 --temperature 0.7 --out results/grpo
```

This writes `completions.jsonl` (every sample, the extracted answer and the
verdict, for error analysis), `metrics.json` and a `report.md` with:

- `accuracy@1` (first sample; greedy when `--temperature 0`), `pass@1` averaged
  over samples, unbiased `pass@k`, and `maj@k` (self-consistency)
- JEE marks with negative marking, as % of max
- the **abstention curve**: the exam score if the model only answers MCQs where
  the majority vote has at least a share tau of the samples (the JEEBench paper's
  thresholding trick, here as a baseline for the trained abstention in RQ3)
- the same broken down by subject and by question type

`--rescore results/x/completions.jsonl` recomputes everything from saved
completions without generating again, for when the answer checker changes.

## Results

Base model `Qwen2.5-7B-Instruct`, JEEBench (515 questions), 8 samples at T=0.7,
max 4096 new tokens. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the
exact protocol.

| model | maths | physics | chemistry | overall pass@1 | maj@8 | JEE marks % |
|---|---|---|---|---|---|---|
| base | - | - | - | - | - | - |
| + SFT | - | - | - | - | - | - |
| + SFT + GRPO (0/1 reward) | - | - | - | - | - | - |
| + SFT + GRPO (marks reward, skip allowed) | - | - | - | - | - | - |
| + SFT + DPO | - | - | - | - | - | - |

*Pending runs.*

## Layout

```
src/jee_llm/
  answers.py        answer extraction + checking (the verifier)
  scoring.py        JEE marking scheme
  rewards.py        GRPO reward functions
  metrics.py        pass@k, majority vote, abstention curve
  contamination.py  n-gram overlap check
  data.py           Problem schema + adapters for JEEBench / MATH / GSM8K
  prompts.py        the single prompt format used everywhere
  train_sft.py  train_grpo.py  train_dpo.py  evaluate.py  generation.py
configs/            one yaml per stage
scripts/            data prep, validation, preference pairs, merging, GGUF export
tests/              unit tests + end-to-end smoke test on a tiny random model
```

## Known limitations

- The answer checker handles numbers, option letters and simple LaTeX. It
  doesn't do full symbolic equivalence (`\frac{\sqrt3}{2}` vs `\sin 60^\circ`), so
  open-ended answers that need it are left out of the RL set instead of being
  scored wrong.
- There's much less open training data with checkable answers for physics and
  chemistry than for maths. RQ2 is partly a consequence of that.
- Some JEE questions depend on a figure. The model only gets the text.
- Partial marking for multi-correct questions changed in recent years (2022+);
  the JEEBench scheme is used for comparability. `scoring.MARKS` can be edited.

## References

- Arora et al., *Have LLMs Advanced Enough? A Challenging Problem Solving Benchmark for Large Language Models* (JEEBench), EMNLP 2023
- Shao et al., *DeepSeekMath* (GRPO), 2024
- Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO), 2025
- Dettmers et al., *QLoRA*, 2023
- Rafailov et al., *Direct Preference Optimization*, 2023
- Chen et al., *Evaluating Large Language Models Trained on Code* (unbiased pass@k), 2021
- Brown et al., *Language Models are Few-Shot Learners* (13-gram contamination analysis), 2020

## License

MIT
