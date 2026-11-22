# Experiment protocol

Notes on how the runs in the README table are supposed to be done, so they're
comparable and someone else could redo them.

## Fixed across all runs

- Base: `Qwen/Qwen2.5-7B-Instruct` (and `Qwen2.5-1.5B-Instruct` for the cheap version of every run)
- Prompt: `prompts.SYSTEM_PROMPT` + `format_question`, the same in training and eval. The
  only exception is `--allow-skip`, which adds one sentence telling the model `SKIP` is allowed.
- Eval: all 515 JEEBench questions, 8 samples, temperature 0.7, top-p 0.95, 4096 max new tokens, seed 0
- Training data: whatever `prepare_data.py` produces with default flags. Record the
  line counts of `train_sft.jsonl` / `train_rl.jsonl` and the size of
  `contamination.json` in the run notes.
- Seeds: 42 for training. For the headline numbers, repeat GRPO with seeds 43 and 44 and
  report mean ± std, since RL runs vary a lot between seeds.

## Runs

| id | what | command |
|---|---|---|
| R0 | base model | `make eval-base` |
| R1 | SFT | `make sft merge-sft eval-sft` |
| R2 | R1 + GRPO, 0/1 reward | `make grpo eval-grpo` |
| R3 | R1 + GRPO, marks reward, skip allowed | `make grpo-marks`, then eval with `--allow-skip` |
| R4 | R1 + DPO on rejection-sampled pairs | `make pairs dpo`, eval like R2 |
| R5 | R0 + GRPO directly (no SFT) | `python -m jee_llm.train_grpo configs/grpo.yaml model.name=Qwen/Qwen2.5-7B-Instruct output_dir=checkpoints/grpo-zero` |

R5 is there to check whether SFT is even needed before RL (the "R1-Zero" question).

For R2 vs R4 (RQ4), the comparison should be at matched sampling compute: DPO pairs
come from k=8 samples over N prompts, so give GRPO the same N prompts × 8
rollouts (one epoch over the same prompts).

## What to report

From `metrics.json` of every run:

- `overall`: `pass@1`, `maj@8`, `jee_score_pct`, `no_answer_rate`
- `by_subject`: `pass@1` for maths / physics / chemistry (RQ2)
- `by_type`: mainly to see if multi-correct MCQs behave differently, since their
  -2 penalty is the harshest
- `abstention`: the curve over tau for R2 vs. R3 (RQ3)

For RQ3 specifically, split R3's MCQ answers into answered and skipped, and check
what R2 got on the questions R3 skipped. If skipping is calibrated, R2's accuracy on
those questions should be well below its overall accuracy.

## Things to watch during GRPO

- `reward` and `completions/mean_length` in W&B. A length blow-up with a flat
  reward usually means the format reward is being gamed or completions are being truncated.
- `completions/clipped_ratio`: if many completions hit `max_completion_length`,
  `mask_truncated_completions` is dropping them from the loss. Raise the limit or
  expect slow learning.
- With `allow_skip`, the share of `\boxed{SKIP}` completions. If it heads to 100%,
  the -0.5 penalty for a wrong multi-correct answer is too strong relative to the
  success rate. Lower it in `scoring.MARKS` or start from a stronger SFT model.

## Run log

| date | id | model | notes | metrics |
|---|---|---|---|---|
| | | | | |
