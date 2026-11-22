BASE  ?= Qwen/Qwen2.5-7B-Instruct
EVAL  ?= dataset/jeebench.jsonl
PY    := PYTHONPATH=src python

.PHONY: install test test-fast lint data sft merge-sft grpo grpo-marks pairs dpo eval-base eval-sft eval-grpo smoke

install:
	pip install -e ".[train,dev]"

test:
	pytest -q

test-fast:
	pytest -q -m "not smoke"

lint:
	ruff check src scripts tests

data:
	$(PY) scripts/prepare_data.py --out dataset/
	$(PY) scripts/validate_dataset.py dataset/train_sft.jsonl dataset/jeebench.jsonl

sft:
	$(PY) -m jee_llm.train_sft configs/sft_qlora.yaml model.name=$(BASE)

merge-sft:
	$(PY) scripts/merge_adapters.py --base $(BASE) --adapter checkpoints/sft/final --out checkpoints/sft-merged

grpo:
	$(PY) -m jee_llm.train_grpo configs/grpo.yaml

# abstention experiment: exam-marks reward + SKIP allowed
grpo-marks:
	$(PY) -m jee_llm.train_grpo configs/grpo.yaml output_dir=checkpoints/grpo-marks \
		"reward.funcs=[jee_marks, format]" data.allow_skip=true training.run_name=grpo-jee-marks

pairs:
	$(PY) scripts/build_preference_pairs.py --model checkpoints/sft-merged --data dataset/train_rl.jsonl \
		--k 8 --limit 10000 --out dataset/dpo_pairs.jsonl

dpo:
	$(PY) -m jee_llm.train_dpo configs/dpo.yaml

EVAL_ARGS = --data $(EVAL) --num-samples 8 --temperature 0.7 --max-new-tokens 4096

eval-base:
	$(PY) -m jee_llm.evaluate --model $(BASE) $(EVAL_ARGS) --out results/base

eval-sft:
	$(PY) -m jee_llm.evaluate --model checkpoints/sft-merged $(EVAL_ARGS) --out results/sft

eval-grpo:
	$(PY) -m jee_llm.evaluate --model checkpoints/sft-merged --adapter checkpoints/grpo/final $(EVAL_ARGS) --out results/grpo

# whole pipeline on a tiny random model, CPU only, ~30s
smoke:
	pytest -q -m smoke
