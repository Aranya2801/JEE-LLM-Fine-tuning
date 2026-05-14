# ============================================================
# Makefile — JEE-LLM-Fine-tuning Development Commands
# ============================================================

.PHONY: all install install-dev lint format test clean train-sft train-dpo train-grpo evaluate serve-gradio serve-api help

# ── Colors ──────────────────────────────────────────────────
BOLD   := $(shell tput bold)
GREEN  := $(shell tput setaf 2)
YELLOW := $(shell tput setaf 3)
BLUE   := $(shell tput setaf 4)
RESET  := $(shell tput sgr0)

# ── Variables ────────────────────────────────────────────────
MODEL_PATH   ?= checkpoints/jee-llm-v1
DEVICE       ?= cuda
PORT         ?= 7860
API_PORT     ?= 8000

help: ## Show this help message
	@echo "$(BOLD)$(BLUE)JEE-LLM Development Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

# ── Installation ─────────────────────────────────────────────
install: ## Install production dependencies
	@echo "$(YELLOW)Installing dependencies...$(RESET)"
	pip install torch --index-url https://download.pytorch.org/whl/cu118
	pip install -r requirements.txt
	pip install flash-attn --no-build-isolation
	@echo "$(GREEN)✅ Installation complete!$(RESET)"

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install
	@echo "$(GREEN)✅ Dev environment ready!$(RESET)"

# ── Code Quality ─────────────────────────────────────────────
lint: ## Run all linters (flake8, mypy)
	@echo "$(YELLOW)Running linters...$(RESET)"
	flake8 src/ scripts/ tests/ --max-line-length 100 --ignore E501,W503
	@echo "$(GREEN)✅ Lint passed!$(RESET)"

format: ## Format code with black and isort
	@echo "$(YELLOW)Formatting code...$(RESET)"
	black src/ scripts/ tests/ --line-length 100
	isort src/ scripts/ tests/
	@echo "$(GREEN)✅ Code formatted!$(RESET)"

format-check: ## Check formatting without modifying files
	black --check --diff src/ scripts/ tests/ --line-length 100
	isort --check-only --diff src/ scripts/ tests/

# ── Testing ──────────────────────────────────────────────────
test: ## Run all unit tests
	@echo "$(YELLOW)Running tests...$(RESET)"
	pytest tests/ -v --timeout=120 --tb=short
	@echo "$(GREEN)✅ All tests passed!$(RESET)"

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# ── Dataset Preparation ──────────────────────────────────────
prepare-data: ## Download and prepare all datasets
	@echo "$(YELLOW)Preparing datasets...$(RESET)"
	python scripts/prepare_datasets.py \
		--output_dir dataset/ \
		--num_workers 8
	@echo "$(GREEN)✅ Datasets ready!$(RESET)"

validate-data: ## Validate dataset quality
	python scripts/validate_dataset.py --data_dir dataset/ --report

# ── Training ─────────────────────────────────────────────────
train-sft: ## Run Stage 1: QLoRA SFT training
	@echo "$(YELLOW)Starting SFT training...$(RESET)"
	bash scripts/train_qlora.sh

train-dpo: ## Run Stage 2c: DPO alignment
	@echo "$(YELLOW)Starting DPO training...$(RESET)"
	python src/training/train_dpo.py --config configs/dpo.yaml

train-grpo: ## Run Stage 2b: GRPO training
	@echo "$(YELLOW)Starting GRPO training...$(RESET)"
	python src/training/train_rlhf.py --config configs/grpo.yaml

train-all: train-sft train-dpo train-grpo ## Run full training pipeline (Stage 1 → 2 → 3)
	@echo "$(GREEN)🎉 Full training pipeline complete!$(RESET)"

merge: ## Merge LoRA adapters into base model
	python scripts/merge_adapters.py \
		--base_model meta-llama/Meta-Llama-3-8B-Instruct \
		--adapter_path checkpoints/stage2_dpo/final \
		--output_path checkpoints/jee-llm-v1-merged

# ── Evaluation ───────────────────────────────────────────────
evaluate: ## Run full evaluation suite
	@echo "$(YELLOW)Running evaluation...$(RESET)"
	python src/evaluation/evaluate.py \
		--model_path $(MODEL_PATH) \
		--benchmarks jeebench \
		--subjects mathematics physics chemistry \
		--output_dir results/

evaluate-quick: ## Quick evaluation on small sample
	python src/evaluation/evaluate.py \
		--model_path $(MODEL_PATH) \
		--benchmarks jeebench \
		--num_samples 50 \
		--output_dir results/quick/

# ── Inference ────────────────────────────────────────────────
serve-gradio: ## Launch Gradio web interface
	@echo "$(GREEN)🚀 Launching Gradio UI at http://localhost:$(PORT)$(RESET)"
	python src/inference/gradio_app.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--port $(PORT)

serve-gradio-public: ## Launch Gradio with public URL
	python src/inference/gradio_app.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--port $(PORT) \
		--share

serve-api: ## Launch FastAPI inference server
	@echo "$(GREEN)🚀 API server at http://localhost:$(API_PORT)$(RESET)"
	JEE_MODEL_PATH=$(MODEL_PATH) JEE_DEVICE=$(DEVICE) \
	python src/inference/api_server.py \
		--model_path $(MODEL_PATH) \
		--port $(API_PORT)

# ── Export ───────────────────────────────────────────────────
export-gguf: ## Export model to GGUF (for llama.cpp)
	python scripts/export_gguf.py --model_path $(MODEL_PATH)

push-hub: ## Push model to HuggingFace Hub
	python scripts/merge_adapters.py \
		--base_model meta-llama/Meta-Llama-3-8B-Instruct \
		--adapter_path checkpoints/stage2_dpo/final \
		--output_path checkpoints/jee-llm-v1-merged \
		--push_to_hub \
		--hub_repo Aranya2801/JEE-LLM-v1

# ── Cleanup ──────────────────────────────────────────────────
clean: ## Remove generated files and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov/ 2>/dev/null || true
	@echo "$(GREEN)✅ Cleaned!$(RESET)"

clean-checkpoints: ## Remove all training checkpoints (CAREFUL!)
	@echo "$(YELLOW)WARNING: This will delete all checkpoints!$(RESET)"
	@read -p "Are you sure? (yes/no): " yn; \
	if [ "$$yn" = "yes" ]; then rm -rf checkpoints/; echo "Deleted."; fi

# ── Docker ───────────────────────────────────────────────────
docker-build: ## Build Docker image
	docker build -t jee-llm:latest .

docker-run: ## Run Docker container
	docker run --gpus all -p $(PORT):7860 -p $(API_PORT):8000 \
		-v $(PWD)/checkpoints:/app/checkpoints \
		jee-llm:latest

# ── Info ─────────────────────────────────────────────────────
info: ## Show environment info
	@echo "$(BOLD)Environment Info$(RESET)"
	python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
	python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	python -c "import peft; print(f'PEFT: {peft.__version__}')"
	nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPU found"
