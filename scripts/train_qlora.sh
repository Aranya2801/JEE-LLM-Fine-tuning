#!/usr/bin/env bash
# =============================================================
# scripts/train_qlora.sh — Single-GPU QLoRA Training Script
# =============================================================
# Usage:
#   bash scripts/train_qlora.sh \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --subject all \
#     --config configs/sft_qlora.yaml
# =============================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
SUBJECT="all"
CONFIG="configs/sft_qlora.yaml"
OUTPUT_DIR="checkpoints/stage1_sft"
WANDB_PROJECT="jee-llm-finetuning"

# ── Parse arguments ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)      MODEL="$2";      shift 2 ;;
    --subject)    SUBJECT="$2";    shift 2 ;;
    --config)     CONFIG="$2";     shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Environment setup ─────────────────────────────────────────
echo "════════════════════════════════════════════════"
echo "  JEE-LLM QLoRA Training"
echo "  Model   : $MODEL"
echo "  Subject : $SUBJECT"
echo "  Config  : $CONFIG"
echo "  Output  : $OUTPUT_DIR"
echo "════════════════════════════════════════════════"

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
  echo "⚠️  WARNING: nvidia-smi not found. Training on CPU (very slow)."
else
  echo "GPU Info:"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi

# Activate conda env if available
if command -v conda &> /dev/null; then
  conda activate jee-llm 2>/dev/null || true
fi

# ── Pre-training checks ───────────────────────────────────────
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM: {vram:.1f} GB')
    if vram < 16:
        print('⚠️  WARNING: Less than 16GB VRAM. Consider reducing batch size or using CPU offload.')
"

# ── Run training ──────────────────────────────────────────────
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="$WANDB_PROJECT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python src/training/train_sft.py \
  --config "$CONFIG"

echo ""
echo "✅ QLoRA SFT training complete!"
echo "   Checkpoints saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  bash scripts/train_dpo.sh         # Stage 2c: DPO alignment"
echo "  python src/inference/gradio_app.py --model_path $OUTPUT_DIR/final  # Start UI"
