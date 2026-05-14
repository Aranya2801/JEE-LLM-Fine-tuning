#!/usr/bin/env bash
# =============================================================
# scripts/train_distributed.sh — Multi-GPU Distributed Training
# Uses DeepSpeed ZeRO-3 for maximum GPU efficiency
# =============================================================
# Usage:
#   bash scripts/train_distributed.sh --gpus 4 --model meta-llama/Meta-Llama-3-70B
# =============================================================

set -euo pipefail

GPUS=4
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
CONFIG="configs/sft_qlora.yaml"
DS_CONFIG="configs/deepspeed_zero3.json"
OUTPUT_DIR="checkpoints/stage1_sft_distributed"
MASTER_PORT=29500

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)       GPUS="$2";       shift 2 ;;
    --model)      MODEL="$2";      shift 2 ;;
    --config)     CONFIG="$2";     shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

echo "════════════════════════════════════════════════"
echo "  JEE-LLM Distributed Training (DeepSpeed ZeRO-3)"
echo "  GPUs    : $GPUS"
echo "  Model   : $MODEL"
echo "════════════════════════════════════════════════"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export WANDB_PROJECT="jee-llm-finetuning"

torchrun \
  --nproc_per_node="$GPUS" \
  --master_port="$MASTER_PORT" \
  src/training/train_sft.py \
  --config "$CONFIG" \
  --deepspeed "$DS_CONFIG" \
  --output_dir "$OUTPUT_DIR"

echo "✅ Distributed training complete! Output: $OUTPUT_DIR"
