# Troubleshooting Guide — JEE-LLM

## Common Errors

---

### 🔴 `CUDA out of memory`

**Symptom:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate X GiB (GPU 0; Y GiB total capacity; Z GiB already allocated)
```

**Solutions (try in order):**

1. **Reduce batch size** in config:
   ```yaml
   training:
     per_device_train_batch_size: 2   # from 4
     gradient_accumulation_steps: 16  # keep effective batch same
   ```

2. **Reduce sequence length:**
   ```yaml
   training:
     max_seq_length: 2048   # from 4096
   ```

3. **Enable gradient checkpointing** (already default, verify):
   ```yaml
   training:
     gradient_checkpointing: true
   ```

4. **Reduce LoRA rank:**
   ```yaml
   lora:
     r: 32      # from 64
     lora_alpha: 64
   ```

5. **Use CPU optimizer offload** (DeepSpeed ZeRO-3):
   ```bash
   bash scripts/train_distributed.sh --gpus 1
   ```

---

### 🔴 `Flash Attention not installed`

```
ImportError: FlashAttention is not installed.
```

**Fix:**
```bash
pip install flash-attn --no-build-isolation
# If that fails (CUDA version mismatch):
pip install flash-attn --no-build-isolation --no-cache-dir
```

If still failing, disable it in config:
```yaml
model:
  attn_implementation: "eager"   # fallback
```

---

### 🔴 `ValueError: Tokenizer class ... does not exist`

```
ValueError: Tokenizer class LlamaTokenizer does not exist or is not currently imported.
```

**Fix:**
```bash
pip install --upgrade transformers tokenizers
```

---

### 🔴 `WandB authentication error`

```
wandb: ERROR: api_key not configured.
```

**Fix:**
```bash
wandb login
# Enter your API key from https://wandb.ai/authorize
```
Or set in `.env`:
```
WANDB_API_KEY=your_key_here
```

---

### 🔴 `HuggingFace 403 Forbidden`

```
OSError: We couldn't connect to 'https://huggingface.co'...
```

For gated models (LLaMA-3), accept the license on HuggingFace and:
```bash
huggingface-cli login
# Enter your HF token from https://huggingface.co/settings/tokens
```

Or set in `.env`:
```
HF_TOKEN=hf_your_token
```

---

### 🟡 Model generates garbage/repetitive output

**Symptoms:** Output is repetitive, off-topic, or just "..." repeated.

**Solutions:**

1. Verify the chat template is correct:
   ```python
   print(tokenizer.chat_template)
   # Should be non-empty for LLaMA-3
   ```

2. Increase repetition penalty:
   ```python
   model.generate(..., repetition_penalty=1.15)
   ```

3. Check the model was saved correctly:
   ```python
   from peft import PeftModel
   # Make sure merge_adapters.py was run before loading
   ```

4. Ensure you're using the merged model (not just adapter):
   ```bash
   python scripts/merge_adapters.py --base_model ... --adapter_path ...
   ```

---

### 🟡 Training loss not decreasing

**Symptoms:** Loss stays at ~3.0+ after 500+ steps.

**Solutions:**

1. **Increase learning rate:** Try `5e-4` if using warmup ratio correctly.

2. **Check data format:** Ensure prompts are being formatted correctly.
   ```python
   # Debug: print first 3 formatted examples
   python -c "
   from datasets import load_dataset
   ds = load_dataset('json', data_files='dataset/jee_cot_gpt4o.jsonl', split='train[:3]')
   print(ds[0])
   "
   ```

3. **Verify LoRA targets:** Some models use different attention module names.
   ```python
   from peft import get_peft_model, LoraConfig
   # Print all model module names
   for name, _ in model.named_modules():
       print(name)
   ```

---

### 🟡 `ImportError` on `latex2sympy2`

```
ImportError: No module named 'latex2sympy2'
```

**Fix:**
```bash
pip install latex2sympy2
pip install antlr4-python3-runtime==4.11.0
```

---

### 🟡 Gradio UI not loading LaTeX

LaTeX in Gradio requires `gr.Markdown` with `latex_delimiters`.

**Fix in gradio_app.py:**
```python
solution_out = gr.Markdown(
    latex_delimiters=[
        {"left": "$$", "right": "$$", "display": True},
        {"left": "$", "right": "$", "display": False},
        {"left": "\\[", "right": "\\]", "display": True},
    ]
)
```

---

### 🟡 Multi-GPU training hanging

```
NCCL error: Timeout
```

**Fix:**
```bash
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1   # If InfiniBand not available
bash scripts/train_distributed.sh --gpus 4
```

---

## Performance Tips

| Tip | Impact | Notes |
|-----|--------|-------|
| Enable Flash Attention 2 | +30% speed | Requires compatible GPU |
| Use `paged_adamw_32bit` | -30% VRAM | Default in config |
| Set `group_by_length=true` | -15% training time | Reduces padding |
| Enable `packing=true` | +20% throughput | For short sequences |
| Use `bf16` not `fp16` | Stable training | Requires Ampere+ GPU |
| Increase `dataloader_num_workers` | +10% throughput | Set to 4 |

---

## Getting Help

1. Search existing [GitHub Issues](https://github.com/Aranya2801/JEE-LLM-Fine-tuning/issues)
2. Open a new issue with the [Bug Report template](../.github/ISSUE_TEMPLATE/bug_report.md)
3. Include full error traceback, GPU info, and config file
