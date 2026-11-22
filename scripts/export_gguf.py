"""Convert a merged HF checkpoint to GGUF for llama.cpp / Ollama.

Needs a local llama.cpp checkout that has been built:
    git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
    cmake -B build && cmake --build build --config Release -j

    python scripts/export_gguf.py --model checkpoints/grpo-merged --llama-cpp ~/llama.cpp --quant Q4_K_M
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

QUANTS = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]


def find_quantize(llama_cpp: Path) -> Path:
    # binary was renamed from `quantize` to `llama-quantize` in 2024
    for cand in ("build/bin/llama-quantize", "llama-quantize", "build/bin/quantize", "quantize"):
        if (llama_cpp / cand).exists():
            return llama_cpp / cand
    raise SystemExit(f"no quantize binary under {llama_cpp}, did you build llama.cpp?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged HF model dir")
    ap.add_argument("--llama-cpp", required=True, type=Path)
    ap.add_argument("--quant", default="Q4_K_M", choices=QUANTS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = Path(args.model)
    out = Path(args.out) if args.out else model.with_name(f"{model.name}.{args.quant}.gguf")
    convert = args.llama_cpp / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise SystemExit(f"{convert} not found")

    f16 = out.with_name(out.stem.replace(args.quant, "F16") + ".gguf") if args.quant != "F16" else out
    subprocess.run([sys.executable, str(convert), str(model), "--outfile", str(f16), "--outtype", "f16"], check=True)
    if args.quant != "F16":
        subprocess.run([str(find_quantize(args.llama_cpp)), str(f16), str(out), args.quant], check=True)
        f16.unlink()
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
