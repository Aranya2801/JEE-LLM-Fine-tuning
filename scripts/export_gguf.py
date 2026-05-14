"""
export_gguf.py — Export JEE-LLM to GGUF format for llama.cpp
=============================================================
Converts a merged HuggingFace model to GGUF format with
Q4_K_M quantization for efficient CPU/mobile inference.

Requires: llama.cpp installed and convert.py in PATH
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && make -j

Usage:
    python scripts/export_gguf.py \
        --model_path checkpoints/jee-llm-v1-merged \
        --output_path checkpoints/jee-llm-v1.Q4_K_M.gguf \
        --quantization Q4_K_M

Author: Aranya Ghosh
"""

import os
import sys
import shutil
import logging
import argparse
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")

SUPPORTED_QUANTS = [
    "Q4_K_M",   # Recommended: best quality/size trade-off
    "Q5_K_M",   # Higher quality, larger file
    "Q8_0",     # Near-lossless, 2× model size
    "Q4_0",     # Smallest, some quality loss
    "F16",      # Full 16-bit (large but lossless)
]


def find_llama_cpp() -> Optional[Path]:
    """Find llama.cpp installation."""
    # Check common locations
    candidates = [
        Path("llama.cpp"),
        Path("../llama.cpp"),
        Path(os.path.expanduser("~/llama.cpp")),
    ]
    for c in candidates:
        if (c / "convert_hf_to_gguf.py").exists():
            return c
        if (c / "convert.py").exists():
            return c
    return None


def export_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "Q4_K_M",
    llama_cpp_dir: Optional[str] = None,
) -> str:
    """
    Convert HuggingFace model to GGUF and quantize.

    Steps:
    1. Convert HF model to F16 GGUF using llama.cpp/convert.py
    2. Quantize to target quantization level
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if quantization not in SUPPORTED_QUANTS:
        raise ValueError(
            f"Unsupported quantization '{quantization}'. "
            f"Choose from: {SUPPORTED_QUANTS}"
        )

    # Find llama.cpp
    if llama_cpp_dir:
        llama_dir = Path(llama_cpp_dir)
    else:
        llama_dir = find_llama_cpp()

    if llama_dir is None:
        logger.error(
            "llama.cpp not found. Please clone and build it:\n"
            "  git clone https://github.com/ggerganov/llama.cpp\n"
            "  cd llama.cpp && make -j\n"
            "Then pass --llama_cpp_dir /path/to/llama.cpp"
        )
        sys.exit(1)

    logger.info(f"Using llama.cpp from: {llama_dir}")

    # Step 1: Convert to F16 GGUF
    f16_path = output_path.parent / "jee_llm_f16.gguf"
    convert_script = llama_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        convert_script = llama_dir / "convert.py"

    logger.info(f"Step 1: Converting {model_path} to F16 GGUF...")
    cmd_convert = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outfile", str(f16_path),
        "--outtype", "f16",
    ]
    logger.info(f"Running: {' '.join(cmd_convert)}")
    result = subprocess.run(cmd_convert, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Conversion failed:\n{result.stderr}")
        sys.exit(1)
    logger.info(f"F16 GGUF saved to: {f16_path}")

    # Step 2: Quantize
    if quantization != "F16":
        quantize_bin = llama_dir / "quantize"
        if not quantize_bin.exists():
            quantize_bin = llama_dir / "build" / "bin" / "quantize"

        if not quantize_bin.exists():
            logger.error(
                f"quantize binary not found in {llama_dir}. "
                "Build llama.cpp with: cd llama.cpp && make -j"
            )
            sys.exit(1)

        logger.info(f"Step 2: Quantizing to {quantization}...")
        cmd_quant = [
            str(quantize_bin),
            str(f16_path),
            str(output_path),
            quantization,
        ]
        logger.info(f"Running: {' '.join(cmd_quant)}")
        result = subprocess.run(cmd_quant, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Quantization failed:\n{result.stderr}")
            sys.exit(1)

        # Cleanup intermediate F16 file
        if output_path != f16_path:
            f16_path.unlink(missing_ok=True)
    else:
        shutil.move(str(f16_path), str(output_path))

    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"✅ GGUF model saved to: {output_path}")
    logger.info(f"   File size: {size_mb:.1f} MB")

    # Print usage instructions
    print(f"""
┌─────────────────────────────────────────────────────────────┐
│  ✅ GGUF Export Complete                                     │
├─────────────────────────────────────────────────────────────┤
│  File: {str(output_path):<53}│
│  Size: {size_mb:.1f} MB{' '*(52-len(f'{size_mb:.1f} MB'))}│
├─────────────────────────────────────────────────────────────┤
│  Run with llama.cpp:                                        │
│  ./main -m {str(output_path.name):<47}│
│    -p "Solve: ∫₀^π x·sin(x)dx"                            │
│    -n 1024 --temp 0.7                                       │
│                                                             │
│  Run with Ollama:                                           │
│  ollama create jee-llm -f Modelfile                         │
│  ollama run jee-llm                                         │
└─────────────────────────────────────────────────────────────┘
""")
    return str(output_path)


def create_ollama_modelfile(gguf_path: str, output_dir: str):
    """Create an Ollama Modelfile for easy deployment."""
    modelfile = f"""FROM {gguf_path}

SYSTEM You are JEE-LLM, an expert IIT-JEE Advanced tutor specializing in Mathematics, Physics, and Chemistry. You solve problems step-by-step with complete mathematical rigor. Always present the final answer inside \\boxed{{}} for clarity.

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 1024
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
"""
    path = Path(output_dir) / "Modelfile"
    path.write_text(modelfile)
    logger.info(f"Ollama Modelfile saved to: {path}")
    logger.info("To create Ollama model: ollama create jee-llm -f Modelfile")
    return str(path)


if __name__ == "__main__":
    from typing import Optional

    parser = argparse.ArgumentParser(description="Export JEE-LLM to GGUF")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to merged HuggingFace model directory",
    )
    parser.add_argument(
        "--output_path", type=str,
        default="checkpoints/jee-llm-v1.Q4_K_M.gguf",
    )
    parser.add_argument(
        "--quantization", type=str, default="Q4_K_M",
        choices=SUPPORTED_QUANTS,
    )
    parser.add_argument("--llama_cpp_dir", type=str, default=None)
    parser.add_argument(
        "--create_modelfile", action="store_true",
        help="Also create an Ollama Modelfile",
    )
    args = parser.parse_args()

    gguf_path = export_to_gguf(
        model_path=args.model_path,
        output_path=args.output_path,
        quantization=args.quantization,
        llama_cpp_dir=args.llama_cpp_dir,
    )

    if args.create_modelfile:
        create_ollama_modelfile(
            gguf_path=gguf_path,
            output_dir=str(Path(args.output_path).parent),
        )
