"""End-to-end smoke test on a tiny random model: SFT -> eval -> GRPO.

Slow-ish (~20 s on CPU) and needs torch/trl, so it's skipped if they're missing.
Run just the fast tests with `pytest -m "not smoke"`.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("trl")

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = "dataset/samples/jee_style_samples.jsonl"
pytestmark = pytest.mark.smoke

CPU = [
    "model.dtype=float32", "training.bf16=false", "training.gradient_checkpointing=false",
    "training.report_to=none", "training.max_steps=2",
]


def run(*args):
    env = {"PYTHONPATH": str(ROOT / "src"), "WANDB_DISABLED": "true", "PATH": "/usr/bin:/bin"}
    import os

    env = {**os.environ, **env}
    r = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return r


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny")
    run("scripts/make_tiny_model.py", "--out", str(out))
    return out


def test_sft_eval_grpo(tiny, tmp_path):
    run("-m", "jee_llm.train_sft", "configs/sft_qlora.yaml", f"model.name={tiny}", f"output_dir={tmp_path}/sft",
        "quantization.load_in_4bit=false", f"data.train_files=[{SAMPLES}]", "data.eval_file=null",
        "training.optim=adamw_torch", "training.eval_strategy=no", "training.per_device_train_batch_size=2",
        "training.gradient_accumulation_steps=1", "training.max_length=512", *CPU)
    assert (tmp_path / "sft/final/adapter_config.json").exists()

    run("-m", "jee_llm.evaluate", "--model", str(tiny), "--adapter", f"{tmp_path}/sft/final", "--data", SAMPLES,
        "--num-samples", "2", "--temperature", "1.0", "--max-new-tokens", "16", "--out", f"{tmp_path}/eval")
    assert (tmp_path / "eval/report.md").exists()

    run("-m", "jee_llm.train_grpo", "configs/grpo.yaml", f"model.name={tiny}", f"output_dir={tmp_path}/grpo",
        f"data.train_files=[{SAMPLES}]", "training.use_vllm=false", "training.num_generations=2",
        "training.per_device_train_batch_size=2", "training.gradient_accumulation_steps=1",
        "training.max_completion_length=16", "training.log_completions=false", *CPU)
    assert (tmp_path / "grpo/final/adapter_config.json").exists()
